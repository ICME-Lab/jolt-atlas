use ark_bn254::Fr;
use ark_ff::One;
use itertools::Itertools;
use joltworks::field::JoltField;
use joltworks::transcripts::Blake2bTranscript;
use qwen3_common::{
    IopLayerProof, LayerRmsNormVerifierInput, LayerRopeVerifierInput, LayerSiluVerifierInput,
    LayerSoftmaxVerifierInput, LayerVerifierPublicInput, qwen3_layer_shape,
};
use qwen3_prover::{
    layer::EvalClaim,
    layer::{
        LayerLookupRanges, LayerOpeningWitnesses, LayerProverInput, LayerShape, prove_iop_layer,
    },
    layer_input::{LayerRawWitness, LayerWeights, layer_prover_input},
    opening::prove_layer_opening_reduction_sumcheck,
    ops::{
        add::{AddParams, AddProverInput, AddWitness},
        matmul::{MatMulParams, MatMulProverInput, MatMulWitness},
        mul::{MulParams, MulProverInput, MulWitness},
        pv_matmul::{PvMatmulParams, PvMatmulProverInput, PvMatmulWitness},
        qk_score::{QkScoreParams, QkScoreProverInput, QkScoreWitness},
        rms_norm::{RmsNormAdvice, RmsNormParams, RmsNormProverInput, RmsNormWitness},
        rope::{RopeParams, RopeProverInput, RopeWitness},
        silu::{SiluAdvice, SiluParams, SiluProverInput, SiluWitness},
        softmax::{SoftmaxAdvice, SoftmaxParams, SoftmaxProverInput, SoftmaxWitness},
    },
};
use qwen3_verifier::{verify_iop_layer, verify_layer_opening_reduction};

fn layer_verifier_public_input(input: LayerProverInput) -> LayerVerifierPublicInput {
    LayerVerifierPublicInput {
        seq: input.shape.seq,
        down_proj_weight: input.down_proj.witness.rhs,
        silu: LayerSiluVerifierInput {
            advice: input.silu.advice,
        },
        gate_proj_weight: input.gate_proj.witness.rhs,
        up_proj_weight: input.up_proj.witness.rhs,
        rms_norm_mlp: LayerRmsNormVerifierInput {
            advice: input.rms_norm_mlp.advice,
            weight: input.rms_norm_mlp.witness.weight,
        },
        o_proj_weight: input.o_proj.witness.rhs,
        softmax: LayerSoftmaxVerifierInput {
            advice: input.softmax.advice,
        },
        q_rope: LayerRopeVerifierInput {
            cos: input.q_rope.witness.cos,
            sin: input.q_rope.witness.sin,
        },
        k_rope: LayerRopeVerifierInput {
            cos: input.k_rope.witness.cos,
            sin: input.k_rope.witness.sin,
        },
        q_norm: LayerRmsNormVerifierInput {
            advice: input.q_norm.advice,
            weight: input.q_norm.witness.weight,
        },
        k_norm: LayerRmsNormVerifierInput {
            advice: input.k_norm.advice,
            weight: input.k_norm.witness.weight,
        },
        q_proj_weight: input.q_proj.witness.rhs,
        k_proj_weight: input.k_proj.witness.rhs,
        v_proj_weight: input.v_proj.witness.rhs,
        rms_norm_atten: LayerRmsNormVerifierInput {
            advice: input.rms_norm_atten.advice,
            weight: input.rms_norm_atten.witness.weight,
        },
    }
}

fn iop_layer_proof(output: qwen3_prover::layer::IopLayerOutput) -> IopLayerProof {
    output.proof
}

const FRAC_BITS: usize = 8;

fn eq_eval(index: usize, point: &[Fr]) -> Fr {
    point
        .iter()
        .enumerate()
        .map(|(bit, point)| {
            if ((index >> bit) & 1) == 1 {
                *point
            } else {
                Fr::one() - point
            }
        })
        .product()
}

fn eval_i32(values: &[i32], point: &[Fr]) -> Fr {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| eq_eval(index, point) * Fr::from_i32(*value))
        .sum()
}

fn round_q8(value: i64) -> i32 {
    let rem = value.rem_euclid(256);
    ((value + ((rem >> 7) * 256) - rem) / 256) as i32
}

fn floor_q8(value: i64) -> i32 {
    value.div_euclid(256) as i32
}

fn inv_sum_q16(sum: i32) -> i32 {
    ((1_i64 << 24) as f64 / f64::from(sum)).round() as i32
}

fn bool_bits(values: &[i64]) -> [Vec<bool>; FRAC_BITS] {
    std::array::from_fn(|bit| {
        values
            .iter()
            .map(|value| ((value.rem_euclid(256) >> bit) & 1) == 1)
            .collect()
    })
}

fn zero_bits(len: usize) -> [Vec<bool>; FRAC_BITS] {
    std::array::from_fn(|_| vec![false; len])
}

fn expanded_zero_onehot(rows: usize, entries: usize) -> Vec<u8> {
    let mut values = vec![0_u8; rows * entries];
    for row in 0..rows {
        values[row] = 1;
    }
    values
}

fn zero_u8_bits(len: usize) -> [Vec<u8>; FRAC_BITS] {
    std::array::from_fn(|_| vec![0_u8; len])
}

fn u8_bits(bits: [Vec<bool>; FRAC_BITS]) -> [Vec<u8>; FRAC_BITS] {
    bits.map(|bit| bit.into_iter().map(u8::from).collect())
}

fn zeros(len: usize) -> Vec<i32> {
    vec![0_i32; len]
}

fn zero_matmul(rows: usize, cols: usize, inner: usize) -> MatMulProverInput {
    MatMulProverInput {
        params: MatMulParams::new(rows, cols, inner).unwrap(),
        witness: MatMulWitness {
            lhs: zeros(rows * inner),
            rhs: zeros(inner * cols),
            output: zeros(rows * cols),
            output_remainder: vec![0_u8; rows * cols],
        },
    }
}

fn zero_add(rows: usize, cols: usize) -> AddProverInput {
    AddProverInput {
        params: AddParams::new(rows, cols).unwrap(),
        witness: AddWitness {
            lhs: zeros(rows * cols),
            rhs: zeros(rows * cols),
        },
    }
}

fn zero_rms_norm(rows: usize, cols: usize) -> RmsNormProverInput {
    RmsNormProverInput {
        params: RmsNormParams::new(rows, cols).unwrap(),
        advice: RmsNormAdvice {
            sum_x2: vec![0_i64; rows],
        },
        witness: RmsNormWitness {
            input: zeros(rows * cols),
            inv_rms: zeros(rows * cols),
            norm: zeros(rows * cols),
            weight: zeros(rows * cols),
            output: zeros(rows * cols),
            norm_remainder_bits: zero_bits(rows * cols),
            output_remainder_bits: zero_bits(rows * cols),
        },
    }
}

fn zero_rope(seq: usize, heads: usize, head_dim: usize) -> RopeProverInput {
    RopeProverInput {
        params: RopeParams::new(seq, heads, head_dim).unwrap(),
        witness: RopeWitness {
            input: zeros(seq * heads * head_dim),
            output: zeros(seq * heads * head_dim),
            output_remainder_bits: zero_bits(seq * heads * head_dim),
            cos: vec![256_i32; seq * (head_dim / 2)],
            sin: zeros(seq * (head_dim / 2)),
        },
    }
}

fn softmax_witness(rows: usize, cols: usize) -> SoftmaxProverInput {
    let mut valid = vec![0_i32; rows * cols];
    for row in 0..rows {
        let query_pos = row % cols;
        for col in 0..cols {
            valid[col * rows + row] = i32::from(col <= query_pos);
        }
    }

    let exp = vec![256_i32; rows * cols];
    let sum = (0..rows)
        .map(|row| {
            (0..cols)
                .map(|col| exp[col * rows + row] * valid[col * rows + row])
                .sum::<i32>()
        })
        .collect::<Vec<_>>();
    let mut coefficient = vec![0_i32; rows * cols];
    for row in 0..rows {
        let row_coefficient = inv_sum_q16(sum[row]);
        for col in 0..cols {
            coefficient[col * rows + row] = row_coefficient * valid[col * rows + row];
        }
    }
    let acc = coefficient
        .iter()
        .zip_eq(&exp)
        .map(|(coefficient, exp)| i64::from(*coefficient) * i64::from(*exp))
        .collect::<Vec<_>>();
    let floor = acc.iter().map(|value| floor_q8(*value)).collect::<Vec<_>>();
    let output = floor
        .iter()
        .map(|value| round_q8(i64::from(*value)))
        .collect::<Vec<_>>();

    SoftmaxProverInput {
        params: SoftmaxParams::new(rows, cols).unwrap(),
        advice: SoftmaxAdvice {
            min_diff: 0,
            max_diff: 0,
            row_max: zeros(rows),
            max_index: vec![0; rows],
            sum,
        },
        witness: SoftmaxWitness {
            input: zeros(rows * cols),
            output,
            ra: vec![1_u8; rows * cols],
            exp_acc: vec![256_i64 * 256_i64; rows * cols],
            exp,
            frac_bits: zero_bits(rows * cols),
            exp_remainder_bits: zero_bits(rows * cols),
            acc: acc.clone(),
            floor: floor.clone(),
            floor_remainder_bits: bool_bits(&acc),
            output_remainder_bits: bool_bits(
                &floor
                    .iter()
                    .map(|value| i64::from(*value))
                    .collect::<Vec<_>>(),
            ),
        },
    }
}

fn minimal_layer_input(
    seq: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    hidden: usize,
    intermediate: usize,
) -> LayerProverInput {
    let intermediate_len = seq * intermediate;
    let q_len = seq * q_heads * head_dim;
    let kv_len = seq * kv_heads * head_dim;
    let score_len = q_heads * seq * seq;
    LayerProverInput {
        shape: LayerShape {
            seq,
            q_heads,
            kv_heads,
            head_dim,
            hidden,
            intermediate,
        },
        opening_witnesses: zero_layer_opening_witnesses(
            seq,
            q_heads,
            kv_heads,
            head_dim,
            hidden,
            intermediate,
        ),
        residual_add_mlp: zero_add(seq, hidden),
        down_proj: zero_matmul(seq, hidden, intermediate),
        silu_up: MulProverInput {
            params: MulParams::matrix(seq, intermediate).unwrap(),
            witness: MulWitness {
                a: zeros(intermediate_len),
                b: zeros(intermediate_len),
                bits: zero_bits(intermediate_len),
            },
        },
        silu: SiluProverInput {
            params: SiluParams::new(seq, intermediate).unwrap(),
            advice: SiluAdvice { min_n: 0, max_n: 0 },
            witness: SiluWitness {
                input: zeros(intermediate_len),
                output: zeros(intermediate_len),
                ra: vec![1_u8; intermediate_len],
                input_remainder_bits: zero_bits(intermediate_len),
                output_remainder_bits: zero_bits(intermediate_len),
            },
        },
        gate_proj: zero_matmul(seq, intermediate, hidden),
        up_proj: zero_matmul(seq, intermediate, hidden),
        rms_norm_mlp: zero_rms_norm(seq, hidden),
        residual_add_attn: zero_add(seq, hidden),
        o_proj: zero_matmul(seq, hidden, q_heads * head_dim),
        pv_matmul: PvMatmulProverInput {
            params: PvMatmulParams::new(seq, q_heads, kv_heads, head_dim).unwrap(),
            witness: PvMatmulWitness {
                p: softmax_witness(q_heads * seq, seq).witness.output,
                v: zeros(kv_len),
                context_remainder_bits: zero_bits(q_len),
            },
        },
        softmax: softmax_witness(q_heads * seq, seq),
        qk_score: QkScoreProverInput {
            params: QkScoreParams::new(seq, q_heads, kv_heads, head_dim).unwrap(),
            witness: QkScoreWitness {
                q: zeros(q_len),
                k: zeros(kv_len),
                dot: zeros(score_len),
                score_remainder_bits: zero_bits(score_len),
                dot_remainder_bits: zero_bits(score_len),
            },
        },
        q_rope: zero_rope(seq, q_heads, head_dim),
        k_rope: zero_rope(seq, kv_heads, head_dim),
        q_norm: zero_rms_norm(seq * q_heads, head_dim),
        k_norm: zero_rms_norm(seq * kv_heads, head_dim),
        q_proj: zero_matmul(seq, q_heads * head_dim, hidden),
        k_proj: zero_matmul(seq, kv_heads * head_dim, hidden),
        v_proj: zero_matmul(seq, kv_heads * head_dim, hidden),
        rms_norm_atten: zero_rms_norm(seq, hidden),
    }
}

fn zero_layer_opening_witnesses(
    seq: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    hidden: usize,
    intermediate: usize,
) -> LayerOpeningWitnesses {
    let hidden_len = seq * hidden;
    let intermediate_len = seq * intermediate;
    let q_len = seq * q_heads * head_dim;
    let kv_len = seq * kv_heads * head_dim;
    let score_len = q_heads * seq * seq;
    LayerOpeningWitnesses {
        hidden_out: zeros(hidden_len),
        hidden_in_a: zeros(hidden_len),
        hidden_in_b: zeros(hidden_len),
        silu_lookup_ra: expanded_zero_onehot(intermediate_len, 16),
        softmax_lookup_ra: expanded_zero_onehot(score_len, 2),
        down_proj_output_frac_bits: zero_bits(hidden_len),
        silu_up_output_frac_bits: zero_bits(intermediate_len),
        silu_input_frac_bits: zero_bits(intermediate_len),
        silu_output_frac_bits: zero_bits(intermediate_len),
        gate_proj_output_frac_bits: zero_bits(intermediate_len),
        up_proj_output_frac_bits: zero_bits(intermediate_len),
        rms_norm_mlp_norm_frac_bits: zero_bits(hidden_len),
        rms_norm_mlp_output_frac_bits: zero_bits(hidden_len),
        o_proj_output_frac_bits: zero_bits(hidden_len),
        pv_matmul_output_frac_bits: zero_bits(q_len),
        softmax_floor_frac_bits: zero_bits(score_len),
        softmax_output_frac_bits: zero_bits(score_len),
        softmax_exp_frac_bits: zero_bits(score_len),
        qk_score_dot_output_frac_bits: zero_bits(score_len),
        qk_score_output_frac_bits: zero_bits(score_len),
        q_rope_output_frac_bits: zero_bits(q_len),
        k_rope_output_frac_bits: zero_bits(kv_len),
        q_norm_norm_frac_bits: zero_bits(q_len),
        q_norm_output_frac_bits: zero_bits(q_len),
        k_norm_norm_frac_bits: zero_bits(kv_len),
        k_norm_output_frac_bits: zero_bits(kv_len),
        q_proj_output_frac_bits: zero_bits(q_len),
        k_proj_output_frac_bits: zero_bits(kv_len),
        v_proj_output_frac_bits: zero_bits(kv_len),
        rms_norm_atten_norm_frac_bits: zero_bits(hidden_len),
        rms_norm_atten_output_frac_bits: zero_bits(hidden_len),
    }
}

#[test]
fn proves_minimal_layer() {
    let shape = qwen3_layer_shape(1).expect("qwen shape");
    let seq = shape.seq;
    let q_heads = shape.q_heads;
    let kv_heads = shape.kv_heads;
    let head_dim = shape.head_dim;
    let hidden = shape.hidden;
    let intermediate = shape.intermediate;
    let hidden_len = seq * hidden;

    let hidden_point = vec![Fr::from(2_u64); hidden_len.ilog2() as usize];
    let hidden_point_len = hidden_point.len();
    let hidden_out = zeros(hidden_len);
    let hidden_out_claim = EvalClaim {
        value: eval_i32(&hidden_out, &hidden_point),
        point: hidden_point,
    };

    let mut transcript = Blake2bTranscript::default();
    let output = prove_iop_layer(
        hidden_out_claim.value,
        minimal_layer_input(seq, q_heads, kv_heads, head_dim, hidden, intermediate),
        &mut transcript,
    )
    .expect("minimal consistent layer proves");

    assert_eq!(
        output.opening_claims.hidden_in_a.point.len(),
        hidden_point_len
    );
    assert_eq!(
        output.opening_claims.hidden_in_b.point.len(),
        hidden_point_len
    );

    let mut opening_transcript = Blake2bTranscript::default();
    let opening = prove_layer_opening_reduction_sumcheck(
        &output.opening_claims,
        &output.opening_witnesses,
        shape,
        &mut opening_transcript,
    )
    .expect("layer opening reduction proves");

    let mut opening_verifier_transcript = Blake2bTranscript::default();
    let verified_opening = verify_layer_opening_reduction(
        &output.opening_claims,
        shape,
        &opening.proof,
        &mut opening_verifier_transcript,
    )
    .expect("layer opening reduction verifies");
    assert_eq!(opening.reduction_point, verified_opening.sumcheck.point);
    assert_eq!(opening.gamma_powers, verified_opening.gamma_powers);

    let mut verifier_transcript = Blake2bTranscript::default();
    let verify_input = layer_verifier_public_input(minimal_layer_input(
        seq,
        q_heads,
        kv_heads,
        head_dim,
        hidden,
        intermediate,
    ));
    verify_iop_layer(
        verify_input,
        iop_layer_proof(output),
        &mut verifier_transcript,
    )
    .expect("minimal consistent layer verifies");
}

#[test]
fn computes_layer_opening_domain_lengths() {
    let lengths = LayerShape {
        seq: 2,
        q_heads: 2,
        kv_heads: 1,
        head_dim: 2,
        hidden: 4,
        intermediate: 2,
    }
    .opening_domain_lengths(LayerLookupRanges {
        silu_min_n: 0,
        silu_max_n: 3,
        softmax_min_diff: -2,
        softmax_max_diff: 1,
    })
    .expect("valid opening domain lengths");

    assert_eq!(lengths.hidden_out, 8);
    assert_eq!(lengths.silu_lookup_ra, 64);
    assert_eq!(lengths.softmax_lookup_ra, 32);
    assert_eq!(lengths.q_rope_output_frac_bits, 8);
    assert_eq!(lengths.k_rope_output_frac_bits, 4);
    assert_eq!(lengths.max().name, "silu_lookup_ra");
    assert_eq!(lengths.max().len, 64);
}

#[test]
fn converts_raw_layer_input() {
    let seq = 2;
    let q_heads = 2;
    let kv_heads = 1;
    let head_dim = 2;
    let hidden = 4;
    let intermediate = 2;
    let hidden_len = seq * hidden;
    let intermediate_len = seq * intermediate;
    let q_len = seq * q_heads * head_dim;
    let kv_len = seq * kv_heads * head_dim;
    let score_len = q_heads * seq * seq;
    let softmax = softmax_witness(q_heads * seq, seq);
    let softmax_advice = softmax.advice;
    let softmax_witness = softmax.witness;

    let hidden_out = zeros(hidden_len);
    let hidden_point = vec![Fr::from(2_u64), Fr::from(5_u64), Fr::from(7_u64)];
    let hidden_out_claim = EvalClaim {
        value: eval_i32(&hidden_out, &hidden_point),
        point: hidden_point,
    };

    let input = layer_prover_input(
        qwen3_prover::layer::LayerShape {
            seq,
            q_heads,
            kv_heads,
            head_dim,
            hidden,
            intermediate,
        },
        LayerWeights {
            rope_cos: vec![256_i32; seq * (head_dim / 2)],
            rope_sin: zeros(seq * (head_dim / 2)),
            rms_norm_atten: zeros(hidden),
            q_norm: zeros(head_dim),
            k_norm: zeros(head_dim),
            rms_norm_mlp: zeros(hidden),
            o_proj: zeros(q_heads * head_dim * hidden),
            q_proj: zeros(hidden * q_len / seq),
            k_proj: zeros(hidden * kv_len / seq),
            v_proj: zeros(hidden * kv_len / seq),
            gate_proj: zeros(hidden * intermediate),
            up_proj: zeros(hidden * intermediate),
            down_proj: zeros(intermediate * hidden),
        },
        LayerRawWitness {
            hidden_in: zeros(hidden_len),
            hidden_out,
            rms_norm_atten_sum_x2: vec![0_i64; seq],
            rms_norm_atten_norm: zeros(hidden_len),
            rms_norm_atten_norm_frac_bits: zero_u8_bits(hidden_len),
            rms_norm_atten_a: zeros(hidden_len),
            rms_norm_atten_b: zeros(hidden_len),
            rms_norm_atten_c: zeros(hidden_len),
            rms_norm_atten_output_frac_bits: zero_u8_bits(hidden_len),
            context: zeros(q_len),
            pv_matmul_output_frac_bits: zero_u8_bits(q_len),
            o_proj: zeros(hidden_len),
            o_proj_output_frac_bits: zero_u8_bits(hidden_len),
            softmax: softmax_witness.output,
            softmax_acc: softmax_witness.acc,
            softmax_floor: softmax_witness.floor,
            softmax_floor_frac_bits: u8_bits(softmax_witness.floor_remainder_bits),
            softmax_output_frac_bits: u8_bits(softmax_witness.output_remainder_bits),
            softmax_min_diff: softmax_advice.min_diff,
            softmax_max_diff: softmax_advice.max_diff,
            softmax_lookup_ra: softmax_witness.ra,
            softmax_exp_acc: softmax_witness.exp_acc,
            softmax_exp_frac_bits: u8_bits(softmax_witness.exp_remainder_bits),
            qk_score: zeros(score_len),
            qk_score_dot: zeros(score_len),
            qk_score_dot_output_frac_bits: zero_u8_bits(score_len),
            qk_score_output_frac_bits: zero_u8_bits(score_len),
            q_rope: zeros(q_len),
            q_rope_output_frac_bits: zero_u8_bits(q_len),
            k_rope: zeros(kv_len),
            k_rope_output_frac_bits: zero_u8_bits(kv_len),
            q_proj: zeros(q_len),
            k_proj: zeros(kv_len),
            q_norm_sum_x2: vec![0_i64; seq * q_heads],
            k_norm_sum_x2: vec![0_i64; seq * kv_heads],
            q_norm_norm: zeros(q_len),
            k_norm_norm: zeros(kv_len),
            q_norm_norm_frac_bits: zero_u8_bits(q_len),
            k_norm_norm_frac_bits: zero_u8_bits(kv_len),
            q_norm: zeros(q_len),
            k_norm: zeros(kv_len),
            q_norm_output_frac_bits: zero_u8_bits(q_len),
            k_norm_output_frac_bits: zero_u8_bits(kv_len),
            q_proj_output_frac_bits: zero_u8_bits(q_len),
            k_proj_output_frac_bits: zero_u8_bits(kv_len),
            v_proj_output_frac_bits: zero_u8_bits(kv_len),
            softmax_max: softmax_advice.row_max,
            softmax_max_index: softmax_advice.max_index,
            softmax_exp: softmax_witness.exp,
            softmax_sum: softmax_advice.sum,
            v_proj: zeros(kv_len),
            residual_add_attn_a: zeros(hidden_len),
            residual_add_attn_b: zeros(hidden_len),
            rms_norm_mlp_sum_x2: vec![0_i64; seq],
            rms_norm_mlp_norm: zeros(hidden_len),
            rms_norm_mlp_norm_frac_bits: zero_u8_bits(hidden_len),
            rms_norm_mlp_a: zeros(hidden_len),
            rms_norm_mlp_b: zeros(hidden_len),
            rms_norm_mlp_output_frac_bits: zero_u8_bits(hidden_len),
            gate_proj: zeros(intermediate_len),
            gate_proj_output_frac_bits: zero_u8_bits(intermediate_len),
            silu: zeros(intermediate_len),
            silu_min_n: 0,
            silu_max_n: 0,
            silu_input_frac_bits: zero_u8_bits(intermediate_len),
            silu_lookup_ra: vec![1_u8; intermediate_len],
            silu_output_frac_bits: zero_u8_bits(intermediate_len),
            silu_up: zeros(intermediate_len),
            silu_up_output_frac_bits: zero_u8_bits(intermediate_len),
            up_proj: zeros(intermediate_len),
            up_proj_output_frac_bits: zero_u8_bits(intermediate_len),
            down_proj: zeros(hidden_len),
            down_proj_output_frac_bits: zero_u8_bits(hidden_len),
        },
    )
    .expect("raw layer input converts");

    let mut transcript = Blake2bTranscript::default();
    prove_iop_layer(hidden_out_claim.value, input, &mut transcript)
        .expect("converted layer input proves");
}

#[test]
fn pads_raw_layer_input_to_power_of_two_domain() {
    let seq = 3;
    let q_heads = 2;
    let kv_heads = 1;
    let head_dim = 2;
    let hidden = 4;
    let intermediate = 3;
    let hidden_len = seq * hidden;
    let intermediate_len = seq * intermediate;
    let q_len = seq * q_heads * head_dim;
    let kv_len = seq * kv_heads * head_dim;
    let score_len = q_heads * seq * seq;

    let input = layer_prover_input(
        qwen3_prover::layer::LayerShape {
            seq,
            q_heads,
            kv_heads,
            head_dim,
            hidden,
            intermediate,
        },
        LayerWeights {
            rope_cos: vec![256_i32; seq * (head_dim / 2)],
            rope_sin: zeros(seq * (head_dim / 2)),
            rms_norm_atten: zeros(hidden),
            q_norm: zeros(head_dim),
            k_norm: zeros(head_dim),
            rms_norm_mlp: zeros(hidden),
            o_proj: zeros(q_heads * head_dim * hidden),
            q_proj: zeros(hidden * q_heads * head_dim),
            k_proj: zeros(hidden * kv_heads * head_dim),
            v_proj: zeros(hidden * kv_heads * head_dim),
            gate_proj: zeros(hidden * intermediate),
            up_proj: zeros(hidden * intermediate),
            down_proj: zeros(intermediate * hidden),
        },
        LayerRawWitness {
            hidden_in: zeros(hidden_len),
            hidden_out: zeros(hidden_len),
            rms_norm_atten_sum_x2: vec![0_i64; seq],
            rms_norm_atten_norm: zeros(hidden_len),
            rms_norm_atten_norm_frac_bits: zero_u8_bits(hidden_len),
            rms_norm_atten_a: zeros(hidden_len),
            rms_norm_atten_b: zeros(hidden_len),
            rms_norm_atten_c: zeros(hidden_len),
            rms_norm_atten_output_frac_bits: zero_u8_bits(hidden_len),
            context: zeros(q_len),
            pv_matmul_output_frac_bits: zero_u8_bits(q_len),
            o_proj: zeros(hidden_len),
            o_proj_output_frac_bits: zero_u8_bits(hidden_len),
            softmax: zeros(score_len),
            softmax_acc: vec![0_i64; score_len],
            softmax_floor: zeros(score_len),
            softmax_floor_frac_bits: zero_u8_bits(score_len),
            softmax_output_frac_bits: zero_u8_bits(score_len),
            softmax_min_diff: 0,
            softmax_max_diff: 0,
            softmax_lookup_ra: vec![1_u8; score_len],
            softmax_exp_acc: vec![256_i64 * 256_i64; score_len],
            softmax_exp_frac_bits: zero_u8_bits(score_len),
            qk_score: zeros(score_len),
            qk_score_dot: zeros(score_len),
            qk_score_dot_output_frac_bits: zero_u8_bits(score_len),
            qk_score_output_frac_bits: zero_u8_bits(score_len),
            q_rope: zeros(q_len),
            q_rope_output_frac_bits: zero_u8_bits(q_len),
            k_rope: zeros(kv_len),
            k_rope_output_frac_bits: zero_u8_bits(kv_len),
            q_proj: zeros(q_len),
            k_proj: zeros(kv_len),
            q_norm_sum_x2: vec![0_i64; seq * q_heads],
            k_norm_sum_x2: vec![0_i64; seq * kv_heads],
            q_norm_norm: zeros(q_len),
            k_norm_norm: zeros(kv_len),
            q_norm_norm_frac_bits: zero_u8_bits(q_len),
            k_norm_norm_frac_bits: zero_u8_bits(kv_len),
            q_norm: zeros(q_len),
            k_norm: zeros(kv_len),
            q_norm_output_frac_bits: zero_u8_bits(q_len),
            k_norm_output_frac_bits: zero_u8_bits(kv_len),
            q_proj_output_frac_bits: zero_u8_bits(q_len),
            k_proj_output_frac_bits: zero_u8_bits(kv_len),
            v_proj_output_frac_bits: zero_u8_bits(kv_len),
            softmax_max: zeros(q_heads * seq),
            softmax_max_index: vec![0; q_heads * seq],
            softmax_exp: vec![256_i32; score_len],
            softmax_sum: zeros(q_heads * seq),
            v_proj: zeros(kv_len),
            residual_add_attn_a: zeros(hidden_len),
            residual_add_attn_b: zeros(hidden_len),
            rms_norm_mlp_sum_x2: vec![0_i64; seq],
            rms_norm_mlp_norm: zeros(hidden_len),
            rms_norm_mlp_norm_frac_bits: zero_u8_bits(hidden_len),
            rms_norm_mlp_a: zeros(hidden_len),
            rms_norm_mlp_b: zeros(hidden_len),
            rms_norm_mlp_output_frac_bits: zero_u8_bits(hidden_len),
            gate_proj: zeros(intermediate_len),
            gate_proj_output_frac_bits: zero_u8_bits(intermediate_len),
            silu: zeros(intermediate_len),
            silu_min_n: 0,
            silu_max_n: 0,
            silu_input_frac_bits: zero_u8_bits(intermediate_len),
            silu_lookup_ra: vec![1_u8; intermediate_len],
            silu_output_frac_bits: zero_u8_bits(intermediate_len),
            silu_up: zeros(intermediate_len),
            silu_up_output_frac_bits: zero_u8_bits(intermediate_len),
            up_proj: zeros(intermediate_len),
            up_proj_output_frac_bits: zero_u8_bits(intermediate_len),
            down_proj: zeros(hidden_len),
            down_proj_output_frac_bits: zero_u8_bits(hidden_len),
        },
    )
    .expect("non-power-of-two logical input pads");

    assert_eq!(input.down_proj.params.output_shape.rows, 4);
    assert_eq!(input.down_proj.params.inner, 4);
    assert_eq!(input.silu.params.shape.rows, 4);
    assert_eq!(input.silu.params.shape.cols, 4);
    assert_eq!(input.softmax.params.shape.rows, 8);
    assert_eq!(input.softmax.params.shape.cols, 4);
    assert_eq!(input.down_proj.witness.lhs.len(), 16);
    assert_eq!(input.down_proj.witness.rhs.len(), 16);
}
