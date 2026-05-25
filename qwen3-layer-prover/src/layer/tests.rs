use ark_bn254::{Bn254, Fr};
use joltworks::{
    poly::{commitment::commitment_scheme::CommitmentScheme, commitment::hyperkzg::HyperKZG},
    transcripts::Blake2bTranscript,
};

use super::{
    claims::{claim_hidden_out_after_commitments, draw_hidden_out_point, point_matches_claim},
    commitments::{LayerPolynomialMap, absorb_layer_commitments, commit_layer_polynomials},
    iop::{prove_layer_iop, verify_layer_iop},
    openings::{prove_layer_claim_openings, verify_layer_openings},
    *,
};
use crate::ops::round::ROUND_FRAC_BITS;

#[test]
fn proves_and_verifies_layer_first_step() {
    let shape = LayerShape {
        seq: 2,
        hidden: 2,
        intermediate: 3,
        q_heads: 2,
        kv_heads: 1,
        head_dim: 2,
    };
    let tensors = LayerTensorIds::default();
    let weights = nonzero_layer_weights(&shape);
    let witness = layer_witness(&shape, &weights);
    let hidden_out = witness
        .residual_add_attn_a
        .iter()
        .zip(&witness.down_proj)
        .map(|(&lhs, &rhs)| lhs + rhs)
        .collect::<Vec<_>>();
    let polynomials =
        LayerPolynomialMap::from_layer(&hidden_out, &witness, &weights, &shape, &tensors);

    type Pcs = HyperKZG<Bn254>;
    let pcs_setup = Pcs::setup_prover(16);
    let verifier_setup = Pcs::setup_verifier(&pcs_setup);
    let (hidden_in_commitment, _) = Pcs::commit(
        &joltworks::poly::multilinear_polynomial::MultilinearPolynomial::from(pad_power_of_two(
            &witness.hidden_in,
        )),
        &pcs_setup,
    );
    let (hidden_out_commitment, _) = Pcs::commit(
        &joltworks::poly::multilinear_polynomial::MultilinearPolynomial::from(pad_power_of_two(
            &hidden_out,
        )),
        &pcs_setup,
    );
    let commitments = commit_layer_polynomials::<Fr, Pcs>(
        &polynomials,
        HiddenStateCommitments {
            hidden_in: hidden_in_commitment,
            hidden_out: hidden_out_commitment,
        },
        &pcs_setup,
    );

    let mut prover_transcript = Blake2bTranscript::default();
    absorb_layer_commitments(&mut prover_transcript, 0, &shape, &commitments);
    let hidden_out_claim =
        claim_hidden_out_after_commitments::<Fr, _>(&mut prover_transcript, &hidden_out, &shape);
    let result = prove_layer_iop::<Fr, _>(
        hidden_out_claim.clone(),
        &witness,
        &weights,
        &shape,
        &tensors,
        &mut prover_transcript,
    )
    .unwrap();

    let mut verifier_transcript = Blake2bTranscript::default();
    absorb_layer_commitments(&mut verifier_transcript, 0, &shape, &commitments);
    let verifier_hidden_out_point =
        draw_hidden_out_point::<Fr, _>(&mut verifier_transcript, &shape);
    assert!(point_matches_claim(
        &hidden_out_claim,
        &verifier_hidden_out_point
    ));
    let claims = verify_layer_iop::<Fr, _>(
        hidden_out_claim.clone(),
        &result.proof,
        &weights,
        &shape,
        &tensors,
        &mut verifier_transcript,
    )
    .unwrap();

    assert_eq!(claims, result.claims);
    assert_eq!(claims.hidden_in_a.tensor.0, "hidden_in");
    assert_eq!(claims.hidden_in_b.tensor.0, "hidden_in");
    assert_eq!(
        polynomials.missing_opening_claims(&claims.opening_claims()),
        Vec::<String>::new()
    );
    assert_eq!(
        polynomials.opening_value_mismatches(&claims.opening_claims()),
        Vec::<String>::new()
    );

    let opening_proof = prove_layer_claim_openings::<Fr, _, Pcs>(
        &polynomials,
        claims.opening_claims(),
        hidden_out_claim,
        &pcs_setup,
        &mut prover_transcript,
    )
    .unwrap();
    verify_layer_openings::<Fr, _, Pcs>(
        &commitments,
        &opening_proof,
        &verifier_setup,
        &mut verifier_transcript,
    )
    .unwrap();
}

fn nonzero_layer_weights(shape: &LayerShape) -> LayerWeights {
    LayerWeights {
        rope_cos: seq_i32(shape.seq * (shape.head_dim / 2)),
        rope_sin: seq_i32(shape.seq * (shape.head_dim / 2)),
        rms_norm_atten: seq_i32(shape.hidden),
        q_norm: seq_i32(shape.head_dim),
        k_norm: seq_i32(shape.head_dim),
        rms_norm_mlp: seq_i32(shape.hidden),
        o_proj: seq_i32(shape.attention_width() * shape.hidden),
        q_proj: seq_i32(shape.hidden * shape.attention_width()),
        k_proj: seq_i32(shape.hidden * shape.kv_heads * shape.head_dim),
        v_proj: seq_i32(shape.hidden * shape.kv_heads * shape.head_dim),
        gate_proj: gate_test_weights(shape),
        up_proj: seq_i32(shape.hidden * shape.intermediate),
        down_proj: seq_i32(shape.intermediate * shape.hidden),
    }
}

fn seq_i32(len: usize) -> Vec<i32> {
    vec![1; len]
}

fn gate_test_weights(shape: &LayerShape) -> Vec<i32> {
    let pattern = [1, 128, 256];
    (0..shape.hidden)
        .flat_map(|_| (0..shape.intermediate).map(|col| pattern[col % pattern.len()]))
        .collect()
}

fn layer_witness(shape: &LayerShape, weights: &LayerWeights) -> LayerWitness {
    let hidden_in = seq_i32(shape.seq * shape.hidden);
    let rms_norm_atten = rms_norm(&hidden_in, &weights.rms_norm_atten, shape.seq, shape.hidden);
    let q_proj = matmul(
        &rms_norm_atten.output,
        &weights.q_proj,
        shape.seq,
        shape.hidden,
        shape.attention_width(),
    );
    let k_proj = matmul(
        &rms_norm_atten.output,
        &weights.k_proj,
        shape.seq,
        shape.hidden,
        shape.kv_heads * shape.head_dim,
    );
    let v_proj = matmul(
        &rms_norm_atten.output,
        &weights.v_proj,
        shape.seq,
        shape.hidden,
        shape.kv_heads * shape.head_dim,
    );
    let q_norm = rms_norm(
        &q_proj.output,
        &weights.q_norm,
        shape.seq * shape.q_heads,
        shape.head_dim,
    );
    let k_norm = rms_norm(
        &k_proj.output,
        &weights.k_norm,
        shape.seq * shape.kv_heads,
        shape.head_dim,
    );
    let q_rope = rope(
        &q_norm.output,
        &weights.rope_cos,
        &weights.rope_sin,
        shape.seq,
        shape.q_heads,
        shape.head_dim,
    );
    let k_rope = rope(
        &k_norm.output,
        &weights.rope_cos,
        &weights.rope_sin,
        shape.seq,
        shape.kv_heads,
        shape.head_dim,
    );
    let qk_score = qk_score(&q_rope.output, &k_rope.output, shape);
    let softmax = causal_softmax(&qk_score.output, shape);
    let context = pv_matmul(&softmax.output, &v_proj.output, shape);
    let o_proj = matmul(
        &context.output,
        &weights.o_proj,
        shape.seq,
        shape.attention_width(),
        shape.hidden,
    );
    let residual_add_attn = hidden_in
        .iter()
        .zip(&o_proj.output)
        .map(|(&lhs, &rhs)| lhs + rhs)
        .collect::<Vec<_>>();
    let rms_norm_mlp = rms_norm(
        &residual_add_attn,
        &weights.rms_norm_mlp,
        shape.seq,
        shape.hidden,
    );
    let gate_proj = matmul(
        &rms_norm_mlp.output,
        &weights.gate_proj,
        shape.seq,
        shape.hidden,
        shape.intermediate,
    );
    let up_proj = matmul(
        &rms_norm_mlp.output,
        &weights.up_proj,
        shape.seq,
        shape.hidden,
        shape.intermediate,
    );
    let silu = silu(&gate_proj.output);
    let silu_up = hadamard(&silu.output, &up_proj.output);
    let down_proj = matmul(
        &silu_up.output,
        &weights.down_proj,
        shape.seq,
        shape.intermediate,
        shape.hidden,
    );
    LayerWitness {
        hidden_in: hidden_in.clone(),
        rms_norm_atten_sum_x2: rms_norm_atten.sum_x2,
        rms_norm_atten_norm_acc: rms_norm_atten.norm_acc,
        rms_norm_atten_norm: rms_norm_atten.norm,
        rms_norm_atten_norm_frac_bits: rms_norm_atten.norm_frac_bits,
        rms_norm_atten_acc: rms_norm_atten.acc,
        rms_norm_atten_a: rms_norm_atten.output.clone(),
        rms_norm_atten_b: rms_norm_atten.output.clone(),
        rms_norm_atten_c: rms_norm_atten.output,
        rms_norm_atten_frac_bits: rms_norm_atten.frac_bits,
        context_acc: context.acc,
        context: context.output.clone(),
        context_frac_bits: context.frac_bits,
        o_proj: o_proj.output.clone(),
        o_proj_acc: o_proj.acc,
        o_proj_frac_bits: o_proj.frac_bits,
        softmax: softmax.output,
        softmax_acc: softmax.acc,
        softmax_floor: softmax.floor,
        softmax_floor_frac_bits: softmax.floor_frac_bits,
        softmax_frac_bits: softmax.frac_bits,
        softmax_max_index: softmax.max_index,
        softmax_min_diff: softmax.min_diff,
        softmax_max_diff: softmax.max_diff,
        softmax_ra: softmax.ra,
        softmax_exp_acc: softmax.exp_acc,
        softmax_exp_frac_bits: softmax.exp_frac_bits,
        qk_score: qk_score.output,
        qk_score_acc: qk_score.acc,
        qk_score_dot: qk_score.dot,
        qk_score_dot_frac_bits: qk_score.dot_frac_bits,
        qk_score_scale_acc: qk_score.scale_acc,
        qk_score_frac_bits: qk_score.frac_bits,
        q_rope_acc: q_rope.acc,
        q_rope: q_rope.output,
        q_rope_frac_bits: q_rope.frac_bits,
        k_rope_acc: k_rope.acc,
        k_rope: k_rope.output,
        k_rope_frac_bits: k_rope.frac_bits,
        q_proj: q_proj.output.clone(),
        k_proj: k_proj.output.clone(),
        q_norm_sum_x2: q_norm.sum_x2,
        k_norm_sum_x2: k_norm.sum_x2,
        q_norm_norm_acc: q_norm.norm_acc,
        k_norm_norm_acc: k_norm.norm_acc,
        q_norm_norm: q_norm.norm,
        k_norm_norm: k_norm.norm,
        q_norm_norm_frac_bits: q_norm.norm_frac_bits,
        k_norm_norm_frac_bits: k_norm.norm_frac_bits,
        q_norm_acc: q_norm.acc,
        k_norm_acc: k_norm.acc,
        q_norm: q_norm.output,
        k_norm: k_norm.output,
        q_norm_frac_bits: q_norm.frac_bits,
        k_norm_frac_bits: k_norm.frac_bits,
        q_proj_acc: q_proj.acc,
        q_proj_frac_bits: q_proj.frac_bits,
        k_proj_acc: k_proj.acc,
        k_proj_frac_bits: k_proj.frac_bits,
        v_proj_acc: v_proj.acc,
        v_proj_frac_bits: v_proj.frac_bits,
        softmax_max: softmax.max,
        softmax_exp: softmax.exp,
        softmax_sum: softmax.sum,
        v_proj: v_proj.output,
        residual_add_attn_a: residual_add_attn.clone(),
        residual_add_attn_b: residual_add_attn,
        rms_norm_mlp_sum_x2: rms_norm_mlp.sum_x2,
        rms_norm_mlp_norm_acc: rms_norm_mlp.norm_acc,
        rms_norm_mlp_norm: rms_norm_mlp.norm,
        rms_norm_mlp_norm_frac_bits: rms_norm_mlp.norm_frac_bits,
        rms_norm_mlp_acc: rms_norm_mlp.acc,
        rms_norm_mlp_a: rms_norm_mlp.output.clone(),
        rms_norm_mlp_b: rms_norm_mlp.output,
        rms_norm_mlp_frac_bits: rms_norm_mlp.frac_bits,
        gate_proj_acc: gate_proj.acc,
        gate_proj: gate_proj.output.clone(),
        gate_proj_frac_bits: gate_proj.frac_bits,
        silu_acc: silu.acc,
        silu_min_n: silu.min_n,
        silu_max_n: silu.max_n,
        silu_frac_bits: silu.frac_bits,
        silu_ra: silu.ra,
        silu: silu.output,
        silu_out_frac_bits: silu.out_frac_bits,
        silu_up_acc: silu_up.acc,
        silu_up: silu_up.output,
        silu_up_frac_bits: silu_up.frac_bits,
        up_proj_acc: up_proj.acc,
        up_proj: up_proj.output,
        up_proj_frac_bits: up_proj.frac_bits,
        down_proj_acc: down_proj.acc,
        down_proj: down_proj.output,
        down_proj_frac_bits: down_proj.frac_bits,
    }
}

struct SoftmaxFixture {
    output: Vec<i32>,
    acc: Vec<i64>,
    sum: Vec<i32>,
    max_index: Vec<usize>,
    max: Vec<i32>,
    min_diff: i64,
    max_diff: i64,
    ra: Vec<u8>,
    exp_acc: Vec<i64>,
    exp: Vec<i32>,
    exp_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    floor: Vec<i32>,
    floor_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
}

struct RoundFixture {
    acc: Vec<i64>,
    output: Vec<i32>,
    frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
}

struct RmsFixture {
    sum_x2: Vec<i64>,
    norm_acc: Vec<i64>,
    norm: Vec<i32>,
    norm_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    acc: Vec<i64>,
    output: Vec<i32>,
    frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
}

struct QkFixture {
    acc: Vec<i64>,
    dot: Vec<i32>,
    dot_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    scale_acc: Vec<i64>,
    output: Vec<i32>,
    frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
}

struct SiluFixture {
    acc: Vec<i64>,
    output: Vec<i32>,
    frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    ra: Vec<u8>,
    out_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    min_n: i64,
    max_n: i64,
}

fn matmul(a: &[i32], w: &[i32], m: usize, k: usize, n: usize) -> RoundFixture {
    let mut acc = vec![0_i64; m * n];
    for row in 0..m {
        for col in 0..n {
            acc[row * n + col] = (0..k)
                .map(|kk| i64::from(a[row * k + kk]) * i64::from(w[kk * n + col]))
                .sum();
        }
    }
    round_fixture(acc)
}

fn rms_norm(input: &[i32], weight: &[i32], rows: usize, cols: usize) -> RmsFixture {
    let mut sum_x2 = vec![0_i64; rows];
    let mut norm_acc = vec![0_i64; rows * cols];
    for row in 0..rows {
        sum_x2[row] = input[row * cols..(row + 1) * cols]
            .iter()
            .map(|&value| i64::from(value) * i64::from(value))
            .sum();
        let inv = rms_inv_from_square_sum(sum_x2[row], cols);
        for col in 0..cols {
            norm_acc[row * cols + col] = i64::from(input[row * cols + col]) * inv;
        }
    }
    let norm = round_fixture(norm_acc);
    let acc = norm
        .output
        .iter()
        .enumerate()
        .map(|(idx, &value)| i64::from(value) * i64::from(weight[idx % cols]))
        .collect::<Vec<_>>();
    let rounded = round_fixture(acc);
    RmsFixture {
        sum_x2,
        norm_acc: norm.acc,
        norm: norm.output,
        norm_frac_bits: norm.frac_bits,
        acc: rounded.acc,
        output: rounded.output,
        frac_bits: rounded.frac_bits,
    }
}

fn rope(
    input: &[i32],
    cos: &[i32],
    sin: &[i32],
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> RoundFixture {
    let mut acc = vec![0_i64; seq * heads * head_dim];
    for s in 0..seq {
        for h in 0..heads {
            for pair in 0..head_dim / 2 {
                let base = (s * heads + h) * head_dim;
                let lo = base + pair;
                let hi = base + head_dim / 2 + pair;
                let coeff = s * (head_dim / 2) + pair;
                let x0 = i64::from(input[lo]);
                let x1 = i64::from(input[hi]);
                let c = i64::from(cos[coeff]);
                let si = i64::from(sin[coeff]);
                acc[lo] = x0 * c - x1 * si;
                acc[hi] = x0 * si + x1 * c;
            }
        }
    }
    round_fixture(acc)
}

fn qk_score(q: &[i32], k: &[i32], shape: &LayerShape) -> QkFixture {
    let mut acc = vec![0_i64; shape.q_heads * shape.seq * shape.seq];
    for h in 0..shape.q_heads {
        let kh = h / (shape.q_heads / shape.kv_heads);
        for i in 0..shape.seq {
            for j in 0..shape.seq {
                let out = (h * shape.seq + i) * shape.seq + j;
                acc[out] = (0..shape.head_dim)
                    .map(|d| {
                        let qi = (i * shape.q_heads + h) * shape.head_dim + d;
                        let ki = (j * shape.kv_heads + kh) * shape.head_dim + d;
                        i64::from(q[qi]) * i64::from(k[ki])
                    })
                    .sum();
            }
        }
    }
    let dot = round_fixture(acc);
    let inv_sqrt = ((1.0 / (shape.head_dim as f64).sqrt()) * 256.0).round() as i64;
    let scale_acc = dot
        .output
        .iter()
        .map(|&value| i64::from(value) * inv_sqrt)
        .collect::<Vec<_>>();
    let score = round_fixture(scale_acc);
    QkFixture {
        acc: dot.acc,
        dot: dot.output,
        dot_frac_bits: dot.frac_bits,
        scale_acc: score.acc,
        output: score.output,
        frac_bits: score.frac_bits,
    }
}

fn pv_matmul(p: &[i32], v: &[i32], shape: &LayerShape) -> RoundFixture {
    let mut acc = vec![0_i64; shape.seq * shape.q_heads * shape.head_dim];
    for i in 0..shape.seq {
        for h in 0..shape.q_heads {
            let kh = h / (shape.q_heads / shape.kv_heads);
            for d in 0..shape.head_dim {
                let out = (i * shape.q_heads + h) * shape.head_dim + d;
                acc[out] = (0..shape.seq)
                    .map(|j| {
                        let pi = (h * shape.seq + i) * shape.seq + j;
                        let vi = (j * shape.kv_heads + kh) * shape.head_dim + d;
                        i64::from(p[pi]) * i64::from(v[vi])
                    })
                    .sum();
            }
        }
    }
    round_fixture(acc)
}

fn hadamard(lhs: &[i32], rhs: &[i32]) -> RoundFixture {
    round_fixture(
        lhs.iter()
            .zip(rhs)
            .map(|(&lhs, &rhs)| i64::from(lhs) * i64::from(rhs))
            .collect(),
    )
}

fn silu(gate: &[i32]) -> SiluFixture {
    let rounded = gate
        .iter()
        .map(|&value| round_q8(i64::from(value)))
        .collect::<Vec<_>>();
    let min_n = *rounded.iter().min().unwrap() as i64;
    let max_n = *rounded.iter().max().unwrap() as i64;
    let entries = (max_n - min_n + 1) as usize;
    let mut ra = vec![0; gate.len() * entries];
    let mut acc = vec![0_i64; gate.len()];
    for (idx, &gate_value) in gate.iter().enumerate() {
        let n = i64::from(rounded[idx]);
        ra[idx * entries + (n - min_n) as usize] = 1;
        acc[idx] = silu_base(n) + (i64::from(gate_value) - n * 256) * silu_slope(n);
    }
    let rounded = round_fixture(acc);
    SiluFixture {
        acc: rounded.acc,
        output: rounded.output,
        frac_bits: frac_bits_i32(gate),
        ra,
        out_frac_bits: rounded.frac_bits,
        min_n,
        max_n,
    }
}

fn causal_softmax(scores: &[i32], shape: &LayerShape) -> SoftmaxFixture {
    let rows = shape.q_heads * shape.seq;
    let cols = shape.seq;
    let mut output = vec![0; rows * cols];
    let mut acc = vec![0; rows * cols];
    let mut floor = vec![0; rows * cols];
    let mut sum = vec![0; rows];
    let mut max_index = vec![0; rows];
    let mut max = vec![0; rows];
    let mut exp = vec![256; rows * cols];
    let mut exp_acc = vec![256_i64 * 256_i64; rows * cols];
    let mut diffs = vec![0_i64; rows * cols];
    let mut min_diff = 0_i64;
    let max_diff = 0_i64;
    for row in 0..rows {
        let query_pos = row % shape.seq;
        let row_scores = &scores[row * cols..(row + 1) * cols];
        let (idx, value) = row_scores[..=query_pos]
            .iter()
            .enumerate()
            .max_by_key(|(_, value)| *value)
            .unwrap();
        max_index[row] = idx;
        max[row] = *value;
        for col in 0..cols {
            let out = row * cols + col;
            if col <= query_pos {
                let diff = i64::from(scores[out]) - i64::from(max[row]);
                let n = floor_q8(diff);
                diffs[out] = i64::from(n);
                min_diff = min_diff.min(i64::from(n));
                exp_acc[out] = softmax_exp_acc_q8(diff);
                exp[out] = softmax_exp_coarse_q8(diff);
                sum[row] += exp[out];
            }
        }
        let inv = inv_sum_q16(sum[row]);
        for col in 0..cols {
            let idx = row * cols + col;
            if col <= query_pos {
                acc[idx] = i64::from(exp[idx]) * inv;
                floor[idx] = floor_q8(acc[idx]);
                output[idx] = round_q8(i64::from(floor[idx]));
            }
        }
    }
    let entries = (max_diff - min_diff + 1) as usize;
    let mut ra = vec![0; scores.len() * entries];
    for (idx, &diff) in diffs.iter().enumerate() {
        ra[idx * entries + (diff - min_diff) as usize] = 1;
    }
    let floor_frac_bits = frac_bits_i64(&acc);
    let frac_bits = frac_bits_i32(&floor);
    let exp_frac_bits = frac_bits_i64(&exp_acc);
    SoftmaxFixture {
        output,
        acc,
        sum,
        max_index,
        max,
        min_diff,
        max_diff,
        ra,
        exp_acc,
        exp,
        exp_frac_bits,
        floor,
        floor_frac_bits,
        frac_bits,
    }
}

fn round_fixture(acc: Vec<i64>) -> RoundFixture {
    RoundFixture {
        output: acc.iter().map(|&value| round_q8(value)).collect(),
        frac_bits: frac_bits_i64(&acc),
        acc,
    }
}

fn rms_inv_from_square_sum(square_sum: i64, hidden_size: usize) -> i64 {
    let mean = square_sum as f64 / hidden_size as f64 / (256.0 * 256.0);
    ((1.0 / (mean + 1e-6).sqrt()) * 256.0).round() as i64
}

fn silu_base(n: i64) -> i64 {
    let n_q8 = n * 256;
    n_q8 * sigmoid_q8(n)
}

fn silu_slope(n: i64) -> i64 {
    let n_q8 = n * 256;
    let sig = sigmoid_q8(n);
    let sig_slope = i64::from(round_q8(sig * (256 - sig)));
    sig + i64::from(round_q8(n_q8 * sig_slope))
}

fn sigmoid_q8(n: i64) -> i64 {
    ((1.0 / (1.0 + (-(n as f64)).exp())) * 256.0).round() as i64
}

fn softmax_exp_coarse_q8(delta_q8: i64) -> i32 {
    round_q8(softmax_exp_acc_q8(delta_q8))
}

fn softmax_exp_acc_q8(delta_q8: i64) -> i64 {
    let n = delta_q8.div_euclid(256);
    let f = delta_q8 - n * 256;
    let exp_n = exp_lut_q8(n);
    let corr = (256 + f).max(0);
    exp_n * corr
}

fn exp_lut_q8(n: i64) -> i64 {
    let n = n.clamp(-16, 0);
    (f64::exp(n as f64) * 256.0).round() as i64
}

fn inv_sum_q16(sum: i32) -> i64 {
    ((1_i64 << 24) as f64 / f64::from(sum)).round() as i64
}

fn frac_bits_i64(values: &[i64]) -> [Vec<u8>; ROUND_FRAC_BITS] {
    std::array::from_fn(|bit| {
        values
            .iter()
            .map(|value| ((value.rem_euclid(256) >> bit) & 1) as u8)
            .collect()
    })
}

fn frac_bits_i32(values: &[i32]) -> [Vec<u8>; ROUND_FRAC_BITS] {
    std::array::from_fn(|bit| {
        values
            .iter()
            .map(|value| ((i64::from(*value).rem_euclid(256) >> bit) & 1) as u8)
            .collect()
    })
}

fn round_q8(value: i64) -> i32 {
    ((value + ((value.rem_euclid(256) >> 7) * 256) - value.rem_euclid(256)) / 256) as i32
}

fn floor_q8(value: i64) -> i32 {
    value.div_euclid(256) as i32
}

fn pad_power_of_two<T: Copy + Default>(values: &[T]) -> Vec<T> {
    let len = values.len().next_power_of_two();
    let mut out = vec![T::default(); len];
    out[..values.len()].copy_from_slice(values);
    out
}
