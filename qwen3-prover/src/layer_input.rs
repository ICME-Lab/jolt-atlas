use crate::{
    layer::{LayerOpeningWitnesses, LayerProverInput, LayerShape},
    ops::{
        add::{AddParams, AddProverInput, AddWitness},
        matmul::{MatMulParams, MatMulProverInput, MatMulWitness},
        mul::{MulParams, MulProverInput, MulWitness},
        pv_matmul::{PvMatmulParams, PvMatmulProverInput, PvMatmulWitness},
        qk_score::{QkScoreParams, QkScoreProverInput, QkScoreWitness},
        rms_norm::{
            RmsNormAdvice, RmsNormParams, RmsNormProverInput, RmsNormWitness,
            rms_inv_from_square_sum,
        },
        rope::{RopeParams, RopeProverInput, RopeWitness},
        silu::{SiluAdvice, SiluParams, SiluProverInput, SiluWitness},
        softmax::{SoftmaxAdvice, SoftmaxParams, SoftmaxProverInput, SoftmaxWitness},
    },
};

use itertools::Itertools;
use qwen3_common::FRAC_BITS;
pub use qwen3_common::{LayerRawWitness, LayerWeights};

pub fn layer_prover_input(
    shape: LayerShape,
    weights: LayerWeights,
    witness: LayerRawWitness,
) -> Option<LayerProverInput> {
    validate_logical_shape(shape)?;
    let domain = domain_shape(shape);

    let hidden = shape.seq.checked_mul(shape.hidden)?;
    let attention_width = shape.q_heads.checked_mul(shape.head_dim)?;
    let kv_width = shape.kv_heads.checked_mul(shape.head_dim)?;
    (witness.hidden_out.len() == hidden).then_some(())?;

    let silu_entries = entries(witness.silu_min_n, witness.silu_max_n)?;
    let softmax_entries = entries(witness.softmax_min_diff, witness.softmax_max_diff)?;
    let silu_zero = zero_lookup_index(witness.silu_min_n, witness.silu_max_n)?;
    let softmax_zero = zero_lookup_index(witness.softmax_min_diff, witness.softmax_max_diff)?;

    let hidden_in = pad2(
        witness.hidden_in,
        shape.seq,
        shape.hidden,
        domain.seq,
        domain.hidden,
    )?;
    let residual_add_attn_a = pad2(
        witness.residual_add_attn_a,
        shape.seq,
        shape.hidden,
        domain.seq,
        domain.hidden,
    )?;
    let residual_add_attn_b = pad2(
        witness.residual_add_attn_b,
        shape.seq,
        shape.hidden,
        domain.seq,
        domain.hidden,
    )?;
    let rms_norm_atten_a = pad2(
        witness.rms_norm_atten_a,
        shape.seq,
        shape.hidden,
        domain.seq,
        domain.hidden,
    )?;
    let rms_norm_atten_b = pad2(
        witness.rms_norm_atten_b,
        shape.seq,
        shape.hidden,
        domain.seq,
        domain.hidden,
    )?;
    let rms_norm_atten_c = pad2(
        witness.rms_norm_atten_c,
        shape.seq,
        shape.hidden,
        domain.seq,
        domain.hidden,
    )?;
    let rms_norm_atten_norm = pad2(
        witness.rms_norm_atten_norm,
        shape.seq,
        shape.hidden,
        domain.seq,
        domain.hidden,
    )?;
    let rms_norm_mlp_a = pad2(
        witness.rms_norm_mlp_a,
        shape.seq,
        shape.hidden,
        domain.seq,
        domain.hidden,
    )?;
    let rms_norm_mlp_b = pad2(
        witness.rms_norm_mlp_b,
        shape.seq,
        shape.hidden,
        domain.seq,
        domain.hidden,
    )?;
    let rms_norm_mlp_norm = pad2(
        witness.rms_norm_mlp_norm,
        shape.seq,
        shape.hidden,
        domain.seq,
        domain.hidden,
    )?;
    let context = pad3(
        witness.context,
        [shape.seq, shape.q_heads, shape.head_dim],
        [domain.seq, domain.q_heads, domain.head_dim],
    )?;
    let o_proj = pad2(
        witness.o_proj,
        shape.seq,
        shape.hidden,
        domain.seq,
        domain.hidden,
    )?;
    let down_proj = pad2(
        witness.down_proj,
        shape.seq,
        shape.hidden,
        domain.seq,
        domain.hidden,
    )?;
    let gate_proj = pad2(
        witness.gate_proj,
        shape.seq,
        shape.intermediate,
        domain.seq,
        domain.intermediate,
    )?;
    let up_proj = pad2(
        witness.up_proj,
        shape.seq,
        shape.intermediate,
        domain.seq,
        domain.intermediate,
    )?;
    let silu = pad2(
        witness.silu,
        shape.seq,
        shape.intermediate,
        domain.seq,
        domain.intermediate,
    )?;
    let silu_up = pad2(
        witness.silu_up,
        shape.seq,
        shape.intermediate,
        domain.seq,
        domain.intermediate,
    )?;
    let q_proj = pad3(
        witness.q_proj,
        [shape.seq, shape.q_heads, shape.head_dim],
        [domain.seq, domain.q_heads, domain.head_dim],
    )?;
    let k_proj = pad3(
        witness.k_proj,
        [shape.seq, shape.kv_heads, shape.head_dim],
        [domain.seq, domain.kv_heads, domain.head_dim],
    )?;
    let v_proj = pad3(
        witness.v_proj,
        [shape.seq, shape.kv_heads, shape.head_dim],
        [domain.seq, domain.kv_heads, domain.head_dim],
    )?;
    let q_norm = pad3(
        witness.q_norm,
        [shape.seq, shape.q_heads, shape.head_dim],
        [domain.seq, domain.q_heads, domain.head_dim],
    )?;
    let k_norm = pad3(
        witness.k_norm,
        [shape.seq, shape.kv_heads, shape.head_dim],
        [domain.seq, domain.kv_heads, domain.head_dim],
    )?;
    let q_norm_norm = pad3(
        witness.q_norm_norm,
        [shape.seq, shape.q_heads, shape.head_dim],
        [domain.seq, domain.q_heads, domain.head_dim],
    )?;
    let k_norm_norm = pad3(
        witness.k_norm_norm,
        [shape.seq, shape.kv_heads, shape.head_dim],
        [domain.seq, domain.kv_heads, domain.head_dim],
    )?;
    let q_rope = pad3(
        witness.q_rope,
        [shape.seq, shape.q_heads, shape.head_dim],
        [domain.seq, domain.q_heads, domain.head_dim],
    )?;
    let k_rope = pad3(
        witness.k_rope,
        [shape.seq, shape.kv_heads, shape.head_dim],
        [domain.seq, domain.kv_heads, domain.head_dim],
    )?;
    let qk_score = pad_score(
        witness.qk_score,
        shape.q_heads,
        shape.seq,
        domain.q_heads,
        domain.seq,
    )?;
    let qk_score_dot = pad_score(
        witness.qk_score_dot,
        shape.q_heads,
        shape.seq,
        domain.q_heads,
        domain.seq,
    )?;
    let softmax = pad_score(
        witness.softmax,
        shape.q_heads,
        shape.seq,
        domain.q_heads,
        domain.seq,
    )?;
    let softmax_acc = pad_score(
        witness.softmax_acc,
        shape.q_heads,
        shape.seq,
        domain.q_heads,
        domain.seq,
    )?;
    let softmax_floor = pad_score(
        witness.softmax_floor,
        shape.q_heads,
        shape.seq,
        domain.q_heads,
        domain.seq,
    )?;
    let softmax_exp = pad_score_with(
        witness.softmax_exp,
        shape.q_heads,
        shape.seq,
        domain.q_heads,
        domain.seq,
        256,
    )?;
    let softmax_exp_acc = pad_score_with(
        witness.softmax_exp_acc,
        shape.q_heads,
        shape.seq,
        domain.q_heads,
        domain.seq,
        256_i64 * 256_i64,
    )?;
    let softmax_max = pad_score_rows(
        witness.softmax_max,
        shape.q_heads,
        shape.seq,
        domain.q_heads,
        domain.seq,
    )?;
    let softmax_max_index = pad_score_rows(
        witness.softmax_max_index,
        shape.q_heads,
        shape.seq,
        domain.q_heads,
        domain.seq,
    )?;
    let softmax_sum = pad_score_rows(
        witness.softmax_sum,
        shape.q_heads,
        shape.seq,
        domain.q_heads,
        domain.seq,
    )?;
    let q_norm_sum_x2 = pad2(
        witness.q_norm_sum_x2,
        shape.seq,
        shape.q_heads,
        domain.seq,
        domain.q_heads,
    )?;
    let k_norm_sum_x2 = pad2(
        witness.k_norm_sum_x2,
        shape.seq,
        shape.kv_heads,
        domain.seq,
        domain.kv_heads,
    )?;
    let rms_norm_atten_sum_x2 = pad1(witness.rms_norm_atten_sum_x2, shape.seq, domain.seq)?;
    let rms_norm_mlp_sum_x2 = pad1(witness.rms_norm_mlp_sum_x2, shape.seq, domain.seq)?;
    let silu_lookup_ra = pad_lookup_ra2(
        witness.silu_lookup_ra,
        shape.seq,
        shape.intermediate,
        domain.seq,
        domain.intermediate,
        silu_entries,
        silu_zero,
    )?;
    let softmax_lookup_ra = pad_lookup_ra3(
        witness.softmax_lookup_ra,
        [shape.q_heads, shape.seq, shape.seq],
        [domain.q_heads, domain.seq, domain.seq],
        softmax_entries,
        softmax_zero,
    )?;
    let rope_cos = pad2(
        weights.rope_cos,
        shape.seq,
        shape.head_dim / 2,
        domain.seq,
        domain.head_dim / 2,
    )?;
    let rope_sin = pad2(
        weights.rope_sin,
        shape.seq,
        shape.head_dim / 2,
        domain.seq,
        domain.head_dim / 2,
    )?;
    let w_gate = pad2(
        weights.gate_proj,
        shape.hidden,
        shape.intermediate,
        domain.hidden,
        domain.intermediate,
    )?;
    let w_up = pad2(
        weights.up_proj,
        shape.hidden,
        shape.intermediate,
        domain.hidden,
        domain.intermediate,
    )?;
    let w_down = pad2(
        weights.down_proj,
        shape.intermediate,
        shape.hidden,
        domain.intermediate,
        domain.hidden,
    )?;
    let w_o = pad_tensor_rows_col_major(
        weights.o_proj,
        shape.q_heads,
        shape.head_dim,
        shape.hidden,
        domain.q_heads,
        domain.head_dim,
        domain.hidden,
    )?;
    let w_q = pad_tensor_cols_col_major(
        weights.q_proj,
        shape.hidden,
        shape.q_heads,
        shape.head_dim,
        domain.hidden,
        domain.q_heads,
        domain.head_dim,
    )?;
    let w_k = pad_tensor_cols_col_major(
        weights.k_proj,
        shape.hidden,
        shape.kv_heads,
        shape.head_dim,
        domain.hidden,
        domain.kv_heads,
        domain.head_dim,
    )?;
    let w_v = pad_tensor_cols_col_major(
        weights.v_proj,
        shape.hidden,
        shape.kv_heads,
        shape.head_dim,
        domain.hidden,
        domain.kv_heads,
        domain.head_dim,
    )?;
    let softmax_output_frac_bits = softmax_output_frac_bits(
        &qk_score,
        &softmax_max,
        &softmax_lookup_ra,
        witness.softmax_min_diff,
        witness.softmax_max_diff,
        domain.q_heads * domain.seq,
        domain.seq,
    )?;
    let silu_opening_ra = expand_ra2_for_opening(
        &silu_lookup_ra,
        domain.seq,
        domain.intermediate,
        silu_entries,
        silu_padded_lut_len(silu_entries),
    )?;
    let softmax_opening_ra = expand_ra2_for_opening(
        &softmax_lookup_ra,
        domain.q_heads * domain.seq,
        domain.seq,
        softmax_entries,
        softmax_padded_lut_len(softmax_entries),
    )?;
    let opening_witnesses = LayerOpeningWitnesses {
        hidden_out: pad2_opening(
            witness.hidden_out,
            shape.seq,
            shape.hidden,
            domain.seq,
            domain.hidden,
        )?,
        hidden_in_a: hidden_in.clone(),
        hidden_in_b: hidden_in.clone(),
        silu_lookup_ra: silu_opening_ra,
        softmax_lookup_ra: softmax_opening_ra,
        down_proj_output_frac_bits: bits2_opening(
            witness.down_proj_output_frac_bits.clone(),
            shape.seq,
            shape.hidden,
            domain.seq,
            domain.hidden,
        )?,
        silu_up_output_frac_bits: bits2_opening(
            witness.silu_up_output_frac_bits.clone(),
            shape.seq,
            shape.intermediate,
            domain.seq,
            domain.intermediate,
        )?,
        silu_input_frac_bits: bits2_opening(
            witness.silu_input_frac_bits.clone(),
            shape.seq,
            shape.intermediate,
            domain.seq,
            domain.intermediate,
        )?,
        silu_output_frac_bits: bits2_opening(
            witness.silu_output_frac_bits.clone(),
            shape.seq,
            shape.intermediate,
            domain.seq,
            domain.intermediate,
        )?,
        gate_proj_output_frac_bits: bits2_opening(
            witness.gate_proj_output_frac_bits.clone(),
            shape.seq,
            shape.intermediate,
            domain.seq,
            domain.intermediate,
        )?,
        up_proj_output_frac_bits: bits2_opening(
            witness.up_proj_output_frac_bits.clone(),
            shape.seq,
            shape.intermediate,
            domain.seq,
            domain.intermediate,
        )?,
        rms_norm_mlp_norm_frac_bits: bits2_opening(
            witness.rms_norm_mlp_norm_frac_bits.clone(),
            shape.seq,
            shape.hidden,
            domain.seq,
            domain.hidden,
        )?,
        rms_norm_mlp_output_frac_bits: bits2_opening(
            witness.rms_norm_mlp_output_frac_bits.clone(),
            shape.seq,
            shape.hidden,
            domain.seq,
            domain.hidden,
        )?,
        o_proj_output_frac_bits: bits2_opening(
            witness.o_proj_output_frac_bits.clone(),
            shape.seq,
            shape.hidden,
            domain.seq,
            domain.hidden,
        )?,
        pv_matmul_output_frac_bits: bits3_opening(
            witness.pv_matmul_output_frac_bits.clone(),
            [shape.seq, shape.q_heads, shape.head_dim],
            [domain.seq, domain.q_heads, domain.head_dim],
        )?,
        softmax_floor_frac_bits: bits_score_opening(
            witness.softmax_floor_frac_bits.clone(),
            shape.q_heads,
            shape.seq,
            domain.q_heads,
            domain.seq,
        )?,
        softmax_output_frac_bits: bits_score_opening(
            witness.softmax_output_frac_bits.clone(),
            shape.q_heads,
            shape.seq,
            domain.q_heads,
            domain.seq,
        )?,
        softmax_exp_frac_bits: bits_score_opening(
            witness.softmax_exp_frac_bits.clone(),
            shape.q_heads,
            shape.seq,
            domain.q_heads,
            domain.seq,
        )?,
        qk_score_dot_output_frac_bits: bits_score_opening(
            witness.qk_score_dot_output_frac_bits.clone(),
            shape.q_heads,
            shape.seq,
            domain.q_heads,
            domain.seq,
        )?,
        qk_score_output_frac_bits: bits_score_opening(
            witness.qk_score_output_frac_bits.clone(),
            shape.q_heads,
            shape.seq,
            domain.q_heads,
            domain.seq,
        )?,
        q_rope_output_frac_bits: bits3_opening(
            witness.q_rope_output_frac_bits.clone(),
            [shape.seq, shape.q_heads, shape.head_dim],
            [domain.seq, domain.q_heads, domain.head_dim],
        )?,
        k_rope_output_frac_bits: bits3_opening(
            witness.k_rope_output_frac_bits.clone(),
            [shape.seq, shape.kv_heads, shape.head_dim],
            [domain.seq, domain.kv_heads, domain.head_dim],
        )?,
        q_norm_norm_frac_bits: bits3_opening(
            witness.q_norm_norm_frac_bits.clone(),
            [shape.seq, shape.q_heads, shape.head_dim],
            [domain.seq, domain.q_heads, domain.head_dim],
        )?,
        q_norm_output_frac_bits: bits3_opening(
            witness.q_norm_output_frac_bits.clone(),
            [shape.seq, shape.q_heads, shape.head_dim],
            [domain.seq, domain.q_heads, domain.head_dim],
        )?,
        k_norm_norm_frac_bits: bits3_opening(
            witness.k_norm_norm_frac_bits.clone(),
            [shape.seq, shape.kv_heads, shape.head_dim],
            [domain.seq, domain.kv_heads, domain.head_dim],
        )?,
        k_norm_output_frac_bits: bits3_opening(
            witness.k_norm_output_frac_bits.clone(),
            [shape.seq, shape.kv_heads, shape.head_dim],
            [domain.seq, domain.kv_heads, domain.head_dim],
        )?,
        q_proj_output_frac_bits: bits3_opening(
            witness.q_proj_output_frac_bits.clone(),
            [shape.seq, shape.q_heads, shape.head_dim],
            [domain.seq, domain.q_heads, domain.head_dim],
        )?,
        k_proj_output_frac_bits: bits3_opening(
            witness.k_proj_output_frac_bits.clone(),
            [shape.seq, shape.kv_heads, shape.head_dim],
            [domain.seq, domain.kv_heads, domain.head_dim],
        )?,
        v_proj_output_frac_bits: bits3_opening(
            witness.v_proj_output_frac_bits.clone(),
            [shape.seq, shape.kv_heads, shape.head_dim],
            [domain.seq, domain.kv_heads, domain.head_dim],
        )?,
        rms_norm_atten_norm_frac_bits: bits2_opening(
            witness.rms_norm_atten_norm_frac_bits.clone(),
            shape.seq,
            shape.hidden,
            domain.seq,
            domain.hidden,
        )?,
        rms_norm_atten_output_frac_bits: bits2_opening(
            witness.rms_norm_atten_output_frac_bits.clone(),
            shape.seq,
            shape.hidden,
            domain.seq,
            domain.hidden,
        )?,
    };

    Some(LayerProverInput {
        shape: domain,
        opening_witnesses,
        residual_add_mlp: add(
            domain.seq,
            domain.hidden,
            residual_add_attn_a,
            down_proj.clone(),
        )?,
        down_proj: matmul(
            domain.seq,
            domain.hidden,
            domain.intermediate,
            silu_up.clone(),
            w_down,
            down_proj,
            bits2(
                witness.down_proj_output_frac_bits,
                shape.seq,
                shape.hidden,
                domain.seq,
                domain.hidden,
            )?,
        )?,
        silu_up: MulProverInput {
            params: MulParams::matrix(domain.seq, domain.intermediate)?,
            witness: MulWitness {
                a: silu.clone(),
                b: up_proj.clone(),
                bits: bits2(
                    witness.silu_up_output_frac_bits,
                    shape.seq,
                    shape.intermediate,
                    domain.seq,
                    domain.intermediate,
                )?,
            },
        },
        silu: SiluProverInput {
            params: SiluParams::new(domain.seq, domain.intermediate)?,
            advice: SiluAdvice {
                min_n: witness.silu_min_n,
                max_n: witness.silu_max_n,
            },
            witness: SiluWitness {
                input: gate_proj.clone(),
                output: silu,
                ra: silu_lookup_ra,
                input_remainder_bits: bits2(
                    witness.silu_input_frac_bits,
                    shape.seq,
                    shape.intermediate,
                    domain.seq,
                    domain.intermediate,
                )?,
                output_remainder_bits: bits2(
                    witness.silu_output_frac_bits,
                    shape.seq,
                    shape.intermediate,
                    domain.seq,
                    domain.intermediate,
                )?,
            },
        },
        gate_proj: matmul(
            domain.seq,
            domain.intermediate,
            domain.hidden,
            rms_norm_mlp_a,
            w_gate,
            gate_proj,
            bits2(
                witness.gate_proj_output_frac_bits,
                shape.seq,
                shape.intermediate,
                domain.seq,
                domain.intermediate,
            )?,
        )?,
        up_proj: matmul(
            domain.seq,
            domain.intermediate,
            domain.hidden,
            rms_norm_mlp_b.clone(),
            w_up,
            up_proj,
            bits2(
                witness.up_proj_output_frac_bits,
                shape.seq,
                shape.intermediate,
                domain.seq,
                domain.intermediate,
            )?,
        )?,
        rms_norm_mlp: rms_norm(
            domain.seq,
            domain.hidden,
            rms_norm_mlp_sum_x2,
            residual_add_attn_b.clone(),
            rms_norm_mlp_norm,
            weights.rms_norm_mlp,
            rms_norm_mlp_b,
            bits2(
                witness.rms_norm_mlp_norm_frac_bits,
                shape.seq,
                shape.hidden,
                domain.seq,
                domain.hidden,
            )?,
            bits2(
                witness.rms_norm_mlp_output_frac_bits,
                shape.seq,
                shape.hidden,
                domain.seq,
                domain.hidden,
            )?,
        )?,
        residual_add_attn: add(domain.seq, domain.hidden, hidden_in.clone(), o_proj.clone())?,
        o_proj: matmul(
            domain.seq,
            domain.hidden,
            attention_width,
            context.clone(),
            w_o,
            o_proj,
            bits2(
                witness.o_proj_output_frac_bits,
                shape.seq,
                shape.hidden,
                domain.seq,
                domain.hidden,
            )?,
        )?,
        pv_matmul: PvMatmulProverInput {
            params: PvMatmulParams::new(
                domain.seq,
                domain.q_heads,
                domain.kv_heads,
                domain.head_dim,
            )?,
            witness: PvMatmulWitness {
                p: softmax.clone(),
                v: v_proj.clone(),
                context_remainder_bits: bits3(
                    witness.pv_matmul_output_frac_bits,
                    [shape.seq, shape.q_heads, shape.head_dim],
                    [domain.seq, domain.q_heads, domain.head_dim],
                )?,
            },
        },
        softmax: SoftmaxProverInput {
            params: SoftmaxParams::new(domain.q_heads * domain.seq, domain.seq)?,
            advice: SoftmaxAdvice {
                min_diff: witness.softmax_min_diff,
                max_diff: witness.softmax_max_diff,
                row_max: softmax_max,
                max_index: softmax_max_index,
                sum: softmax_sum,
            },
            witness: SoftmaxWitness {
                input: qk_score.clone(),
                output: softmax,
                ra: softmax_lookup_ra,
                exp_acc: softmax_exp_acc,
                exp: softmax_exp,
                frac_bits: softmax_output_frac_bits,
                exp_remainder_bits: bits_score_opening(
                    witness.softmax_exp_frac_bits,
                    shape.q_heads,
                    shape.seq,
                    domain.q_heads,
                    domain.seq,
                )?,
                acc: softmax_acc,
                floor: softmax_floor,
                floor_remainder_bits: bits_score_opening(
                    witness.softmax_floor_frac_bits,
                    shape.q_heads,
                    shape.seq,
                    domain.q_heads,
                    domain.seq,
                )?,
                output_remainder_bits: bits_score_opening(
                    witness.softmax_output_frac_bits,
                    shape.q_heads,
                    shape.seq,
                    domain.q_heads,
                    domain.seq,
                )?,
            },
        },
        qk_score: QkScoreProverInput {
            params: QkScoreParams::new(
                domain.seq,
                domain.q_heads,
                domain.kv_heads,
                domain.head_dim,
            )?,
            witness: QkScoreWitness {
                q: q_rope.clone(),
                k: k_rope.clone(),
                dot: qk_score_dot,
                score_remainder_bits: bits_score_opening(
                    witness.qk_score_output_frac_bits,
                    shape.q_heads,
                    shape.seq,
                    domain.q_heads,
                    domain.seq,
                )?,
                dot_remainder_bits: bits_score_opening(
                    witness.qk_score_dot_output_frac_bits,
                    shape.q_heads,
                    shape.seq,
                    domain.q_heads,
                    domain.seq,
                )?,
            },
        },
        q_rope: rope(
            domain.seq,
            domain.q_heads,
            domain.head_dim,
            q_norm.clone(),
            q_rope,
            bits3(
                witness.q_rope_output_frac_bits,
                [shape.seq, shape.q_heads, shape.head_dim],
                [domain.seq, domain.q_heads, domain.head_dim],
            )?,
            rope_cos.clone(),
            rope_sin.clone(),
        )?,
        k_rope: rope(
            domain.seq,
            domain.kv_heads,
            domain.head_dim,
            k_norm.clone(),
            k_rope,
            bits3(
                witness.k_rope_output_frac_bits,
                [shape.seq, shape.kv_heads, shape.head_dim],
                [domain.seq, domain.kv_heads, domain.head_dim],
            )?,
            rope_cos,
            rope_sin,
        )?,
        q_norm: rms_norm(
            domain.seq * domain.q_heads,
            domain.head_dim,
            q_norm_sum_x2,
            q_proj.clone(),
            q_norm_norm,
            weights.q_norm,
            q_norm,
            bits3(
                witness.q_norm_norm_frac_bits,
                [shape.seq, shape.q_heads, shape.head_dim],
                [domain.seq, domain.q_heads, domain.head_dim],
            )?,
            bits3(
                witness.q_norm_output_frac_bits,
                [shape.seq, shape.q_heads, shape.head_dim],
                [domain.seq, domain.q_heads, domain.head_dim],
            )?,
        )?,
        k_norm: rms_norm(
            domain.seq * domain.kv_heads,
            domain.head_dim,
            k_norm_sum_x2,
            k_proj.clone(),
            k_norm_norm,
            weights.k_norm,
            k_norm,
            bits3(
                witness.k_norm_norm_frac_bits,
                [shape.seq, shape.kv_heads, shape.head_dim],
                [domain.seq, domain.kv_heads, domain.head_dim],
            )?,
            bits3(
                witness.k_norm_output_frac_bits,
                [shape.seq, shape.kv_heads, shape.head_dim],
                [domain.seq, domain.kv_heads, domain.head_dim],
            )?,
        )?,
        q_proj: matmul(
            domain.seq,
            attention_width,
            domain.hidden,
            rms_norm_atten_a,
            w_q,
            q_proj,
            bits3(
                witness.q_proj_output_frac_bits,
                [shape.seq, shape.q_heads, shape.head_dim],
                [domain.seq, domain.q_heads, domain.head_dim],
            )?,
        )?,
        k_proj: matmul(
            domain.seq,
            kv_width,
            domain.hidden,
            rms_norm_atten_b,
            w_k,
            k_proj,
            bits3(
                witness.k_proj_output_frac_bits,
                [shape.seq, shape.kv_heads, shape.head_dim],
                [domain.seq, domain.kv_heads, domain.head_dim],
            )?,
        )?,
        v_proj: matmul(
            domain.seq,
            kv_width,
            domain.hidden,
            rms_norm_atten_c.clone(),
            w_v,
            v_proj,
            bits3(
                witness.v_proj_output_frac_bits,
                [shape.seq, shape.kv_heads, shape.head_dim],
                [domain.seq, domain.kv_heads, domain.head_dim],
            )?,
        )?,
        rms_norm_atten: rms_norm(
            domain.seq,
            domain.hidden,
            rms_norm_atten_sum_x2,
            hidden_in,
            rms_norm_atten_norm,
            weights.rms_norm_atten,
            rms_norm_atten_c,
            bits2(
                witness.rms_norm_atten_norm_frac_bits,
                shape.seq,
                shape.hidden,
                domain.seq,
                domain.hidden,
            )?,
            bits2(
                witness.rms_norm_atten_output_frac_bits,
                shape.seq,
                shape.hidden,
                domain.seq,
                domain.hidden,
            )?,
        )?,
    })
}

fn add(rows: usize, cols: usize, lhs: Vec<i32>, rhs: Vec<i32>) -> Option<AddProverInput> {
    Some(AddProverInput {
        params: AddParams::new(rows, cols)?,
        witness: AddWitness { lhs, rhs },
    })
}

fn matmul(
    rows: usize,
    cols: usize,
    inner: usize,
    lhs: Vec<i32>,
    rhs: Vec<i32>,
    output: Vec<i32>,
    output_remainder_bits: [Vec<bool>; FRAC_BITS],
) -> Option<MatMulProverInput> {
    Some(MatMulProverInput {
        params: MatMulParams::new(rows, cols, inner)?,
        witness: MatMulWitness {
            lhs,
            rhs,
            output,
            output_remainder: byte_remainders(output_remainder_bits)?,
        },
    })
}

fn byte_remainders(bits: [Vec<bool>; FRAC_BITS]) -> Option<Vec<u8>> {
    let len = bits.first()?.len();
    bits.iter().all(|bit| bit.len() == len).then_some(())?;

    let mut remainders = vec![0_u8; len];
    for (bit, values) in bits.into_iter().enumerate() {
        for (remainder, value) in remainders.iter_mut().zip_eq(values) {
            *remainder += u8::from(value) << bit;
        }
    }
    Some(remainders)
}

fn rms_norm(
    rows: usize,
    cols: usize,
    sum_x2: Vec<i64>,
    input: Vec<i32>,
    norm: Vec<i32>,
    weight: Vec<i32>,
    output: Vec<i32>,
    norm_remainder_bits: [Vec<bool>; FRAC_BITS],
    output_remainder_bits: [Vec<bool>; FRAC_BITS],
) -> Option<RmsNormProverInput> {
    Some(RmsNormProverInput {
        params: RmsNormParams::new(rows, cols)?,
        advice: RmsNormAdvice {
            sum_x2: sum_x2.clone(),
        },
        witness: RmsNormWitness {
            input,
            inv_rms: expand_inv_rms(&sum_x2, rows, cols)?,
            norm,
            weight: expand_cols(weight, rows, cols)?,
            output,
            norm_remainder_bits,
            output_remainder_bits,
        },
    })
}

fn rope(
    seq: usize,
    heads: usize,
    head_dim: usize,
    input: Vec<i32>,
    output: Vec<i32>,
    output_remainder_bits: [Vec<bool>; FRAC_BITS],
    cos: Vec<i32>,
    sin: Vec<i32>,
) -> Option<RopeProverInput> {
    Some(RopeProverInput {
        params: RopeParams::new(seq, heads, head_dim)?,
        witness: RopeWitness {
            input,
            output,
            output_remainder_bits,
            cos,
            sin,
        },
    })
}

fn validate_logical_shape(shape: LayerShape) -> Option<()> {
    (shape.seq > 0
        && shape.q_heads.is_power_of_two()
        && shape.kv_heads.is_power_of_two()
        && shape.head_dim.is_power_of_two()
        && shape.hidden.is_power_of_two()
        && shape.intermediate > 0)
        .then_some(())
}

fn domain_shape(shape: LayerShape) -> LayerShape {
    shape.padded()
}

fn entries(min: i64, max: i64) -> Option<usize> {
    (max >= min)
        .then_some(max - min + 1)
        .and_then(|entries| usize::try_from(entries).ok())
}

fn zero_lookup_index(min: i64, max: i64) -> Option<usize> {
    (min <= 0 && 0 <= max)
        .then_some(-min)
        .and_then(|index| usize::try_from(index).ok())
}

fn silu_padded_lut_len(entries: usize) -> usize {
    (entries + 1).next_power_of_two().max(16)
}

fn softmax_padded_lut_len(entries: usize) -> usize {
    entries.next_power_of_two().max(2)
}

fn expand_ra2_for_opening(
    ra: &[u8],
    rows: usize,
    cols: usize,
    entries: usize,
    lut_len: usize,
) -> Option<Vec<u8>> {
    (lut_len >= entries).then_some(())?;
    let tensor_len = rows.checked_mul(cols)?;
    (ra.len() == tensor_len.checked_mul(entries)?).then_some(())?;
    let mut expanded = vec![0_u8; tensor_len.checked_mul(lut_len)?];
    for entry in 0..entries {
        let source = entry * tensor_len;
        let target = entry * tensor_len;
        expanded[target..target + tensor_len].copy_from_slice(&ra[source..source + tensor_len]);
    }
    Some(expanded)
}

fn pad1<T>(values: Vec<T>, len: usize, padded_len: usize) -> Option<Vec<T>>
where
    T: Copy + Default,
{
    (values.len() == len && len <= padded_len).then_some(())?;
    let mut out = vec![T::default(); padded_len];
    out[..len].copy_from_slice(&values);
    Some(out)
}

fn pad2<T>(
    values: Vec<T>,
    rows: usize,
    cols: usize,
    padded_rows: usize,
    padded_cols: usize,
) -> Option<Vec<T>>
where
    T: Copy + Default,
{
    pad2_with(values, rows, cols, padded_rows, padded_cols, T::default())
}

fn pad2_with<T>(
    values: Vec<T>,
    rows: usize,
    cols: usize,
    padded_rows: usize,
    padded_cols: usize,
    default: T,
) -> Option<Vec<T>>
where
    T: Copy,
{
    (values.len() == rows.checked_mul(cols)?).then_some(())?;
    (rows <= padded_rows && cols <= padded_cols).then_some(())?;
    let mut out = vec![default; padded_rows.checked_mul(padded_cols)?];
    for row in 0..rows {
        for col in 0..cols {
            out[col * padded_rows + row] = values[row * cols + col];
        }
    }
    Some(out)
}

fn pad3<T>(values: Vec<T>, dims: [usize; 3], padded_dims: [usize; 3]) -> Option<Vec<T>>
where
    T: Copy + Default,
{
    pad3_with(values, dims, padded_dims, T::default())
}

fn pad3_with<T>(
    values: Vec<T>,
    [a, b, c]: [usize; 3],
    [pa, pb, pc]: [usize; 3],
    default: T,
) -> Option<Vec<T>>
where
    T: Copy,
{
    (values.len() == a.checked_mul(b)?.checked_mul(c)?).then_some(())?;
    (a <= pa && b <= pb && c <= pc).then_some(())?;
    let mut out = vec![default; pa.checked_mul(pb)?.checked_mul(pc)?];
    for i in 0..a {
        for j in 0..b {
            for k in 0..c {
                let source = (i * b + j) * c + k;
                out[(k * pb + j) * pa + i] = values[source];
            }
        }
    }
    Some(out)
}

fn pad_tensor_rows_col_major<T>(
    values: Vec<T>,
    a: usize,
    b: usize,
    c: usize,
    padded_a: usize,
    padded_b: usize,
    padded_c: usize,
) -> Option<Vec<T>>
where
    T: Copy + Default,
{
    (values.len() == a.checked_mul(b)?.checked_mul(c)?).then_some(())?;
    (a <= padded_a && b <= padded_b && c <= padded_c).then_some(())?;
    let rows = padded_a.checked_mul(padded_b)?;
    let mut out = vec![T::default(); rows.checked_mul(padded_c)?];
    for i in 0..a {
        for j in 0..b {
            for k in 0..c {
                let source = (i * b + j) * c + k;
                let row = j * padded_a + i;
                out[k * rows + row] = values[source];
            }
        }
    }
    Some(out)
}

fn pad_tensor_cols_col_major<T>(
    values: Vec<T>,
    a: usize,
    b: usize,
    c: usize,
    padded_a: usize,
    padded_b: usize,
    padded_c: usize,
) -> Option<Vec<T>>
where
    T: Copy + Default,
{
    (values.len() == a.checked_mul(b)?.checked_mul(c)?).then_some(())?;
    (a <= padded_a && b <= padded_b && c <= padded_c).then_some(())?;
    let cols = padded_b.checked_mul(padded_c)?;
    let mut out = vec![T::default(); padded_a.checked_mul(cols)?];
    for i in 0..a {
        for j in 0..b {
            for k in 0..c {
                let source = (i * b + j) * c + k;
                let col = k * padded_b + j;
                out[col * padded_a + i] = values[source];
            }
        }
    }
    Some(out)
}

fn pad_score<T>(
    values: Vec<T>,
    heads: usize,
    seq: usize,
    padded_heads: usize,
    padded_seq: usize,
) -> Option<Vec<T>>
where
    T: Copy + Default,
{
    pad_score_with(values, heads, seq, padded_heads, padded_seq, T::default())
}

fn pad_score_with<T>(
    values: Vec<T>,
    heads: usize,
    seq: usize,
    padded_heads: usize,
    padded_seq: usize,
    default: T,
) -> Option<Vec<T>>
where
    T: Copy,
{
    (values.len() == heads.checked_mul(seq)?.checked_mul(seq)?).then_some(())?;
    (heads <= padded_heads && seq <= padded_seq).then_some(())?;
    let rows = padded_seq.checked_mul(padded_heads)?;
    let mut out = vec![default; rows.checked_mul(padded_seq)?];
    for head in 0..heads {
        for qpos in 0..seq {
            for kpos in 0..seq {
                let source = (head * seq + qpos) * seq + kpos;
                let row = qpos + padded_seq * head;
                out[kpos * rows + row] = values[source];
            }
        }
    }
    Some(out)
}

fn pad_score_rows<T>(
    values: Vec<T>,
    heads: usize,
    seq: usize,
    padded_heads: usize,
    padded_seq: usize,
) -> Option<Vec<T>>
where
    T: Copy + Default,
{
    (values.len() == heads.checked_mul(seq)?).then_some(())?;
    (heads <= padded_heads && seq <= padded_seq).then_some(())?;
    let mut out = vec![T::default(); padded_heads.checked_mul(padded_seq)?];
    for head in 0..heads {
        for qpos in 0..seq {
            out[qpos + padded_seq * head] = values[head * seq + qpos];
        }
    }
    Some(out)
}

fn pad2_opening<T>(
    values: Vec<T>,
    rows: usize,
    cols: usize,
    padded_rows: usize,
    padded_cols: usize,
) -> Option<Vec<T>>
where
    T: Copy + Default,
{
    (values.len() == rows.checked_mul(cols)?).then_some(())?;
    (rows <= padded_rows && cols <= padded_cols).then_some(())?;
    let mut out = vec![T::default(); padded_rows.checked_mul(padded_cols)?];
    for row in 0..rows {
        for col in 0..cols {
            out[col * padded_rows + row] = values[row * cols + col];
        }
    }
    Some(out)
}

fn pad3_opening<T>(
    values: Vec<T>,
    [a, b, c]: [usize; 3],
    [pa, pb, pc]: [usize; 3],
) -> Option<Vec<T>>
where
    T: Copy + Default,
{
    (values.len() == a.checked_mul(b)?.checked_mul(c)?).then_some(())?;
    (a <= pa && b <= pb && c <= pc).then_some(())?;
    let mut out = vec![T::default(); pa.checked_mul(pb)?.checked_mul(pc)?];
    for i in 0..a {
        for j in 0..b {
            for k in 0..c {
                out[(k * pb + j) * pa + i] = values[(i * b + j) * c + k];
            }
        }
    }
    Some(out)
}

fn pad_lookup_ra2(
    ra: Vec<u8>,
    rows: usize,
    cols: usize,
    padded_rows: usize,
    padded_cols: usize,
    entries: usize,
    default_entry: usize,
) -> Option<Vec<u8>> {
    (default_entry < entries).then_some(())?;
    (ra.len() == rows.checked_mul(cols)?.checked_mul(entries)?).then_some(())?;
    (rows <= padded_rows && cols <= padded_cols).then_some(())?;
    let mut out = vec![0_u8; padded_rows.checked_mul(padded_cols)?.checked_mul(entries)?];
    let tensor_len = padded_rows.checked_mul(padded_cols)?;
    for row in 0..padded_rows {
        for col in 0..padded_cols {
            let tensor_index = row + padded_rows * col;
            if row < rows && col < cols {
                let source = (row * cols + col) * entries;
                for entry in 0..entries {
                    out[entry * tensor_len + tensor_index] = ra[source + entry];
                }
            } else {
                out[default_entry * tensor_len + tensor_index] = 1;
            }
        }
    }
    Some(out)
}

fn pad_lookup_ra3(
    ra: Vec<u8>,
    [a, b, c]: [usize; 3],
    [pa, pb, pc]: [usize; 3],
    entries: usize,
    default_entry: usize,
) -> Option<Vec<u8>> {
    (default_entry < entries).then_some(())?;
    (ra.len() == a.checked_mul(b)?.checked_mul(c)?.checked_mul(entries)?).then_some(())?;
    (a <= pa && b <= pb && c <= pc).then_some(())?;
    let mut out = vec![0_u8; pa.checked_mul(pb)?.checked_mul(pc)?.checked_mul(entries)?];
    let rows = pa.checked_mul(pb)?;
    let tensor_len = rows.checked_mul(pc)?;
    for i in 0..pa {
        for j in 0..pb {
            for k in 0..pc {
                let tensor_index = j + pb * i + rows * k;
                if i < a && j < b && k < c {
                    let source = ((i * b + j) * c + k) * entries;
                    for entry in 0..entries {
                        out[entry * tensor_len + tensor_index] = ra[source + entry];
                    }
                } else {
                    out[default_entry * tensor_len + tensor_index] = 1;
                }
            }
        }
    }
    Some(out)
}

fn expand_inv_rms(sum_x2: &[i64], rows: usize, cols: usize) -> Option<Vec<i32>> {
    (sum_x2.len() == rows).then_some(())?;
    let mut out = Vec::with_capacity(rows.checked_mul(cols)?);
    for _col in 0..cols {
        for sum in sum_x2 {
            out.push(rms_inv_from_square_sum(*sum, cols));
        }
    }
    Some(out)
}

fn expand_cols(values: Vec<i32>, rows: usize, cols: usize) -> Option<Vec<i32>> {
    (values.len() == cols).then_some(())?;
    let mut out = Vec::with_capacity(rows.checked_mul(cols)?);
    for value in values {
        out.extend(std::iter::repeat(value).take(rows));
    }
    Some(out)
}

fn bits(bits: [Vec<u8>; FRAC_BITS]) -> Option<[Vec<bool>; FRAC_BITS]> {
    bits.map(|bit| {
        bit.into_iter()
            .map(|value| match value {
                0 => Some(false),
                1 => Some(true),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
    })
    .into_iter()
    .collect::<Option<Vec<_>>>()?
    .try_into()
    .ok()
}

fn bits2(
    bit_tables: [Vec<u8>; FRAC_BITS],
    rows: usize,
    cols: usize,
    padded_rows: usize,
    padded_cols: usize,
) -> Option<[Vec<bool>; FRAC_BITS]> {
    let [b0, b1, b2, b3, b4, b5, b6, b7] = bit_tables;
    bits([
        pad2(b0, rows, cols, padded_rows, padded_cols)?,
        pad2(b1, rows, cols, padded_rows, padded_cols)?,
        pad2(b2, rows, cols, padded_rows, padded_cols)?,
        pad2(b3, rows, cols, padded_rows, padded_cols)?,
        pad2(b4, rows, cols, padded_rows, padded_cols)?,
        pad2(b5, rows, cols, padded_rows, padded_cols)?,
        pad2(b6, rows, cols, padded_rows, padded_cols)?,
        pad2(b7, rows, cols, padded_rows, padded_cols)?,
    ])
}

fn bits3(
    bit_tables: [Vec<u8>; FRAC_BITS],
    dims: [usize; 3],
    padded_dims: [usize; 3],
) -> Option<[Vec<bool>; FRAC_BITS]> {
    let [b0, b1, b2, b3, b4, b5, b6, b7] = bit_tables;
    bits([
        pad3(b0, dims, padded_dims)?,
        pad3(b1, dims, padded_dims)?,
        pad3(b2, dims, padded_dims)?,
        pad3(b3, dims, padded_dims)?,
        pad3(b4, dims, padded_dims)?,
        pad3(b5, dims, padded_dims)?,
        pad3(b6, dims, padded_dims)?,
        pad3(b7, dims, padded_dims)?,
    ])
}

fn bits2_opening(
    bit_tables: [Vec<u8>; FRAC_BITS],
    rows: usize,
    cols: usize,
    padded_rows: usize,
    padded_cols: usize,
) -> Option<[Vec<bool>; FRAC_BITS]> {
    let [b0, b1, b2, b3, b4, b5, b6, b7] = bit_tables;
    bits([
        pad2_opening(b0, rows, cols, padded_rows, padded_cols)?,
        pad2_opening(b1, rows, cols, padded_rows, padded_cols)?,
        pad2_opening(b2, rows, cols, padded_rows, padded_cols)?,
        pad2_opening(b3, rows, cols, padded_rows, padded_cols)?,
        pad2_opening(b4, rows, cols, padded_rows, padded_cols)?,
        pad2_opening(b5, rows, cols, padded_rows, padded_cols)?,
        pad2_opening(b6, rows, cols, padded_rows, padded_cols)?,
        pad2_opening(b7, rows, cols, padded_rows, padded_cols)?,
    ])
}

fn bits3_opening(
    bit_tables: [Vec<u8>; FRAC_BITS],
    dims: [usize; 3],
    padded_dims: [usize; 3],
) -> Option<[Vec<bool>; FRAC_BITS]> {
    let [b0, b1, b2, b3, b4, b5, b6, b7] = bit_tables;
    bits([
        pad3_opening(b0, dims, padded_dims)?,
        pad3_opening(b1, dims, padded_dims)?,
        pad3_opening(b2, dims, padded_dims)?,
        pad3_opening(b3, dims, padded_dims)?,
        pad3_opening(b4, dims, padded_dims)?,
        pad3_opening(b5, dims, padded_dims)?,
        pad3_opening(b6, dims, padded_dims)?,
        pad3_opening(b7, dims, padded_dims)?,
    ])
}

fn bits_score_opening(
    bit_tables: [Vec<u8>; FRAC_BITS],
    heads: usize,
    seq: usize,
    padded_heads: usize,
    padded_seq: usize,
) -> Option<[Vec<bool>; FRAC_BITS]> {
    let [b0, b1, b2, b3, b4, b5, b6, b7] = bit_tables;
    bits([
        pad_score_opening(b0, heads, seq, padded_heads, padded_seq)?,
        pad_score_opening(b1, heads, seq, padded_heads, padded_seq)?,
        pad_score_opening(b2, heads, seq, padded_heads, padded_seq)?,
        pad_score_opening(b3, heads, seq, padded_heads, padded_seq)?,
        pad_score_opening(b4, heads, seq, padded_heads, padded_seq)?,
        pad_score_opening(b5, heads, seq, padded_heads, padded_seq)?,
        pad_score_opening(b6, heads, seq, padded_heads, padded_seq)?,
        pad_score_opening(b7, heads, seq, padded_heads, padded_seq)?,
    ])
}

fn pad_score_opening<T>(
    values: Vec<T>,
    heads: usize,
    seq: usize,
    padded_heads: usize,
    padded_seq: usize,
) -> Option<Vec<T>>
where
    T: Copy + Default,
{
    (values.len() == heads.checked_mul(seq)?.checked_mul(seq)?).then_some(())?;
    (heads <= padded_heads && seq <= padded_seq).then_some(())?;
    let mut out = vec![
        T::default();
        padded_heads
            .checked_mul(padded_seq)?
            .checked_mul(padded_seq)?
    ];
    for head in 0..heads {
        for qpos in 0..seq {
            for kpos in 0..seq {
                let source = (head * seq + qpos) * seq + kpos;
                let row = head * padded_seq + qpos;
                out[kpos * (padded_heads * padded_seq) + row] = values[source];
            }
        }
    }
    Some(out)
}

fn softmax_output_frac_bits(
    input: &[i32],
    row_max: &[i32],
    ra: &[u8],
    min_diff: i64,
    max_diff: i64,
    rows: usize,
    cols: usize,
) -> Option<[Vec<bool>; FRAC_BITS]> {
    (input.len() == rows.checked_mul(cols)?).then_some(())?;
    (row_max.len() == rows).then_some(())?;
    (max_diff >= min_diff).then_some(())?;
    let entries = usize::try_from(max_diff - min_diff + 1).ok()?;
    (ra.len() == input.len().checked_mul(entries)?).then_some(())?;

    let mut remainders = vec![0_i64; input.len()];
    for row in 0..rows {
        let query = row % cols;
        for col in 0..cols {
            let tensor_index = row + rows * col;
            let selected =
                selected_lookup_index_address_major(ra, tensor_index, rows * cols, entries)?;
            let valid = i64::from(col <= query);
            let diff = valid * i64::from(input[tensor_index] - row_max[row]);
            let n = min_diff + i64::try_from(selected).ok()?;
            remainders[tensor_index] = (diff - 256 * n).rem_euclid(256);
        }
    }

    Some(std::array::from_fn(|bit| {
        remainders
            .iter()
            .map(|value| ((value >> bit) & 1) == 1)
            .collect()
    }))
}

fn selected_lookup_index_address_major(
    ra: &[u8],
    tensor_index: usize,
    tensor_len: usize,
    entries: usize,
) -> Option<usize> {
    let mut selected = None;
    for entry in 0..entries {
        match (ra[entry * tensor_len + tensor_index], selected) {
            (1, None) => selected = Some(entry),
            (1, Some(_)) => return None,
            (0, _) => {}
            _ => return None,
        }
    }
    selected
}
