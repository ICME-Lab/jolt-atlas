use crate::ops::{
    hadamard_mul::HadamardRoundWitness, pv_matmul::PvMatmulWitness, qk_score::QkScoreWitness,
    rms_norm::RmsNormWitness, rope::RopeWitness, round::ROUND_FRAC_BITS, silu::SiluRoundWitness,
    softmax::SoftmaxWitness,
};

pub(crate) use crate::trace::build_layer_witness_from_trace_dir;

// Prover-side tensor material. The complete prover owns materialization; the
// IOP only borrows these views when proving individual equations.

#[derive(Debug, Clone)]
pub struct LayerWitness {
    pub hidden_in: Vec<i32>,
    pub rms_norm_atten_sum_x2: Vec<i64>,
    pub rms_norm_atten_norm_acc: Vec<i64>,
    pub rms_norm_atten_norm: Vec<i32>,
    pub rms_norm_atten_norm_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub rms_norm_atten_acc: Vec<i64>,
    pub rms_norm_atten_a: Vec<i32>,
    pub rms_norm_atten_b: Vec<i32>,
    pub rms_norm_atten_c: Vec<i32>,
    pub rms_norm_atten_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub context_acc: Vec<i64>,
    pub context: Vec<i32>,
    pub context_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub o_proj: Vec<i32>,
    pub o_proj_acc: Vec<i64>,
    pub o_proj_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub softmax: Vec<i32>,
    pub softmax_acc: Vec<i64>,
    pub softmax_floor: Vec<i32>,
    pub softmax_floor_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub softmax_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub softmax_max_index: Vec<usize>,
    pub softmax_min_diff: i64,
    pub softmax_max_diff: i64,
    pub softmax_ra: Vec<u8>,
    pub softmax_exp_acc: Vec<i64>,
    pub softmax_exp_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub qk_score: Vec<i32>,
    pub qk_score_acc: Vec<i64>,
    pub qk_score_dot: Vec<i32>,
    pub qk_score_dot_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub qk_score_scale_acc: Vec<i64>,
    pub qk_score_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub q_rope_acc: Vec<i64>,
    pub q_rope: Vec<i32>,
    pub q_rope_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub k_rope_acc: Vec<i64>,
    pub k_rope: Vec<i32>,
    pub k_rope_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub q_proj: Vec<i32>,
    pub k_proj: Vec<i32>,
    pub q_norm_sum_x2: Vec<i64>,
    pub k_norm_sum_x2: Vec<i64>,
    pub q_norm_norm_acc: Vec<i64>,
    pub k_norm_norm_acc: Vec<i64>,
    pub q_norm_norm: Vec<i32>,
    pub k_norm_norm: Vec<i32>,
    pub q_norm_norm_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub k_norm_norm_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub q_norm_acc: Vec<i64>,
    pub k_norm_acc: Vec<i64>,
    pub q_norm: Vec<i32>,
    pub k_norm: Vec<i32>,
    pub q_norm_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub k_norm_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub q_proj_acc: Vec<i64>,
    pub q_proj_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub k_proj_acc: Vec<i64>,
    pub k_proj_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub v_proj_acc: Vec<i64>,
    pub v_proj_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub softmax_max: Vec<i32>,
    pub softmax_exp: Vec<i32>,
    pub softmax_sum: Vec<i32>,
    pub v_proj: Vec<i32>,
    pub residual_add_attn_a: Vec<i32>,
    pub residual_add_attn_b: Vec<i32>,
    pub rms_norm_mlp_sum_x2: Vec<i64>,
    pub rms_norm_mlp_norm_acc: Vec<i64>,
    pub rms_norm_mlp_norm: Vec<i32>,
    pub rms_norm_mlp_norm_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub rms_norm_mlp_acc: Vec<i64>,
    pub rms_norm_mlp_a: Vec<i32>,
    pub rms_norm_mlp_b: Vec<i32>,
    pub rms_norm_mlp_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub gate_proj_acc: Vec<i64>,
    pub gate_proj: Vec<i32>,
    pub gate_proj_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub silu_acc: Vec<i64>,
    pub silu: Vec<i32>,
    pub silu_min_n: i64,
    pub silu_max_n: i64,
    pub silu_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub silu_ra: Vec<u8>,
    pub silu_out_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub silu_up_acc: Vec<i64>,
    pub silu_up: Vec<i32>,
    pub silu_up_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub up_proj_acc: Vec<i64>,
    pub up_proj: Vec<i32>,
    pub up_proj_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
    pub down_proj_acc: Vec<i64>,
    pub down_proj: Vec<i32>,
    pub down_proj_frac_bits: [Vec<u8>; ROUND_FRAC_BITS],
}

impl LayerWitness {
    pub(crate) fn softmax_witness(&self) -> SoftmaxWitness<'_> {
        SoftmaxWitness {
            input: &self.qk_score,
            max_index: &self.softmax_max_index,
            max: &self.softmax_max,
            min_diff: self.softmax_min_diff,
            max_diff: self.softmax_max_diff,
            ra: &self.softmax_ra,
            exp_acc: &self.softmax_exp_acc,
            exp: &self.softmax_exp,
            exp_frac_bits: self.softmax_exp_frac_bits.each_ref().map(Vec::as_slice),
            sum: &self.softmax_sum,
            acc: &self.softmax_acc,
            floor: &self.softmax_floor,
            floor_frac_bits: self.softmax_floor_frac_bits.each_ref().map(Vec::as_slice),
            output: &self.softmax,
            frac_bits: self.softmax_frac_bits.each_ref().map(Vec::as_slice),
        }
    }

    pub(crate) fn qk_score_witness(&self) -> QkScoreWitness<'_> {
        QkScoreWitness {
            q: &self.q_rope,
            k: &self.k_rope,
            raw_acc: &self.qk_score_acc,
            dot: &self.qk_score_dot,
            dot_frac_bits: self.qk_score_dot_frac_bits.each_ref().map(Vec::as_slice),
            scale_acc: &self.qk_score_scale_acc,
            output: &self.qk_score,
            frac_bits: self.qk_score_frac_bits.each_ref().map(Vec::as_slice),
        }
    }

    pub(crate) fn q_rope_witness(&self) -> RopeWitness<'_> {
        RopeWitness {
            input: &self.q_norm,
            acc: &self.q_rope_acc,
            output: &self.q_rope,
            frac_bits: self.q_rope_frac_bits.each_ref().map(Vec::as_slice),
        }
    }

    pub(crate) fn k_rope_witness(&self) -> RopeWitness<'_> {
        RopeWitness {
            input: &self.k_norm,
            acc: &self.k_rope_acc,
            output: &self.k_rope,
            frac_bits: self.k_rope_frac_bits.each_ref().map(Vec::as_slice),
        }
    }

    pub(crate) fn q_norm_witness(&self) -> RmsNormWitness<'_> {
        RmsNormWitness {
            input: &self.q_proj,
            sum_x2: &self.q_norm_sum_x2,
            norm_acc: &self.q_norm_norm_acc,
            norm: &self.q_norm_norm,
            norm_frac_bits: self.q_norm_norm_frac_bits.each_ref().map(Vec::as_slice),
            acc: &self.q_norm_acc,
            output: &self.q_norm,
            frac_bits: self.q_norm_frac_bits.each_ref().map(Vec::as_slice),
        }
    }

    pub(crate) fn k_norm_witness(&self) -> RmsNormWitness<'_> {
        RmsNormWitness {
            input: &self.k_proj,
            sum_x2: &self.k_norm_sum_x2,
            norm_acc: &self.k_norm_norm_acc,
            norm: &self.k_norm_norm,
            norm_frac_bits: self.k_norm_norm_frac_bits.each_ref().map(Vec::as_slice),
            acc: &self.k_norm_acc,
            output: &self.k_norm,
            frac_bits: self.k_norm_frac_bits.each_ref().map(Vec::as_slice),
        }
    }

    pub(crate) fn rms_norm_atten_witness(&self) -> RmsNormWitness<'_> {
        RmsNormWitness {
            input: &self.hidden_in,
            sum_x2: &self.rms_norm_atten_sum_x2,
            norm_acc: &self.rms_norm_atten_norm_acc,
            norm: &self.rms_norm_atten_norm,
            norm_frac_bits: self
                .rms_norm_atten_norm_frac_bits
                .each_ref()
                .map(Vec::as_slice),
            acc: &self.rms_norm_atten_acc,
            output: &self.rms_norm_atten_a,
            frac_bits: self.rms_norm_atten_frac_bits.each_ref().map(Vec::as_slice),
        }
    }

    pub(crate) fn pv_matmul_witness(&self) -> PvMatmulWitness<'_> {
        PvMatmulWitness {
            p: &self.softmax,
            v: &self.v_proj,
            acc: &self.context_acc,
            output: &self.context,
            frac_bits: self.context_frac_bits.each_ref().map(Vec::as_slice),
        }
    }

    pub(crate) fn rms_norm_mlp_witness(&self) -> RmsNormWitness<'_> {
        RmsNormWitness {
            input: &self.residual_add_attn_b,
            sum_x2: &self.rms_norm_mlp_sum_x2,
            norm_acc: &self.rms_norm_mlp_norm_acc,
            norm: &self.rms_norm_mlp_norm,
            norm_frac_bits: self
                .rms_norm_mlp_norm_frac_bits
                .each_ref()
                .map(Vec::as_slice),
            acc: &self.rms_norm_mlp_acc,
            output: &self.rms_norm_mlp_a,
            frac_bits: self.rms_norm_mlp_frac_bits.each_ref().map(Vec::as_slice),
        }
    }

    pub(crate) fn silu_witness(&self) -> SiluRoundWitness<'_> {
        SiluRoundWitness {
            min_n: self.silu_min_n,
            max_n: self.silu_max_n,
            gate_proj: &self.gate_proj,
            silu_acc: &self.silu_acc,
            silu_ra: &self.silu_ra,
            silu: &self.silu,
        }
    }

    pub(crate) fn silu_up_witness(&self) -> HadamardRoundWitness<'_> {
        HadamardRoundWitness {
            lhs: &self.silu,
            rhs: &self.up_proj,
            acc: &self.silu_up_acc,
            output: &self.silu_up,
            frac_bits: self.silu_up_frac_bits.each_ref().map(Vec::as_slice),
        }
    }
}
