use crate::ops::{
    hadamard_mul::{HadamardMulParams, HadamardRoundParams},
    matadd::MatAddParams,
    matmul::{MatMulParams, MatMulRoundParams},
    pv_matmul::{PvMatmulParams, PvMatmulRoundParams},
    qk_score::{QkScoreParams, QkScoreRoundParams},
    rms_norm::RmsNormParams,
    rope::{RopeParams, RopeRoundParams},
    round::{ROUND_FRAC_BITS, RoundParams},
    silu::{SiluParams, SiluRoundParams},
    softmax::SoftmaxParams,
};

use super::types::LayerShape;

// Tensor names and op parameters. This file is deliberately mechanical: it
// answers "which tensors does this op prover connect?" after `iop.rs` shows the
// proof order.

pub(crate) mod round_site {
    pub const O_PROJ: usize = 0;
    pub const CONTEXT: usize = 1;
    pub const SOFTMAX_OUTPUT: usize = 2;
    pub const SOFTMAX_FLOOR: usize = 3;
    pub const SOFTMAX_EXP: usize = 4;
    pub const QK_SCORE_SCALE: usize = 5;
    pub const QK_SCORE_DOT: usize = 6;
    pub const Q_ROPE: usize = 7;
    pub const K_ROPE: usize = 8;
    pub const Q_NORM: usize = 9;
    pub const Q_NORM_INTERNAL: usize = 10;
    pub const K_NORM: usize = 11;
    pub const K_NORM_INTERNAL: usize = 12;
    pub const Q_PROJ: usize = 13;
    pub const K_PROJ: usize = 14;
    pub const V_PROJ: usize = 15;
    pub const RMS_NORM_ATTEN: usize = 16;
    pub const RMS_NORM_ATTEN_INTERNAL: usize = 17;
    pub const RMS_NORM_MLP: usize = 18;
    pub const RMS_NORM_MLP_INTERNAL: usize = 19;
    pub const DOWN_PROJ: usize = 20;
    pub const GATE_PROJ: usize = 21;
    pub const UP_PROJ: usize = 22;
    pub const SILU_GATE: usize = 23;
    pub const SILU_OUTPUT: usize = 24;
    pub const SILU_UP: usize = 25;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerTensorIds {
    pub hidden_in_a: String,
    pub hidden_in_b: String,
    pub rms_norm_atten_acc: String,
    pub rms_norm_atten_a: String,
    pub rms_norm_atten_b: String,
    pub rms_norm_atten_c: String,
    pub rms_norm_atten_frac_bits: [String; ROUND_FRAC_BITS],
    pub o_proj: String,
    pub o_proj_acc: String,
    pub o_proj_frac_bits: [String; ROUND_FRAC_BITS],
    pub context_acc: String,
    pub context: String,
    pub context_frac_bits: [String; ROUND_FRAC_BITS],
    pub softmax_acc: String,
    pub softmax: String,
    pub softmax_frac_bits: [String; ROUND_FRAC_BITS],
    pub softmax_ra: String,
    pub qk_score_acc: String,
    pub qk_score_dot: String,
    pub qk_score_dot_frac_bits: [String; ROUND_FRAC_BITS],
    pub qk_score_scale_acc: String,
    pub qk_score: String,
    pub qk_score_frac_bits: [String; ROUND_FRAC_BITS],
    pub q_rope_acc: String,
    pub q_rope: String,
    pub q_rope_frac_bits: [String; ROUND_FRAC_BITS],
    pub k_rope_acc: String,
    pub k_rope: String,
    pub k_rope_frac_bits: [String; ROUND_FRAC_BITS],
    pub q_norm_acc: String,
    pub q_norm: String,
    pub q_norm_frac_bits: [String; ROUND_FRAC_BITS],
    pub k_norm_acc: String,
    pub k_norm: String,
    pub k_norm_frac_bits: [String; ROUND_FRAC_BITS],
    pub q_proj_acc: String,
    pub q_proj: String,
    pub q_proj_frac_bits: [String; ROUND_FRAC_BITS],
    pub k_proj_acc: String,
    pub k_proj: String,
    pub k_proj_frac_bits: [String; ROUND_FRAC_BITS],
    pub v_proj_acc: String,
    pub v_proj: String,
    pub v_proj_frac_bits: [String; ROUND_FRAC_BITS],
    pub residual_add_attn_a: String,
    pub residual_add_attn_b: String,
    pub rms_norm_mlp_acc: String,
    pub rms_norm_mlp_a: String,
    pub rms_norm_mlp_b: String,
    pub rms_norm_mlp_frac_bits: [String; ROUND_FRAC_BITS],
    pub gate_proj_acc: String,
    pub gate_proj: String,
    pub gate_proj_frac_bits: [String; ROUND_FRAC_BITS],
    pub silu_acc: String,
    pub silu: String,
    pub silu_frac_bits: [String; ROUND_FRAC_BITS],
    pub silu_ra: String,
    pub silu_out_frac_bits: [String; ROUND_FRAC_BITS],
    pub silu_up_acc: String,
    pub silu_up: String,
    pub silu_up_frac_bits: [String; ROUND_FRAC_BITS],
    pub up_proj_acc: String,
    pub up_proj: String,
    pub up_proj_frac_bits: [String; ROUND_FRAC_BITS],
    pub down_proj_acc: String,
    pub down_proj: String,
    pub down_proj_frac_bits: [String; ROUND_FRAC_BITS],
    pub w_gate_proj: String,
    pub w_up_proj: String,
    pub w_down_proj: String,
    pub w_q_proj: String,
    pub w_k_proj: String,
    pub w_v_proj: String,
}

impl Default for LayerTensorIds {
    fn default() -> Self {
        Self {
            hidden_in_a: "hidden_in".to_string(),
            hidden_in_b: "hidden_in".to_string(),
            rms_norm_atten_acc: "rms_norm_atten_acc".to_string(),
            rms_norm_atten_a: "rms_norm_atten_a".to_string(),
            rms_norm_atten_b: "rms_norm_atten_b".to_string(),
            rms_norm_atten_c: "rms_norm_atten_c".to_string(),
            rms_norm_atten_frac_bits: std::array::from_fn(|idx| {
                format!("rms_norm_atten_frac_bit_{idx}")
            }),
            o_proj: "o_proj".to_string(),
            o_proj_acc: "o_proj_acc".to_string(),
            o_proj_frac_bits: std::array::from_fn(|idx| format!("o_proj_frac_bit_{idx}")),
            context_acc: "context_acc".to_string(),
            context: "context".to_string(),
            context_frac_bits: std::array::from_fn(|idx| format!("context_frac_bit_{idx}")),
            softmax_acc: "softmax_acc".to_string(),
            softmax: "softmax".to_string(),
            softmax_frac_bits: std::array::from_fn(|idx| format!("softmax_frac_bit_{idx}")),
            softmax_ra: "softmax_ra".to_string(),
            qk_score_acc: "qk_score_acc".to_string(),
            qk_score_dot: "qk_score_dot".to_string(),
            qk_score_dot_frac_bits: std::array::from_fn(|idx| {
                format!("qk_score_dot_frac_bit_{idx}")
            }),
            qk_score_scale_acc: "qk_score_scale_acc".to_string(),
            qk_score: "qk_score".to_string(),
            qk_score_frac_bits: std::array::from_fn(|idx| format!("qk_score_frac_bit_{idx}")),
            q_rope_acc: "q_rope_acc".to_string(),
            q_rope: "q_rope".to_string(),
            q_rope_frac_bits: std::array::from_fn(|idx| format!("q_rope_frac_bit_{idx}")),
            k_rope_acc: "k_rope_acc".to_string(),
            k_rope: "k_rope".to_string(),
            k_rope_frac_bits: std::array::from_fn(|idx| format!("k_rope_frac_bit_{idx}")),
            q_norm_acc: "q_norm_acc".to_string(),
            q_norm: "q_norm".to_string(),
            q_norm_frac_bits: std::array::from_fn(|idx| format!("q_norm_frac_bit_{idx}")),
            k_norm_acc: "k_norm_acc".to_string(),
            k_norm: "k_norm".to_string(),
            k_norm_frac_bits: std::array::from_fn(|idx| format!("k_norm_frac_bit_{idx}")),
            q_proj_acc: "q_proj_acc".to_string(),
            q_proj: "q_proj".to_string(),
            q_proj_frac_bits: std::array::from_fn(|idx| format!("q_proj_frac_bit_{idx}")),
            k_proj_acc: "k_proj_acc".to_string(),
            k_proj: "k_proj".to_string(),
            k_proj_frac_bits: std::array::from_fn(|idx| format!("k_proj_frac_bit_{idx}")),
            v_proj_acc: "v_proj_acc".to_string(),
            v_proj: "v_proj".to_string(),
            v_proj_frac_bits: std::array::from_fn(|idx| format!("v_proj_frac_bit_{idx}")),
            residual_add_attn_a: "residual_add_attn_a".to_string(),
            residual_add_attn_b: "residual_add_attn_b".to_string(),
            rms_norm_mlp_acc: "rms_norm_mlp_acc".to_string(),
            rms_norm_mlp_a: "rms_norm_mlp_a".to_string(),
            rms_norm_mlp_b: "rms_norm_mlp_b".to_string(),
            rms_norm_mlp_frac_bits: std::array::from_fn(|idx| {
                format!("rms_norm_mlp_frac_bit_{idx}")
            }),
            gate_proj_acc: "gate_proj_acc".to_string(),
            gate_proj: "gate_proj".to_string(),
            gate_proj_frac_bits: std::array::from_fn(|idx| format!("gate_proj_frac_bit_{idx}")),
            silu_acc: "silu_acc".to_string(),
            silu: "silu".to_string(),
            silu_frac_bits: std::array::from_fn(|idx| format!("silu_frac_bit_{idx}")),
            silu_ra: "silu_ra".to_string(),
            silu_out_frac_bits: std::array::from_fn(|idx| format!("silu_out_frac_bit_{idx}")),
            silu_up_acc: "silu_up_acc".to_string(),
            silu_up: "silu_up".to_string(),
            silu_up_frac_bits: std::array::from_fn(|idx| format!("silu_up_frac_bit_{idx}")),
            up_proj_acc: "up_proj_acc".to_string(),
            up_proj: "up_proj".to_string(),
            up_proj_frac_bits: std::array::from_fn(|idx| format!("up_proj_frac_bit_{idx}")),
            down_proj_acc: "down_proj_acc".to_string(),
            down_proj: "down_proj".to_string(),
            down_proj_frac_bits: std::array::from_fn(|idx| format!("down_proj_frac_bit_{idx}")),
            w_gate_proj: "w_gate_proj".to_string(),
            w_up_proj: "w_up_proj".to_string(),
            w_down_proj: "w_down_proj".to_string(),
            w_q_proj: "w_q_proj".to_string(),
            w_k_proj: "w_k_proj".to_string(),
            w_v_proj: "w_v_proj".to_string(),
        }
    }
}

impl LayerTensorIds {
    pub(crate) fn residual_add_mlp_params(&self, shape: &LayerShape) -> MatAddParams {
        MatAddParams::new(
            vec![shape.seq, shape.hidden],
            self.residual_add_attn_a.clone(),
            self.down_proj.clone(),
        )
    }

    pub(crate) fn residual_add_attn_params(&self, shape: &LayerShape) -> MatAddParams {
        MatAddParams::new(
            vec![shape.seq, shape.hidden],
            self.hidden_in_a.clone(),
            self.o_proj.clone(),
        )
    }

    pub(crate) fn o_proj_params(&self, shape: &LayerShape) -> MatMulRoundParams {
        MatMulRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.hidden],
                self.o_proj_acc.clone(),
                self.o_proj.clone(),
                self.o_proj_frac_bits.clone(),
            )
            .with_lookup_site(round_site::O_PROJ),
            MatMulParams::new(
                shape.seq,
                shape.attention_width(),
                shape.hidden,
                self.context.clone(),
                "w_o_proj",
            ),
        )
    }

    pub(crate) fn pv_matmul_params(&self, shape: &LayerShape) -> PvMatmulRoundParams {
        PvMatmulRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.q_heads, shape.head_dim],
                self.context_acc.clone(),
                self.context.clone(),
                self.context_frac_bits.clone(),
            )
            .with_lookup_site(round_site::CONTEXT),
            PvMatmulParams::new(
                shape.seq,
                shape.q_heads,
                shape.kv_heads,
                shape.head_dim,
                self.softmax.clone(),
                self.v_proj.clone(),
            ),
        )
    }

    pub(crate) fn softmax_params(&self, shape: &LayerShape) -> SoftmaxParams {
        SoftmaxParams::new(
            vec![shape.q_heads, shape.seq, shape.seq],
            self.softmax_acc.clone(),
            self.qk_score.clone(),
            self.softmax.clone(),
            self.softmax_frac_bits.clone(),
            self.softmax_ra.clone(),
            2,
            true,
        )
        .with_lookup_sites(
            round_site::SOFTMAX_OUTPUT,
            round_site::SOFTMAX_FLOOR,
            round_site::SOFTMAX_EXP,
        )
    }

    pub(crate) fn qk_score_params(&self, shape: &LayerShape) -> QkScoreRoundParams {
        QkScoreRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.q_heads, shape.seq, shape.seq],
                self.qk_score_scale_acc.clone(),
                self.qk_score.clone(),
                self.qk_score_frac_bits.clone(),
            )
            .with_lookup_site(round_site::QK_SCORE_SCALE),
            RoundParams::with_frac_bit_tensors(
                vec![shape.q_heads, shape.seq, shape.seq],
                self.qk_score_acc.clone(),
                self.qk_score_dot.clone(),
                self.qk_score_dot_frac_bits.clone(),
            )
            .with_lookup_site(round_site::QK_SCORE_DOT),
            QkScoreParams::new(
                shape.seq,
                shape.q_heads,
                shape.kv_heads,
                shape.head_dim,
                self.q_rope.clone(),
                self.k_rope.clone(),
            ),
        )
    }

    pub(crate) fn q_rope_params(&self, shape: &LayerShape) -> RopeRoundParams {
        RopeRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.q_heads, shape.head_dim],
                self.q_rope_acc.clone(),
                self.q_rope.clone(),
                self.q_rope_frac_bits.clone(),
            )
            .with_lookup_site(round_site::Q_ROPE),
            RopeParams::new(
                shape.seq,
                shape.q_heads,
                shape.head_dim,
                self.q_norm.clone(),
            ),
        )
    }

    pub(crate) fn k_rope_params(&self, shape: &LayerShape) -> RopeRoundParams {
        RopeRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.kv_heads, shape.head_dim],
                self.k_rope_acc.clone(),
                self.k_rope.clone(),
                self.k_rope_frac_bits.clone(),
            )
            .with_lookup_site(round_site::K_ROPE),
            RopeParams::new(
                shape.seq,
                shape.kv_heads,
                shape.head_dim,
                self.k_norm.clone(),
            ),
        )
    }

    pub(crate) fn q_norm_params(&self, shape: &LayerShape) -> RmsNormParams {
        RmsNormParams::new_nd(
            vec![shape.seq, shape.q_heads, shape.head_dim],
            self.q_proj.clone(),
            "w_q_norm",
            self.q_norm_acc.clone(),
            self.q_norm.clone(),
            self.q_norm_frac_bits.clone(),
        )
        .with_lookup_sites(round_site::Q_NORM, round_site::Q_NORM_INTERNAL)
    }

    pub(crate) fn k_norm_params(&self, shape: &LayerShape) -> RmsNormParams {
        RmsNormParams::new_nd(
            vec![shape.seq, shape.kv_heads, shape.head_dim],
            self.k_proj.clone(),
            "w_k_norm",
            self.k_norm_acc.clone(),
            self.k_norm.clone(),
            self.k_norm_frac_bits.clone(),
        )
        .with_lookup_sites(round_site::K_NORM, round_site::K_NORM_INTERNAL)
    }

    pub(crate) fn q_proj_params(&self, shape: &LayerShape) -> MatMulRoundParams {
        MatMulRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.attention_width()],
                self.q_proj_acc.clone(),
                self.q_proj.clone(),
                self.q_proj_frac_bits.clone(),
            )
            .with_lookup_site(round_site::Q_PROJ),
            MatMulParams::new(
                shape.seq,
                shape.hidden,
                shape.attention_width(),
                self.rms_norm_atten_a.clone(),
                self.w_q_proj.clone(),
            ),
        )
    }

    pub(crate) fn k_proj_params(&self, shape: &LayerShape) -> MatMulRoundParams {
        MatMulRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.kv_heads * shape.head_dim],
                self.k_proj_acc.clone(),
                self.k_proj.clone(),
                self.k_proj_frac_bits.clone(),
            )
            .with_lookup_site(round_site::K_PROJ),
            MatMulParams::new(
                shape.seq,
                shape.hidden,
                shape.kv_heads * shape.head_dim,
                self.rms_norm_atten_b.clone(),
                self.w_k_proj.clone(),
            ),
        )
    }

    pub(crate) fn rms_norm_atten_params(&self, shape: &LayerShape) -> RmsNormParams {
        RmsNormParams::new(
            shape.seq,
            shape.hidden,
            self.hidden_in_b.clone(),
            "w_rms_norm_atten",
            self.rms_norm_atten_acc.clone(),
            self.rms_norm_atten_a.clone(),
            self.rms_norm_atten_frac_bits.clone(),
        )
        .with_lookup_sites(
            round_site::RMS_NORM_ATTEN,
            round_site::RMS_NORM_ATTEN_INTERNAL,
        )
    }

    pub(crate) fn v_proj_params(&self, shape: &LayerShape) -> MatMulRoundParams {
        MatMulRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.kv_heads * shape.head_dim],
                self.v_proj_acc.clone(),
                self.v_proj.clone(),
                self.v_proj_frac_bits.clone(),
            )
            .with_lookup_site(round_site::V_PROJ),
            MatMulParams::new(
                shape.seq,
                shape.hidden,
                shape.kv_heads * shape.head_dim,
                self.rms_norm_atten_c.clone(),
                self.w_v_proj.clone(),
            ),
        )
    }

    pub(crate) fn rms_norm_mlp_params(&self, shape: &LayerShape) -> RmsNormParams {
        RmsNormParams::new(
            shape.seq,
            shape.hidden,
            self.residual_add_attn_b.clone(),
            "w_rms_norm_mlp",
            self.rms_norm_mlp_acc.clone(),
            self.rms_norm_mlp_a.clone(),
            self.rms_norm_mlp_frac_bits.clone(),
        )
        .with_lookup_sites(round_site::RMS_NORM_MLP, round_site::RMS_NORM_MLP_INTERNAL)
    }

    pub(crate) fn down_proj_params(&self, shape: &LayerShape) -> MatMulRoundParams {
        MatMulRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.hidden],
                self.down_proj_acc.clone(),
                self.down_proj.clone(),
                self.down_proj_frac_bits.clone(),
            )
            .with_lookup_site(round_site::DOWN_PROJ),
            MatMulParams::new(
                shape.seq,
                shape.intermediate,
                shape.hidden,
                self.silu_up.clone(),
                self.w_down_proj.clone(),
            ),
        )
    }

    pub(crate) fn gate_proj_params(&self, shape: &LayerShape) -> MatMulRoundParams {
        MatMulRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.intermediate],
                self.gate_proj_acc.clone(),
                self.gate_proj.clone(),
                self.gate_proj_frac_bits.clone(),
            )
            .with_lookup_site(round_site::GATE_PROJ),
            MatMulParams::new(
                shape.seq,
                shape.hidden,
                shape.intermediate,
                self.rms_norm_mlp_a.clone(),
                self.w_gate_proj.clone(),
            ),
        )
    }

    pub(crate) fn up_proj_params(&self, shape: &LayerShape) -> MatMulRoundParams {
        MatMulRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.intermediate],
                self.up_proj_acc.clone(),
                self.up_proj.clone(),
                self.up_proj_frac_bits.clone(),
            )
            .with_lookup_site(round_site::UP_PROJ),
            MatMulParams::new(
                shape.seq,
                shape.hidden,
                shape.intermediate,
                self.rms_norm_mlp_b.clone(),
                self.w_up_proj.clone(),
            ),
        )
    }

    pub(crate) fn silu_params(&self, shape: &LayerShape) -> SiluRoundParams {
        SiluRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.intermediate],
                self.silu_acc.clone(),
                self.silu.clone(),
                self.silu_out_frac_bits.clone(),
            )
            .with_lookup_site(round_site::SILU_OUTPUT),
            SiluParams::new(
                vec![shape.seq, shape.intermediate],
                self.gate_proj.clone(),
                self.silu_acc.clone(),
                self.silu_ra.clone(),
            ),
        )
    }

    pub(crate) fn silu_up_params(&self, shape: &LayerShape) -> HadamardRoundParams {
        HadamardRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.intermediate],
                self.silu_up_acc.clone(),
                self.silu_up.clone(),
                self.silu_up_frac_bits.clone(),
            )
            .with_lookup_site(round_site::SILU_UP),
            HadamardMulParams::new(
                vec![shape.seq, shape.intermediate],
                self.silu.clone(),
                self.up_proj.clone(),
            ),
        )
    }
}
