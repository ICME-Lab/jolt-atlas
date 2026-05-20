use std::time::Instant;

use joltworks::{field::JoltField, transcripts::Transcript, utils::errors::ProofVerifyError};

use crate::{
    claim::{Claim, Shape},
    error::Result,
    ops::hadamard_mul::HadamardMulParams,
    ops::hadamard_round::{
        HadamardRoundParams, HadamardRoundProof, HadamardRoundWitness, prove_hadamard_round,
        verify_hadamard_round,
    },
    ops::matadd::{MatAddParams, MatAddProof, prove_matadd, verify_matadd},
    ops::matmul::MatMulParams,
    ops::matmul_round::{
        MatMulRoundParams, MatMulRoundProof, MatMulRoundWitness, prove_matmul_round,
        verify_matmul_round,
    },
    ops::pv_matmul::{
        PvMatmulParams, PvMatmulProof, PvMatmulRoundParams, PvMatmulWitness, prove_pv_matmul_round,
        verify_pv_matmul_round,
    },
    ops::qk_score::{
        QkScoreParams, QkScoreProof, QkScoreRoundParams, QkScoreWitness, prove_qk_score_round,
        verify_qk_score_round,
    },
    ops::rms_norm::{
        RmsNormParams, RmsNormProof, RmsNormWitness, prove_rmsnorm_round, verify_rmsnorm_round,
    },
    ops::rope::{
        RopeParams, RopeProof, RopeRoundParams, RopeWitness, prove_rope_round, verify_rope_round,
    },
    ops::round::{ROUND_FRAC_BITS, RoundParams},
    ops::silu::SiluParams,
    ops::silu_round::{
        SiluRoundParams, SiluRoundProof, SiluRoundWitness, prove_silu_round, verify_silu_round,
    },
    ops::softmax::{
        SoftmaxParams, SoftmaxProof, SoftmaxWitness, prove_softmax_round, verify_softmax_round,
    },
    proof::ProveResult,
};

// Layer prover skeleton, written in reverse claim flow.
//
// Keep the flow expanded here.  Do not split every node into a tiny layer-level
// helper while the graph is still being designed.  The first step is the final
// MLP residual add:
//
//     hidden_out = residual_add_attn + down_proj
//
// The caller provides the output claims on `hidden_out`.  The add prover batches
// those claims and returns one claim for each add input.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerShape {
    pub seq: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl LayerShape {
    pub fn hidden_shape(&self) -> Shape {
        Shape::new(vec![self.seq, self.hidden])
    }

    pub fn intermediate_shape(&self) -> Shape {
        Shape::new(vec![self.seq, self.intermediate])
    }

    pub fn attention_width(&self) -> usize {
        self.q_heads * self.head_dim
    }
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
            hidden_in_a: "hidden_in_a".to_string(),
            hidden_in_b: "hidden_in_b".to_string(),
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
    fn residual_add_mlp_params(&self, shape: &LayerShape) -> MatAddParams {
        MatAddParams::new(
            vec![shape.seq, shape.hidden],
            self.residual_add_attn_a.clone(),
            self.down_proj.clone(),
        )
    }

    fn residual_add_attn_params(&self, shape: &LayerShape) -> MatAddParams {
        MatAddParams::new(
            vec![shape.seq, shape.hidden],
            self.hidden_in_a.clone(),
            self.o_proj.clone(),
        )
    }

    fn o_proj_params(&self, shape: &LayerShape) -> MatMulRoundParams {
        MatMulRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.hidden],
                self.o_proj_acc.clone(),
                self.o_proj.clone(),
                self.o_proj_frac_bits.clone(),
            ),
            MatMulParams::new(
                shape.seq,
                shape.attention_width(),
                shape.hidden,
                self.context.clone(),
                "w_o_proj",
            ),
        )
    }

    fn pv_matmul_params(&self, shape: &LayerShape) -> PvMatmulRoundParams {
        PvMatmulRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.q_heads, shape.head_dim],
                self.context_acc.clone(),
                self.context.clone(),
                self.context_frac_bits.clone(),
            ),
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

    fn softmax_params(&self, shape: &LayerShape) -> SoftmaxParams {
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
    }

    fn qk_score_params(&self, shape: &LayerShape) -> QkScoreRoundParams {
        QkScoreRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.q_heads, shape.seq, shape.seq],
                self.qk_score_scale_acc.clone(),
                self.qk_score.clone(),
                self.qk_score_frac_bits.clone(),
            ),
            RoundParams::with_frac_bit_tensors(
                vec![shape.q_heads, shape.seq, shape.seq],
                self.qk_score_acc.clone(),
                self.qk_score_dot.clone(),
                self.qk_score_dot_frac_bits.clone(),
            ),
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

    fn q_rope_params(&self, shape: &LayerShape) -> RopeRoundParams {
        RopeRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.q_heads, shape.head_dim],
                self.q_rope_acc.clone(),
                self.q_rope.clone(),
                self.q_rope_frac_bits.clone(),
            ),
            RopeParams::new(
                shape.seq,
                shape.q_heads,
                shape.head_dim,
                self.q_norm.clone(),
            ),
        )
    }

    fn k_rope_params(&self, shape: &LayerShape) -> RopeRoundParams {
        RopeRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.kv_heads, shape.head_dim],
                self.k_rope_acc.clone(),
                self.k_rope.clone(),
                self.k_rope_frac_bits.clone(),
            ),
            RopeParams::new(
                shape.seq,
                shape.kv_heads,
                shape.head_dim,
                self.k_norm.clone(),
            ),
        )
    }

    fn q_norm_params(&self, shape: &LayerShape) -> RmsNormParams {
        RmsNormParams::new_nd(
            vec![shape.seq, shape.q_heads, shape.head_dim],
            self.q_proj.clone(),
            "w_q_norm",
            self.q_norm_acc.clone(),
            self.q_norm.clone(),
            self.q_norm_frac_bits.clone(),
        )
    }

    fn k_norm_params(&self, shape: &LayerShape) -> RmsNormParams {
        RmsNormParams::new_nd(
            vec![shape.seq, shape.kv_heads, shape.head_dim],
            self.k_proj.clone(),
            "w_k_norm",
            self.k_norm_acc.clone(),
            self.k_norm.clone(),
            self.k_norm_frac_bits.clone(),
        )
    }

    fn q_proj_params(&self, shape: &LayerShape) -> MatMulRoundParams {
        MatMulRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.attention_width()],
                self.q_proj_acc.clone(),
                self.q_proj.clone(),
                self.q_proj_frac_bits.clone(),
            ),
            MatMulParams::new(
                shape.seq,
                shape.hidden,
                shape.attention_width(),
                self.rms_norm_atten_a.clone(),
                self.w_q_proj.clone(),
            ),
        )
    }

    fn k_proj_params(&self, shape: &LayerShape) -> MatMulRoundParams {
        MatMulRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.kv_heads * shape.head_dim],
                self.k_proj_acc.clone(),
                self.k_proj.clone(),
                self.k_proj_frac_bits.clone(),
            ),
            MatMulParams::new(
                shape.seq,
                shape.hidden,
                shape.kv_heads * shape.head_dim,
                self.rms_norm_atten_b.clone(),
                self.w_k_proj.clone(),
            ),
        )
    }

    fn rms_norm_atten_params(&self, shape: &LayerShape) -> RmsNormParams {
        RmsNormParams::new(
            shape.seq,
            shape.hidden,
            self.hidden_in_b.clone(),
            "w_rms_norm_atten",
            self.rms_norm_atten_acc.clone(),
            self.rms_norm_atten_a.clone(),
            self.rms_norm_atten_frac_bits.clone(),
        )
    }

    fn v_proj_params(&self, shape: &LayerShape) -> MatMulRoundParams {
        MatMulRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.kv_heads * shape.head_dim],
                self.v_proj_acc.clone(),
                self.v_proj.clone(),
                self.v_proj_frac_bits.clone(),
            ),
            MatMulParams::new(
                shape.seq,
                shape.hidden,
                shape.kv_heads * shape.head_dim,
                self.rms_norm_atten_c.clone(),
                self.w_v_proj.clone(),
            ),
        )
    }

    fn rms_norm_mlp_params(&self, shape: &LayerShape) -> RmsNormParams {
        RmsNormParams::new(
            shape.seq,
            shape.hidden,
            self.residual_add_attn_b.clone(),
            "w_rms_norm_mlp",
            self.rms_norm_mlp_acc.clone(),
            self.rms_norm_mlp_a.clone(),
            self.rms_norm_mlp_frac_bits.clone(),
        )
    }

    fn down_proj_params(&self, shape: &LayerShape) -> MatMulRoundParams {
        MatMulRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.hidden],
                self.down_proj_acc.clone(),
                self.down_proj.clone(),
                self.down_proj_frac_bits.clone(),
            ),
            MatMulParams::new(
                shape.seq,
                shape.intermediate,
                shape.hidden,
                self.silu_up.clone(),
                self.w_down_proj.clone(),
            ),
        )
    }

    fn gate_proj_params(&self, shape: &LayerShape) -> MatMulRoundParams {
        MatMulRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.intermediate],
                self.gate_proj_acc.clone(),
                self.gate_proj.clone(),
                self.gate_proj_frac_bits.clone(),
            ),
            MatMulParams::new(
                shape.seq,
                shape.hidden,
                shape.intermediate,
                self.rms_norm_mlp_a.clone(),
                self.w_gate_proj.clone(),
            ),
        )
    }

    fn up_proj_params(&self, shape: &LayerShape) -> MatMulRoundParams {
        MatMulRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.intermediate],
                self.up_proj_acc.clone(),
                self.up_proj.clone(),
                self.up_proj_frac_bits.clone(),
            ),
            MatMulParams::new(
                shape.seq,
                shape.hidden,
                shape.intermediate,
                self.rms_norm_mlp_b.clone(),
                self.w_up_proj.clone(),
            ),
        )
    }

    fn silu_params(&self, shape: &LayerShape) -> SiluRoundParams {
        SiluRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.intermediate],
                self.silu_acc.clone(),
                self.silu.clone(),
                self.silu_out_frac_bits.clone(),
            ),
            SiluParams::new(
                vec![shape.seq, shape.intermediate],
                self.gate_proj.clone(),
                self.silu_acc.clone(),
                self.silu_ra.clone(),
            ),
        )
    }

    fn silu_up_params(&self, shape: &LayerShape) -> HadamardRoundParams {
        HadamardRoundParams::new(
            RoundParams::with_frac_bit_tensors(
                vec![shape.seq, shape.intermediate],
                self.silu_up_acc.clone(),
                self.silu_up.clone(),
                self.silu_up_frac_bits.clone(),
            ),
            HadamardMulParams::new(
                vec![shape.seq, shape.intermediate],
                self.silu.clone(),
                self.up_proj.clone(),
            ),
        )
    }
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub rope_cos: Vec<i32>,
    pub rope_sin: Vec<i32>,
    pub rms_norm_atten: Vec<i32>,
    pub q_norm: Vec<i32>,
    pub k_norm: Vec<i32>,
    pub rms_norm_mlp: Vec<i32>,
    pub o_proj: Vec<i32>,
    pub q_proj: Vec<i32>,
    pub k_proj: Vec<i32>,
    pub v_proj: Vec<i32>,
    pub gate_proj: Vec<i32>,
    pub up_proj: Vec<i32>,
    pub down_proj: Vec<i32>,
}

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
    fn o_proj_witness(&self) -> MatMulRoundWitness {
        MatMulRoundWitness {
            input: self.context.clone(),
            acc: self.o_proj_acc.clone(),
            output: self.o_proj.clone(),
            frac_bits: self.o_proj_frac_bits.clone(),
        }
    }

    fn softmax_witness(&self) -> SoftmaxWitness {
        SoftmaxWitness {
            input: self.qk_score.clone(),
            max_index: self.softmax_max_index.clone(),
            max: self.softmax_max.clone(),
            min_diff: self.softmax_min_diff,
            max_diff: self.softmax_max_diff,
            ra: self.softmax_ra.clone(),
            exp_acc: self.softmax_exp_acc.clone(),
            exp: self.softmax_exp.clone(),
            exp_frac_bits: self.softmax_exp_frac_bits.clone(),
            sum: self.softmax_sum.clone(),
            acc: self.softmax_acc.clone(),
            floor: self.softmax_floor.clone(),
            floor_frac_bits: self.softmax_floor_frac_bits.clone(),
            output: self.softmax.clone(),
            frac_bits: self.softmax_frac_bits.clone(),
        }
    }

    fn qk_score_witness(&self) -> QkScoreWitness {
        QkScoreWitness {
            q: self.q_rope.clone(),
            k: self.k_rope.clone(),
            raw_acc: self.qk_score_acc.clone(),
            dot: self.qk_score_dot.clone(),
            dot_frac_bits: self.qk_score_dot_frac_bits.clone(),
            scale_acc: self.qk_score_scale_acc.clone(),
            output: self.qk_score.clone(),
            frac_bits: self.qk_score_frac_bits.clone(),
        }
    }

    fn q_rope_witness(&self) -> RopeWitness {
        RopeWitness {
            input: self.q_norm.clone(),
            acc: self.q_rope_acc.clone(),
            output: self.q_rope.clone(),
            frac_bits: self.q_rope_frac_bits.clone(),
        }
    }

    fn k_rope_witness(&self) -> RopeWitness {
        RopeWitness {
            input: self.k_norm.clone(),
            acc: self.k_rope_acc.clone(),
            output: self.k_rope.clone(),
            frac_bits: self.k_rope_frac_bits.clone(),
        }
    }

    fn q_norm_witness(&self) -> RmsNormWitness {
        RmsNormWitness {
            input: self.q_proj.clone(),
            sum_x2: self.q_norm_sum_x2.clone(),
            norm_acc: self.q_norm_norm_acc.clone(),
            norm: self.q_norm_norm.clone(),
            norm_frac_bits: self.q_norm_norm_frac_bits.clone(),
            acc: self.q_norm_acc.clone(),
            output: self.q_norm.clone(),
            frac_bits: self.q_norm_frac_bits.clone(),
        }
    }

    fn k_norm_witness(&self) -> RmsNormWitness {
        RmsNormWitness {
            input: self.k_proj.clone(),
            sum_x2: self.k_norm_sum_x2.clone(),
            norm_acc: self.k_norm_norm_acc.clone(),
            norm: self.k_norm_norm.clone(),
            norm_frac_bits: self.k_norm_norm_frac_bits.clone(),
            acc: self.k_norm_acc.clone(),
            output: self.k_norm.clone(),
            frac_bits: self.k_norm_frac_bits.clone(),
        }
    }

    fn q_proj_witness(&self) -> MatMulRoundWitness {
        MatMulRoundWitness {
            input: self.rms_norm_atten_a.clone(),
            acc: self.q_proj_acc.clone(),
            output: self.q_proj.clone(),
            frac_bits: self.q_proj_frac_bits.clone(),
        }
    }

    fn k_proj_witness(&self) -> MatMulRoundWitness {
        MatMulRoundWitness {
            input: self.rms_norm_atten_b.clone(),
            acc: self.k_proj_acc.clone(),
            output: self.k_proj.clone(),
            frac_bits: self.k_proj_frac_bits.clone(),
        }
    }

    fn v_proj_witness(&self) -> MatMulRoundWitness {
        MatMulRoundWitness {
            input: self.rms_norm_atten_c.clone(),
            acc: self.v_proj_acc.clone(),
            output: self.v_proj.clone(),
            frac_bits: self.v_proj_frac_bits.clone(),
        }
    }

    fn rms_norm_atten_witness(&self) -> RmsNormWitness {
        RmsNormWitness {
            input: self.hidden_in.clone(),
            sum_x2: self.rms_norm_atten_sum_x2.clone(),
            norm_acc: self.rms_norm_atten_norm_acc.clone(),
            norm: self.rms_norm_atten_norm.clone(),
            norm_frac_bits: self.rms_norm_atten_norm_frac_bits.clone(),
            acc: self.rms_norm_atten_acc.clone(),
            output: self.rms_norm_atten_a.clone(),
            frac_bits: self.rms_norm_atten_frac_bits.clone(),
        }
    }

    fn pv_matmul_witness(&self) -> PvMatmulWitness {
        PvMatmulWitness {
            p: self.softmax.clone(),
            v: self.v_proj.clone(),
            acc: self.context_acc.clone(),
            output: self.context.clone(),
            frac_bits: self.context_frac_bits.clone(),
        }
    }

    fn down_proj_witness(&self) -> MatMulRoundWitness {
        MatMulRoundWitness {
            input: self.silu_up.clone(),
            acc: self.down_proj_acc.clone(),
            output: self.down_proj.clone(),
            frac_bits: self.down_proj_frac_bits.clone(),
        }
    }

    fn rms_norm_mlp_witness(&self) -> RmsNormWitness {
        RmsNormWitness {
            input: self.residual_add_attn_b.clone(),
            sum_x2: self.rms_norm_mlp_sum_x2.clone(),
            norm_acc: self.rms_norm_mlp_norm_acc.clone(),
            norm: self.rms_norm_mlp_norm.clone(),
            norm_frac_bits: self.rms_norm_mlp_norm_frac_bits.clone(),
            acc: self.rms_norm_mlp_acc.clone(),
            output: self.rms_norm_mlp_a.clone(),
            frac_bits: self.rms_norm_mlp_frac_bits.clone(),
        }
    }

    fn gate_proj_witness(&self) -> MatMulRoundWitness {
        MatMulRoundWitness {
            input: self.rms_norm_mlp_a.clone(),
            acc: self.gate_proj_acc.clone(),
            output: self.gate_proj.clone(),
            frac_bits: self.gate_proj_frac_bits.clone(),
        }
    }

    fn up_proj_witness(&self) -> MatMulRoundWitness {
        MatMulRoundWitness {
            input: self.rms_norm_mlp_b.clone(),
            acc: self.up_proj_acc.clone(),
            output: self.up_proj.clone(),
            frac_bits: self.up_proj_frac_bits.clone(),
        }
    }

    fn silu_witness(&self) -> SiluRoundWitness {
        SiluRoundWitness {
            min_n: self.silu_min_n,
            max_n: self.silu_max_n,
            gate_proj: self.gate_proj.clone(),
            silu_acc: self.silu_acc.clone(),
            silu_ra: self.silu_ra.clone(),
            silu: self.silu.clone(),
        }
    }

    fn silu_up_witness(&self) -> HadamardRoundWitness {
        HadamardRoundWitness {
            lhs: self.silu.clone(),
            rhs: self.up_proj.clone(),
            acc: self.silu_up_acc.clone(),
            output: self.silu_up.clone(),
            frac_bits: self.silu_up_frac_bits.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerClaims<F> {
    pub hidden_in_a: Claim<F>,
    pub hidden_in_b: Claim<F>,
    pub context_round_ra: Claim<F>,
    pub o_proj_round_ra: Claim<F>,
    pub softmax_round_ra: Claim<F>,
    pub softmax_floor_round_ra: Claim<F>,
    pub softmax_input_remainder_ra: Claim<F>,
    pub softmax_ra: Claim<F>,
    pub qk_score_round_ra: Claim<F>,
    pub qk_score_dot_round_ra: Claim<F>,
    pub q_rope_round_ra: Claim<F>,
    pub k_rope_round_ra: Claim<F>,
    pub q_norm_round_ra: Claim<F>,
    pub k_norm_round_ra: Claim<F>,
    pub q_proj_round_ra: Claim<F>,
    pub k_proj_round_ra: Claim<F>,
    pub v_proj_round_ra: Claim<F>,
    pub rms_norm_atten_round_ra: Claim<F>,
    pub rms_norm_mlp_round_ra: Claim<F>,
    pub gate_proj_round_ra: Claim<F>,
    pub silu_gate_round_ra: Claim<F>,
    pub silu_ra: Claim<F>,
    pub silu_out_round_ra: Claim<F>,
    pub silu_up_round_ra: Claim<F>,
    pub up_proj_round_ra: Claim<F>,
    pub down_proj_round_ra: Claim<F>,
}

#[derive(Debug, Clone)]
pub struct LayerProof<F: JoltField, T: Transcript> {
    pub residual_add_mlp_proof: MatAddProof<F, T>,
    pub down_proj_proof: MatMulRoundProof<F, T>,
    pub silu_up_proof: HadamardRoundProof<F, T>,
    pub silu_proof: SiluRoundProof<F, T>,
    pub gate_proj_proof: MatMulRoundProof<F, T>,
    pub up_proj_proof: MatMulRoundProof<F, T>,
    pub rms_norm_mlp_proof: RmsNormProof<F, T>,
    pub residual_add_attn_proof: MatAddProof<F, T>,
    pub o_proj_proof: MatMulRoundProof<F, T>,
    pub pv_matmul_proof: PvMatmulProof<F, T>,
    pub softmax_proof: SoftmaxProof<F, T>,
    pub qk_score_proof: QkScoreProof<F, T>,
    pub q_rope_proof: RopeProof<F, T>,
    pub k_rope_proof: RopeProof<F, T>,
    pub q_norm_proof: RmsNormProof<F, T>,
    pub k_norm_proof: RmsNormProof<F, T>,
    pub q_proj_proof: MatMulRoundProof<F, T>,
    pub k_proj_proof: MatMulRoundProof<F, T>,
    pub v_proj_proof: MatMulRoundProof<F, T>,
    pub rms_norm_atten_proof: RmsNormProof<F, T>,
}

pub fn prove_layer<F, T>(
    hidden_out_a: Claim<F>,
    hidden_out_b: Claim<F>,
    witness: &LayerWitness,
    weights: &LayerWeights,
    shape: &LayerShape,
    tensors: &LayerTensorIds,
    transcript: &mut T,
) -> Result<ProveResult<LayerClaims<F>, LayerProof<F, T>>>
where
    F: JoltField,
    T: Transcript,
{
    macro_rules! timed {
        ($name:literal, $expr:expr) => {{
            let start = Instant::now();
            let out = $expr;
            eprintln!(
                "timing: prove_layer.{} {:.3}s",
                $name,
                start.elapsed().as_secs_f64()
            );
            out
        }};
    }

    // hidden_out = residual_add_attn + down_proj
    let (residual_add_mlp_proof, residual_add_attn_a, down_proj) = timed!(
        "residual_add_mlp",
        prove_matadd(
            vec![hidden_out_a, hidden_out_b],
            &witness.residual_add_attn_a,
            &witness.down_proj,
            &tensors.residual_add_mlp_params(shape),
            transcript,
        )
    )?;

    // down_proj = silu_up @ W_down
    let (down_proj_proof, silu_up, down_proj_round_ra) = timed!(
        "down_proj",
        prove_matmul_round(
            down_proj,
            &witness.down_proj_witness(),
            &weights.down_proj,
            &tensors.down_proj_params(shape),
            transcript,
        )
    )?;

    // silu_up = silu * up_proj
    let (silu_up_proof, silu, up_proj, silu_up_round_ra) = timed!(
        "silu_up",
        prove_hadamard_round(
            silu_up,
            &witness.silu_up_witness(),
            &tensors.silu_up_params(shape),
            transcript,
        )
    )?;

    // silu = SiLU(gate_proj, frac_bits, ra)
    let (silu_proof, gate_proj, silu_gate_round_ra, silu_ra, silu_out_round_ra) = timed!(
        "silu",
        prove_silu_round(
            silu,
            &witness.silu_witness(),
            &tensors.silu_params(shape),
            transcript,
        )
    )?;

    // gate_proj = rms_norm_mlp @ W_gate
    let (gate_proj_proof, rms_norm_mlp_a, gate_proj_round_ra) = timed!(
        "gate_proj",
        prove_matmul_round(
            gate_proj,
            &witness.gate_proj_witness(),
            &weights.gate_proj,
            &tensors.gate_proj_params(shape),
            transcript,
        )
    )?;

    // up_proj = rms_norm_mlp @ W_up
    let (up_proj_proof, rms_norm_mlp_b, up_proj_round_ra) = timed!(
        "up_proj",
        prove_matmul_round(
            up_proj,
            &witness.up_proj_witness(),
            &weights.up_proj,
            &tensors.up_proj_params(shape),
            transcript,
        )
    )?;

    // rms_norm_mlp = RMSNorm(residual_add_attn)
    let (rms_norm_mlp_proof, residual_add_attn_b, rms_norm_mlp_round_ra) = timed!(
        "rms_norm_mlp",
        prove_rmsnorm_round(
            vec![rms_norm_mlp_a, rms_norm_mlp_b],
            &witness.rms_norm_mlp_witness(),
            &weights.rms_norm_mlp,
            &tensors.rms_norm_mlp_params(shape),
            transcript,
        )
    )?;

    // residual_add_attn = hidden_in_a + o_proj
    let (residual_add_attn_proof, hidden_in_a, o_proj) = timed!(
        "residual_add_attn",
        prove_matadd(
            vec![residual_add_attn_a, residual_add_attn_b],
            &witness.hidden_in,
            &witness.o_proj,
            &tensors.residual_add_attn_params(shape),
            transcript,
        )
    )?;

    // o_proj = context @ W_o
    let (o_proj_proof, context, o_proj_round_ra) = timed!(
        "o_proj",
        prove_matmul_round(
            o_proj,
            &witness.o_proj_witness(),
            &weights.o_proj,
            &tensors.o_proj_params(shape),
            transcript,
        )
    )?;
    let context = reshape_claim(
        context,
        Shape::new(vec![shape.seq, shape.q_heads, shape.head_dim]),
    );

    // context = softmax @ v_proj
    let (pv_matmul_proof, softmax, v_proj, context_round_ra) = timed!(
        "pv_matmul",
        prove_pv_matmul_round(
            context,
            &witness.pv_matmul_witness(),
            &tensors.pv_matmul_params(shape),
            transcript,
        )
    )?;
    let v_proj = reshape_claim(
        v_proj,
        Shape::new(vec![shape.seq, shape.kv_heads * shape.head_dim]),
    );

    // softmax = softmax(qk_score)
    let (
        softmax_proof,
        qk_score,
        softmax_round_ra,
        softmax_floor_round_ra,
        softmax_input_remainder_ra,
        softmax_ra,
    ) = timed!(
        "softmax",
        prove_softmax_round(
            vec![softmax],
            &witness.softmax_witness(),
            &tensors.softmax_params(shape),
            transcript,
        )
    )?;

    // qk_score = round(round(q_rope @ k_rope) * inv_sqrt(head_dim))
    let (qk_score_proof, q_rope, k_rope, qk_score_round_ra, qk_score_dot_round_ra) = timed!(
        "qk_score",
        prove_qk_score_round(
            qk_score,
            &witness.qk_score_witness(),
            &tensors.qk_score_params(shape),
            transcript,
        )
    )?;

    // q_rope = RoPE(q_norm)
    let (q_rope_proof, q_norm_a, q_norm_b, q_rope_round_ra) = timed!(
        "q_rope",
        prove_rope_round(
            q_rope,
            &witness.q_rope_witness(),
            &weights.rope_cos,
            &weights.rope_sin,
            &tensors.q_rope_params(shape),
            transcript,
        )
    )?;

    // k_rope = RoPE(k_norm)
    let (k_rope_proof, k_norm_a, k_norm_b, k_rope_round_ra) = timed!(
        "k_rope",
        prove_rope_round(
            k_rope,
            &witness.k_rope_witness(),
            &weights.rope_cos,
            &weights.rope_sin,
            &tensors.k_rope_params(shape),
            transcript,
        )
    )?;

    // q_norm = RMSNorm(q_proj)
    let (q_norm_proof, q_proj, q_norm_round_ra) = timed!(
        "q_norm",
        prove_rmsnorm_round(
            vec![q_norm_a, q_norm_b],
            &witness.q_norm_witness(),
            &weights.q_norm,
            &tensors.q_norm_params(shape),
            transcript,
        )
    )?;
    let q_proj = reshape_claim(q_proj, Shape::new(vec![shape.seq, shape.attention_width()]));

    // k_norm = RMSNorm(k_proj)
    let (k_norm_proof, k_proj, k_norm_round_ra) = timed!(
        "k_norm",
        prove_rmsnorm_round(
            vec![k_norm_a, k_norm_b],
            &witness.k_norm_witness(),
            &weights.k_norm,
            &tensors.k_norm_params(shape),
            transcript,
        )
    )?;
    let k_proj = reshape_claim(
        k_proj,
        Shape::new(vec![shape.seq, shape.kv_heads * shape.head_dim]),
    );

    // q_proj = rms_norm_atten @ W_q
    let (q_proj_proof, rms_norm_atten_a, q_proj_round_ra) = timed!(
        "q_proj",
        prove_matmul_round(
            q_proj,
            &witness.q_proj_witness(),
            &weights.q_proj,
            &tensors.q_proj_params(shape),
            transcript,
        )
    )?;

    // k_proj = rms_norm_atten @ W_k
    let (k_proj_proof, rms_norm_atten_b, k_proj_round_ra) = timed!(
        "k_proj",
        prove_matmul_round(
            k_proj,
            &witness.k_proj_witness(),
            &weights.k_proj,
            &tensors.k_proj_params(shape),
            transcript,
        )
    )?;

    // v_proj = rms_norm_atten @ W_v
    let (v_proj_proof, rms_norm_atten_c, v_proj_round_ra) = timed!(
        "v_proj",
        prove_matmul_round(
            v_proj,
            &witness.v_proj_witness(),
            &weights.v_proj,
            &tensors.v_proj_params(shape),
            transcript,
        )
    )?;

    // rms_norm_atten = RMSNorm(hidden_in)
    let (rms_norm_atten_proof, hidden_in_b, rms_norm_atten_round_ra) = timed!(
        "rms_norm_atten",
        prove_rmsnorm_round(
            vec![rms_norm_atten_a, rms_norm_atten_b, rms_norm_atten_c],
            &witness.rms_norm_atten_witness(),
            &weights.rms_norm_atten,
            &tensors.rms_norm_atten_params(shape),
            transcript,
        )
    )?;

    Ok(ProveResult::new(
        LayerClaims {
            hidden_in_a,
            hidden_in_b,
            context_round_ra,
            o_proj_round_ra,
            softmax_round_ra,
            softmax_floor_round_ra,
            softmax_input_remainder_ra,
            softmax_ra,
            qk_score_round_ra,
            qk_score_dot_round_ra,
            q_rope_round_ra,
            k_rope_round_ra,
            q_norm_round_ra,
            k_norm_round_ra,
            q_proj_round_ra,
            k_proj_round_ra,
            v_proj_round_ra,
            rms_norm_atten_round_ra,
            rms_norm_mlp_round_ra,
            gate_proj_round_ra,
            silu_gate_round_ra,
            silu_ra,
            silu_out_round_ra,
            silu_up_round_ra,
            up_proj_round_ra,
            down_proj_round_ra,
        },
        LayerProof {
            residual_add_mlp_proof,
            down_proj_proof,
            silu_up_proof,
            silu_proof,
            gate_proj_proof,
            up_proj_proof,
            rms_norm_mlp_proof,
            residual_add_attn_proof,
            o_proj_proof,
            pv_matmul_proof,
            softmax_proof,
            qk_score_proof,
            q_rope_proof,
            k_rope_proof,
            q_norm_proof,
            k_norm_proof,
            q_proj_proof,
            k_proj_proof,
            v_proj_proof,
            rms_norm_atten_proof,
        },
    ))
}

pub fn verify_layer<F, T>(
    hidden_out_a: Claim<F>,
    hidden_out_b: Claim<F>,
    proof: &LayerProof<F, T>,
    weights: &LayerWeights,
    shape: &LayerShape,
    tensors: &LayerTensorIds,
    transcript: &mut T,
) -> std::result::Result<LayerClaims<F>, ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    macro_rules! timed {
        ($name:literal, $expr:expr) => {{
            let start = Instant::now();
            let out = $expr;
            eprintln!(
                "timing: verify_layer.{} {:.3}s",
                $name,
                start.elapsed().as_secs_f64()
            );
            out
        }};
    }

    // hidden_out = residual_add_attn + down_proj
    let (residual_add_attn_a, down_proj) = timed!(
        "residual_add_mlp",
        verify_matadd(
            vec![hidden_out_a, hidden_out_b],
            &proof.residual_add_mlp_proof,
            &tensors.residual_add_mlp_params(shape),
            transcript,
        )
    )?;

    // down_proj = silu_up @ W_down
    let (silu_up, down_proj_round_ra) = timed!(
        "down_proj",
        verify_matmul_round(
            down_proj,
            &proof.down_proj_proof,
            &weights.down_proj,
            &tensors.down_proj_params(shape),
            transcript,
        )
    )?;

    // silu_up = silu * up_proj
    let (silu, up_proj, silu_up_round_ra) = timed!(
        "silu_up",
        verify_hadamard_round(
            silu_up,
            &proof.silu_up_proof,
            &tensors.silu_up_params(shape),
            transcript,
        )
    )?;

    // silu = SiLU(gate_proj, frac_bits, ra)
    let (gate_proj, silu_gate_round_ra, silu_ra, silu_out_round_ra) = timed!(
        "silu",
        verify_silu_round(
            silu,
            &proof.silu_proof,
            &tensors.silu_params(shape),
            transcript,
        )
    )?;

    // gate_proj = rms_norm_mlp @ W_gate
    let (rms_norm_mlp_a, gate_proj_round_ra) = timed!(
        "gate_proj",
        verify_matmul_round(
            gate_proj,
            &proof.gate_proj_proof,
            &weights.gate_proj,
            &tensors.gate_proj_params(shape),
            transcript,
        )
    )?;

    // up_proj = rms_norm_mlp @ W_up
    let (rms_norm_mlp_b, up_proj_round_ra) = timed!(
        "up_proj",
        verify_matmul_round(
            up_proj,
            &proof.up_proj_proof,
            &weights.up_proj,
            &tensors.up_proj_params(shape),
            transcript,
        )
    )?;

    // rms_norm_mlp = RMSNorm(residual_add_attn)
    let (residual_add_attn_b, rms_norm_mlp_round_ra) = timed!(
        "rms_norm_mlp",
        verify_rmsnorm_round(
            vec![rms_norm_mlp_a, rms_norm_mlp_b],
            &proof.rms_norm_mlp_proof,
            &weights.rms_norm_mlp,
            &tensors.rms_norm_mlp_params(shape),
            transcript,
        )
    )?;

    // residual_add_attn = hidden_in_a + o_proj
    let (hidden_in_a, o_proj) = timed!(
        "residual_add_attn",
        verify_matadd(
            vec![residual_add_attn_a, residual_add_attn_b],
            &proof.residual_add_attn_proof,
            &tensors.residual_add_attn_params(shape),
            transcript,
        )
    )?;

    // o_proj = context @ W_o
    let (context, o_proj_round_ra) = timed!(
        "o_proj",
        verify_matmul_round(
            o_proj,
            &proof.o_proj_proof,
            &weights.o_proj,
            &tensors.o_proj_params(shape),
            transcript,
        )
    )?;
    let context = reshape_claim(
        context,
        Shape::new(vec![shape.seq, shape.q_heads, shape.head_dim]),
    );

    // context = softmax @ v_proj
    let (softmax, v_proj, context_round_ra) = timed!(
        "pv_matmul",
        verify_pv_matmul_round(
            context,
            &proof.pv_matmul_proof,
            &tensors.pv_matmul_params(shape),
            transcript,
        )
    )?;
    let v_proj = reshape_claim(
        v_proj,
        Shape::new(vec![shape.seq, shape.kv_heads * shape.head_dim]),
    );

    // softmax = softmax(qk_score)
    let (
        qk_score,
        softmax_round_ra,
        softmax_floor_round_ra,
        softmax_input_remainder_ra,
        softmax_ra,
    ) =
        timed!(
            "softmax",
            verify_softmax_round(
                vec![softmax],
                &proof.softmax_proof,
                &tensors.softmax_params(shape),
                transcript,
            )
        )?;

    // qk_score = round(round(q_rope @ k_rope) * inv_sqrt(head_dim))
    let (q_rope, k_rope, qk_score_round_ra, qk_score_dot_round_ra) = timed!(
        "qk_score",
        verify_qk_score_round(
            qk_score,
            &proof.qk_score_proof,
            &tensors.qk_score_params(shape),
            transcript,
        )
    )?;

    // q_rope = RoPE(q_norm)
    let (q_norm_a, q_norm_b, q_rope_round_ra) = timed!(
        "q_rope",
        verify_rope_round(
            q_rope,
            &proof.q_rope_proof,
            &weights.rope_cos,
            &weights.rope_sin,
            &tensors.q_rope_params(shape),
            transcript,
        )
    )?;

    // k_rope = RoPE(k_norm)
    let (k_norm_a, k_norm_b, k_rope_round_ra) = timed!(
        "k_rope",
        verify_rope_round(
            k_rope,
            &proof.k_rope_proof,
            &weights.rope_cos,
            &weights.rope_sin,
            &tensors.k_rope_params(shape),
            transcript,
        )
    )?;

    // q_norm = RMSNorm(q_proj)
    let (q_proj, q_norm_round_ra) = timed!(
        "q_norm",
        verify_rmsnorm_round(
            vec![q_norm_a, q_norm_b],
            &proof.q_norm_proof,
            &weights.q_norm,
            &tensors.q_norm_params(shape),
            transcript,
        )
    )?;
    let q_proj = reshape_claim(q_proj, Shape::new(vec![shape.seq, shape.attention_width()]));

    // k_norm = RMSNorm(k_proj)
    let (k_proj, k_norm_round_ra) = timed!(
        "k_norm",
        verify_rmsnorm_round(
            vec![k_norm_a, k_norm_b],
            &proof.k_norm_proof,
            &weights.k_norm,
            &tensors.k_norm_params(shape),
            transcript,
        )
    )?;
    let k_proj = reshape_claim(
        k_proj,
        Shape::new(vec![shape.seq, shape.kv_heads * shape.head_dim]),
    );

    // q_proj = rms_norm_atten @ W_q
    let (rms_norm_atten_a, q_proj_round_ra) = timed!(
        "q_proj",
        verify_matmul_round(
            q_proj,
            &proof.q_proj_proof,
            &weights.q_proj,
            &tensors.q_proj_params(shape),
            transcript,
        )
    )?;

    // k_proj = rms_norm_atten @ W_k
    let (rms_norm_atten_b, k_proj_round_ra) = timed!(
        "k_proj",
        verify_matmul_round(
            k_proj,
            &proof.k_proj_proof,
            &weights.k_proj,
            &tensors.k_proj_params(shape),
            transcript,
        )
    )?;

    // v_proj = rms_norm_atten @ W_v
    let (rms_norm_atten_c, v_proj_round_ra) = timed!(
        "v_proj",
        verify_matmul_round(
            v_proj,
            &proof.v_proj_proof,
            &weights.v_proj,
            &tensors.v_proj_params(shape),
            transcript,
        )
    )?;

    // rms_norm_atten = RMSNorm(hidden_in)
    let (hidden_in_b, rms_norm_atten_round_ra) = timed!(
        "rms_norm_atten",
        verify_rmsnorm_round(
            vec![rms_norm_atten_a, rms_norm_atten_b, rms_norm_atten_c],
            &proof.rms_norm_atten_proof,
            &weights.rms_norm_atten,
            &tensors.rms_norm_atten_params(shape),
            transcript,
        )
    )?;

    Ok(LayerClaims {
        hidden_in_a,
        hidden_in_b,
        context_round_ra,
        o_proj_round_ra,
        softmax_round_ra,
        softmax_floor_round_ra,
        softmax_input_remainder_ra,
        softmax_ra,
        qk_score_round_ra,
        qk_score_dot_round_ra,
        q_rope_round_ra,
        k_rope_round_ra,
        q_norm_round_ra,
        k_norm_round_ra,
        q_proj_round_ra,
        k_proj_round_ra,
        v_proj_round_ra,
        rms_norm_atten_round_ra,
        rms_norm_mlp_round_ra,
        gate_proj_round_ra,
        silu_gate_round_ra,
        silu_ra,
        silu_out_round_ra,
        silu_up_round_ra,
        up_proj_round_ra,
        down_proj_round_ra,
    })
}

fn reshape_claim<F>(mut claim: Claim<F>, logical_shape: Shape) -> Claim<F> {
    // Projection ops use flattened `[seq, heads * head_dim]` tensors while
    // attention ops use `[seq, heads, head_dim]`.  For Qwen these split
    // dimensions are powers of two, so the row-major MLE point is the same
    // concatenation of bits and this shape bridge needs no sumcheck.
    claim.logical_shape = logical_shape.clone();
    claim.domain_shape = logical_shape.padded_power_of_two();
    claim
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use joltworks::{
        field::JoltField, poly::eq_poly::EqPolynomial, transcripts::Blake2bTranscript,
    };

    use super::*;
    use crate::claim::TensorId;

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
        let points = [
            vec![Fr::from(3u64), Fr::from(5u64)],
            vec![Fr::from(7u64), Fr::from(11u64)],
        ];
        let hidden_shape = shape.hidden_shape();
        let hidden_out_claims = points
            .iter()
            .map(|point| Claim {
                tensor: TensorId::new("hidden_out"),
                logical_shape: hidden_shape.clone(),
                domain_shape: hidden_shape.padded_power_of_two(),
                point: point.clone(),
                value: eval_tensor(&hidden_out, &hidden_shape, point),
            })
            .collect::<Vec<_>>();

        let mut prover_transcript = Blake2bTranscript::default();
        let result = prove_layer::<Fr, _>(
            hidden_out_claims[0].clone(),
            hidden_out_claims[1].clone(),
            &witness,
            &weights,
            &shape,
            &tensors,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        let claims = verify_layer::<Fr, _>(
            hidden_out_claims[0].clone(),
            hidden_out_claims[1].clone(),
            &result.proof,
            &weights,
            &shape,
            &tensors,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(claims, result.claims);
        assert_eq!(claims.hidden_in_a.tensor.0, "hidden_in_a");
        assert_eq!(claims.hidden_in_b.tensor.0, "hidden_in_b");
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

    fn eval_tensor<F: JoltField>(values: &[i32], shape: &Shape, point: &[F]) -> F {
        let eq_by_dim = split_point(shape, point)
            .into_iter()
            .map(EqPolynomial::<F>::evals)
            .collect::<Vec<_>>();
        let strides = row_major_strides(shape.dims());
        let mut out = F::zero();
        for (flat, &value) in values.iter().enumerate() {
            let mut weight = F::one();
            for (dim, (&stride, eq)) in strides.iter().zip(&eq_by_dim).enumerate() {
                let coord = (flat / stride) % shape.dims()[dim];
                weight *= eq[coord];
            }
            out += weight * F::from_i32(value);
        }
        out
    }

    fn split_point<'a, F>(shape: &Shape, point: &'a [F]) -> Vec<&'a [F]> {
        let mut out = Vec::with_capacity(shape.dims().len());
        let mut offset = 0;
        for &dim in shape.dims() {
            let vars = dim.next_power_of_two().trailing_zeros() as usize;
            out.push(&point[offset..offset + vars]);
            offset += vars;
        }
        out
    }

    fn row_major_strides(dims: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; dims.len()];
        let mut stride = 1;
        for (idx, &dim) in dims.iter().enumerate().rev() {
            strides[idx] = stride;
            stride *= dim;
        }
        strides
    }
}
