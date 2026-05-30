use joltworks::{
    field::JoltField, poly::commitment::commitment_scheme::CommitmentScheme,
    transcripts::Transcript,
};

use crate::{
    claim::{Claim, Shape},
    ops::{
        hadamard_mul::HadamardRoundProof, matadd::MatAddProof, matmul::MatMulRoundProof,
        pv_matmul::PvMatmulProof, qk_score::QkScoreProof, rms_norm::RmsNormProof, rope::RopeProof,
        silu::SiluRoundProof, softmax::SoftmaxProof,
    },
};

use super::{commitments::LayerCommitments, openings::LayerOpeningReductionProof};

// === Shape =================================================================

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

// === Public layer data =====================================================

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
pub struct LayerClaims<F: JoltField, C = ()> {
    pub hidden_in_a: Claim<F, C>,
    pub hidden_in_b: Claim<F, C>,
    pub direct_eval_claims: Vec<Claim<F, C>>,
    pub down_proj_round_ra: Vec<Claim<F, C>>,
    pub silu_up_round_ra: Vec<Claim<F, C>>,
    pub silu_gate_round_ra: Vec<Claim<F, C>>,
    pub silu_ra: Vec<Claim<F, C>>,
    pub silu_round_ra: Vec<Claim<F, C>>,
    pub gate_proj_round_ra: Vec<Claim<F, C>>,
    pub up_proj_round_ra: Vec<Claim<F, C>>,
    pub rms_norm_mlp_round_ra: Vec<Claim<F, C>>,
    pub rms_norm_mlp_norm_round_ra: Vec<Claim<F, C>>,
    pub o_proj_round_ra: Vec<Claim<F, C>>,
    pub pv_matmul_round_ra: Vec<Claim<F, C>>,
    pub softmax_round_ra: Vec<Claim<F, C>>,
    pub softmax_floor_round_ra: Vec<Claim<F, C>>,
    pub softmax_exp_round_ra: Vec<Claim<F, C>>,
    pub softmax_input_frac_ra: Vec<Claim<F, C>>,
    pub softmax_ra: Vec<Claim<F, C>>,
    pub qk_score_round_ra: Vec<Claim<F, C>>,
    pub qk_score_dot_round_ra: Vec<Claim<F, C>>,
    pub q_rope_round_ra: Vec<Claim<F, C>>,
    pub k_rope_round_ra: Vec<Claim<F, C>>,
    pub q_norm_round_ra: Vec<Claim<F, C>>,
    pub q_norm_norm_round_ra: Vec<Claim<F, C>>,
    pub k_norm_round_ra: Vec<Claim<F, C>>,
    pub k_norm_norm_round_ra: Vec<Claim<F, C>>,
    pub q_proj_round_ra: Vec<Claim<F, C>>,
    pub k_proj_round_ra: Vec<Claim<F, C>>,
    pub v_proj_round_ra: Vec<Claim<F, C>>,
    pub rms_norm_atten_round_ra: Vec<Claim<F, C>>,
    pub rms_norm_atten_norm_round_ra: Vec<Claim<F, C>>,
}

impl<F: JoltField + Clone, C: Clone> LayerClaims<F, C> {
    pub fn boundary_claims(&self) -> Vec<Claim<F, C>> {
        vec![self.hidden_in_a.clone(), self.hidden_in_b.clone()]
    }

    pub fn pcs_claims(&self) -> Vec<Claim<F, C>> {
        let mut out = Vec::new();
        out.extend(self.down_proj_round_ra.clone());
        out.extend(self.silu_up_round_ra.clone());
        out.extend(self.silu_gate_round_ra.clone());
        out.extend(self.silu_ra.clone());
        out.extend(self.silu_round_ra.clone());
        out.extend(self.gate_proj_round_ra.clone());
        out.extend(self.up_proj_round_ra.clone());
        out.extend(self.rms_norm_mlp_round_ra.clone());
        out.extend(self.rms_norm_mlp_norm_round_ra.clone());
        out.extend(self.o_proj_round_ra.clone());
        out.extend(self.pv_matmul_round_ra.clone());
        out.extend(self.softmax_round_ra.clone());
        out.extend(self.softmax_floor_round_ra.clone());
        out.extend(self.softmax_exp_round_ra.clone());
        out.extend(self.softmax_input_frac_ra.clone());
        out.extend(self.softmax_ra.clone());
        out.extend(self.qk_score_round_ra.clone());
        out.extend(self.qk_score_dot_round_ra.clone());
        out.extend(self.q_rope_round_ra.clone());
        out.extend(self.k_rope_round_ra.clone());
        out.extend(self.q_norm_round_ra.clone());
        out.extend(self.q_norm_norm_round_ra.clone());
        out.extend(self.k_norm_round_ra.clone());
        out.extend(self.k_norm_norm_round_ra.clone());
        out.extend(self.q_proj_round_ra.clone());
        out.extend(self.k_proj_round_ra.clone());
        out.extend(self.v_proj_round_ra.clone());
        out.extend(self.rms_norm_atten_round_ra.clone());
        out.extend(self.rms_norm_atten_norm_round_ra.clone());
        out
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LayerIopProof<F: JoltField, T: Transcript> {
    pub residual_add_mlp: MatAddProof<F, T>,
    pub down_proj: MatMulRoundProof<F, T>,
    pub silu_up: HadamardRoundProof<F, T>,
    pub silu: SiluRoundProof<F, T>,
    pub gate_proj: MatMulRoundProof<F, T>,
    pub up_proj: MatMulRoundProof<F, T>,
    pub rms_norm_mlp: RmsNormProof<F, T>,
    pub residual_add_attn: MatAddProof<F, T>,
    pub o_proj: MatMulRoundProof<F, T>,
    pub pv_matmul: PvMatmulProof<F, T>,
    pub softmax: SoftmaxProof<F, T>,
    pub qk_score: QkScoreProof<F, T>,
    pub q_rope: RopeProof<F, T>,
    pub k_rope: RopeProof<F, T>,
    pub q_norm: RmsNormProof<F, T>,
    pub k_norm: RmsNormProof<F, T>,
    pub q_proj: MatMulRoundProof<F, T>,
    pub k_proj: MatMulRoundProof<F, T>,
    pub v_proj: MatMulRoundProof<F, T>,
    pub rms_norm_atten: RmsNormProof<F, T>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerIopRoundStat {
    pub op: &'static str,
    pub sumchecks: usize,
    pub rounds: usize,
}

impl<F: JoltField, T: Transcript> LayerIopProof<F, T> {
    pub(crate) fn round_stats(&self) -> Vec<LayerIopRoundStat> {
        let mut rows = Vec::new();
        let mut push = |op: &'static str, sumchecks: usize, rounds: usize| {
            rows.push(LayerIopRoundStat {
                op,
                sumchecks,
                rounds,
            });
        };

        push(
            "residual_add_mlp",
            self.residual_add_mlp.sumcheck_count(),
            self.residual_add_mlp.sumcheck_round_count(),
        );
        push(
            "down_proj",
            self.down_proj.sumcheck_count(),
            self.down_proj.sumcheck_round_count(),
        );
        push(
            "silu_up",
            self.silu_up.sumcheck_count(),
            self.silu_up.sumcheck_round_count(),
        );
        push(
            "silu",
            self.silu.sumcheck_count(),
            self.silu.sumcheck_round_count(),
        );
        push(
            "gate_proj",
            self.gate_proj.sumcheck_count(),
            self.gate_proj.sumcheck_round_count(),
        );
        push(
            "up_proj",
            self.up_proj.sumcheck_count(),
            self.up_proj.sumcheck_round_count(),
        );
        push(
            "rms_norm_mlp",
            self.rms_norm_mlp.sumcheck_count(),
            self.rms_norm_mlp.sumcheck_round_count(),
        );
        push(
            "residual_add_attn",
            self.residual_add_attn.sumcheck_count(),
            self.residual_add_attn.sumcheck_round_count(),
        );
        push(
            "o_proj",
            self.o_proj.sumcheck_count(),
            self.o_proj.sumcheck_round_count(),
        );
        push(
            "pv_matmul",
            self.pv_matmul.sumcheck_count(),
            self.pv_matmul.sumcheck_round_count(),
        );
        push(
            "softmax",
            self.softmax.sumcheck_count(),
            self.softmax.sumcheck_round_count(),
        );
        push(
            "qk_score",
            self.qk_score.sumcheck_count(),
            self.qk_score.sumcheck_round_count(),
        );
        push(
            "q_rope",
            self.q_rope.sumcheck_count(),
            self.q_rope.sumcheck_round_count(),
        );
        push(
            "k_rope",
            self.k_rope.sumcheck_count(),
            self.k_rope.sumcheck_round_count(),
        );
        push(
            "q_norm",
            self.q_norm.sumcheck_count(),
            self.q_norm.sumcheck_round_count(),
        );
        push(
            "k_norm",
            self.k_norm.sumcheck_count(),
            self.k_norm.sumcheck_round_count(),
        );
        push(
            "q_proj",
            self.q_proj.sumcheck_count(),
            self.q_proj.sumcheck_round_count(),
        );
        push(
            "k_proj",
            self.k_proj.sumcheck_count(),
            self.k_proj.sumcheck_round_count(),
        );
        push(
            "v_proj",
            self.v_proj.sumcheck_count(),
            self.v_proj.sumcheck_round_count(),
        );
        push(
            "rms_norm_atten",
            self.rms_norm_atten.sumcheck_count(),
            self.rms_norm_atten.sumcheck_round_count(),
        );

        rows
    }
}

#[derive(Debug, Clone)]
pub struct LayerProof<F, T, PCS>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    pub(crate) commitments: LayerCommitments<PCS::Commitment>,
    #[allow(dead_code)]
    pub(crate) iop_proof: LayerIopProof<F, T>,
    pub(crate) opening_reduction: LayerOpeningReductionProof<F, T, PCS>,
}
