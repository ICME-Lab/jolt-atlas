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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerClaims<F> {
    pub hidden_in_a: Claim<F>,
    pub hidden_in_b: Claim<F>,
}

impl<F: Clone> LayerClaims<F> {
    pub fn opening_claims(&self) -> Vec<Claim<F>> {
        vec![self.hidden_in_a.clone(), self.hidden_in_b.clone()]
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LayerIopProof<F: JoltField, T: Transcript> {
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

impl<F: JoltField, T: Transcript> LayerIopProof<F, T> {
    pub fn committed_opening_claims(&self) -> Vec<crate::CommittedOpeningClaim<F>> {
        let mut out = Vec::new();
        out.extend(self.down_proj_proof.committed_opening_claims());
        out.extend(self.silu_up_proof.committed_opening_claims());
        out.extend(self.silu_proof.committed_opening_claims());
        out.extend(self.gate_proj_proof.committed_opening_claims());
        out.extend(self.up_proj_proof.committed_opening_claims());
        out.extend(self.rms_norm_mlp_proof.committed_opening_claims());
        out.extend(self.o_proj_proof.committed_opening_claims());
        out.extend(self.pv_matmul_proof.committed_opening_claims());
        out.extend(self.softmax_proof.committed_opening_claims());
        out.extend(self.qk_score_proof.committed_opening_claims());
        out.extend(self.q_rope_proof.committed_opening_claims());
        out.extend(self.k_rope_proof.committed_opening_claims());
        out.extend(self.q_norm_proof.committed_opening_claims());
        out.extend(self.k_norm_proof.committed_opening_claims());
        out.extend(self.q_proj_proof.committed_opening_claims());
        out.extend(self.k_proj_proof.committed_opening_claims());
        out.extend(self.v_proj_proof.committed_opening_claims());
        out.extend(self.rms_norm_atten_proof.committed_opening_claims());
        out
    }
}

#[derive(Debug, Clone)]
pub struct LayerProof<F, T, PCS>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    pub(crate) hidden_out: Claim<F>,
    pub(crate) commitments: LayerCommitments<PCS::Commitment>,
    pub(crate) iop_proof: LayerIopProof<F, T>,
    pub(crate) opening_reduction: LayerOpeningReductionProof<F, T, PCS>,
}

impl<F, T, PCS> LayerProof<F, T, PCS>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn hidden_out(&self) -> &Claim<F> {
        &self.hidden_out
    }
}
