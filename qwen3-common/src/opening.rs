use ark_bn254::{Bn254, Fr};
use joltworks::{field::JoltField, poly::commitment::hyperkzg::HyperKZGProof};

use crate::{SumCheckRounds, VerifiedSumcheck};

pub const LAYER_OPENING_CHUNK_VARS: usize = 20;
pub const LAYER_OPENING_CHUNK_SIZE: usize = 1 << LAYER_OPENING_CHUNK_VARS;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningReductionProof {
    pub rounds: SumCheckRounds<3>,
    pub evals_at_reduction_point: Vec<Fr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningReductionOutput {
    pub proof: OpeningReductionProof,
    pub reduction_point: Vec<Fr>,
    pub reduction_challenges: Vec<<Fr as JoltField>::Challenge>,
    pub gamma_powers: Vec<Fr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedOpeningReduction {
    pub sumcheck: VerifiedSumcheck,
    pub gamma_powers: Vec<Fr>,
}

#[derive(Debug, Clone)]
pub struct ChunkedLayerPcsOpeningProof {
    pub chunk_evals: Vec<Fr>,
    pub proofs: Vec<HyperKZGProof<Bn254>>,
}
