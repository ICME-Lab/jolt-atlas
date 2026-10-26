use crate::node_view::ComputationNode;
use crate::{
    impl_fused_rescale_proof_api,
    onnx_proof::{ProofId, ProofType, Prover},
};
use atlas_algebra::field::JoltField;
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_trace::op::Mul;
use atlas_transcript::Transcript;
pub use jolt_atlas_verifier::onnx_proof::ops::mul::*;

impl_fused_rescale_proof_api!(Mul, relation MulRelation);
