use crate::node_view::ComputationNode;
use crate::{
    impl_tie_proof_api,
    onnx_proof::{ProofId, Prover},
};
use atlas_algebra::field::JoltField;
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_trace::op::Neg;
use atlas_transcript::Transcript;
pub use jolt_atlas_verifier::onnx_proof::ops::neg::*;

impl_tie_proof_api!(Neg, SignedCopyRelation<Neg>);
