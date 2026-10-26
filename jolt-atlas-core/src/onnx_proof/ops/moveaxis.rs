use crate::node_view::ComputationNode;
use crate::{
    impl_tie_proof_api,
    onnx_proof::{ops::Prover, ProofId},
};
use atlas_algebra::field::JoltField;
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_trace::op::MoveAxis;
use atlas_transcript::Transcript;
pub use jolt_atlas_verifier::onnx_proof::ops::moveaxis::*;

impl_tie_proof_api!(MoveAxis, AxisPermutationRelation);
