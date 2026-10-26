use crate::node_view::ComputationNode;
use crate::{
    impl_tie_proof_api,
    onnx_proof::{ops::Prover, ProofId},
};
use atlas_trace::op::MoveAxis;
pub use jolt_atlas_verifier::onnx_proof::ops::moveaxis::*;
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
};

impl_tie_proof_api!(MoveAxis, AxisPermutationRelation);
