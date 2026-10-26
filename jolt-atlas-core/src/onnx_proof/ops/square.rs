use crate::node_view::ComputationNode;
use crate::{
    impl_fused_rescale_proof_api,
    onnx_proof::{ProofId, ProofType, Prover},
};
use atlas_trace::op::Square;
pub use jolt_atlas_verifier::onnx_proof::ops::square::*;
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
};

impl_fused_rescale_proof_api!(Square, relation SquareRelation);
