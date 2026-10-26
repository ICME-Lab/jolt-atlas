use crate::node_view::ComputationNode;
use crate::{
    impl_tie_proof_api,
    onnx_proof::{ProofId, Prover},
};
use atlas_trace::op::IsNan;
pub use jolt_atlas_verifier::onnx_proof::ops::is_nan::*;
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
};

impl_tie_proof_api!(IsNan, ZeroOutputRelation);
