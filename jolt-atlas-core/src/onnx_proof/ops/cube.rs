use crate::node_view::ComputationNode;
use crate::{
    impl_fused_rescale_proof_api,
    onnx_proof::{ProofId, ProofType, Prover},
};
use atlas_trace::op::Cube;
pub use jolt_atlas_verifier::onnx_proof::ops::cube::*;
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
};

impl_fused_rescale_proof_api!(Cube, relation CubeRelation);
