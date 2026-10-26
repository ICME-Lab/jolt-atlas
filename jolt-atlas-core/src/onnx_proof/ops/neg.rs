use crate::node_view::ComputationNode;
use crate::{
    impl_tie_proof_api,
    onnx_proof::{ProofId, Prover},
};
use atlas_trace::op::Neg;
pub use jolt_atlas_verifier::onnx_proof::ops::neg::*;
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
};

impl_tie_proof_api!(Neg, SignedCopyRelation<Neg>);
