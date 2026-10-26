use crate::node_view::ComputationNode;
use crate::{
    impl_standard_sumcheck_proof_api,
    onnx_proof::{ProofId, ProofType, Prover},
};
use atlas_trace::op::Concat;
pub use jolt_atlas_verifier::onnx_proof::ops::concat::*;
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
};

impl_standard_sumcheck_proof_api!(Concat, model_relation ConcatRelation);
