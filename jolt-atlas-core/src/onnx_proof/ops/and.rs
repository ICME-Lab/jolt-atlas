use crate::node_view::ComputationNode;
use crate::onnx_proof::{ProofId, ProofType, Prover};
use atlas_trace::op::And;
pub use jolt_atlas_verifier::onnx_proof::ops::and::*;
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
};

use crate::impl_standard_sumcheck_proof_api;

impl_standard_sumcheck_proof_api!(And, relation AndRelation);
