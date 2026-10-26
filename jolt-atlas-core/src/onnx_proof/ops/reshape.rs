//! Reshape proof using an eq-selector over the padded input domain.
//!
//! For each raw input element `t`, we compute the padded output index that the
//! same element occupies after reshape and place `eq(r_output, mapped_index(t))`
//! at the corresponding padded input position. The resulting selector MLE
//! satisfies:
//!
//! `out(r_output) = sum_i selector(i) * input(i)`.

use atlas_trace::op::Reshape;
pub use jolt_atlas_verifier::onnx_proof::ops::reshape::*;

use crate::node_view::ComputationNode;
use crate::{
    impl_standard_sumcheck_proof_api,
    onnx_proof::{ProofId, ProofType, Prover},
};
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
};

impl_standard_sumcheck_proof_api!(Reshape, model_relation ReshapeRelation);
