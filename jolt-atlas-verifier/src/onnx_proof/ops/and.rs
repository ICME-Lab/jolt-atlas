use crate::node_view::ComputationNode;
use crate::onnx_proof::{ops::mul::ProductRelation, ProofId, ProofType, Verifier};
use atlas_trace::op::And;
use joltworks::{field::JoltField, transcripts::Transcript, utils::errors::ProofVerifyError};

use crate::impl_standard_sumcheck_proof_api;

/// `And` on booleans is their product.
pub type AndRelation = ProductRelation<And>;

impl_standard_sumcheck_proof_api!(And, relation AndRelation);
