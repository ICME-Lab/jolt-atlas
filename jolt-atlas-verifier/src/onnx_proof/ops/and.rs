use crate::node_view::ComputationNode;
use crate::onnx_proof::{ops::mul::ProductRelation, ProofId, ProofType, Verifier};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_trace::op::And;
use atlas_transcript::Transcript;

use crate::impl_standard_sumcheck_proof_api;

/// `And` on booleans is their product.
pub type AndRelation = ProductRelation<And>;

impl_standard_sumcheck_proof_api!(And, relation AndRelation);
