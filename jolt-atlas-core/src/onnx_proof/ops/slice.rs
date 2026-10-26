use crate::node_view::ComputationNode;
use crate::{
    impl_standard_sumcheck_proof_api,
    onnx_proof::{ProofId, ProofType, Prover},
};
use atlas_algebra::field::JoltField;
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_trace::op::Slice;
use atlas_transcript::Transcript;
pub use jolt_atlas_verifier::onnx_proof::ops::slice::*;

impl_standard_sumcheck_proof_api!(Slice, model_relation SliceRelation);
