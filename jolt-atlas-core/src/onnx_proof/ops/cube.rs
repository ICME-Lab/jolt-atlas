use crate::node_view::ComputationNode;
use crate::{
    impl_fused_rescale_proof_api,
    onnx_proof::{ops::mul::ProductRelation, ProofId, ProofType, Prover, Verifier},
};
use atlas_trace::op::Cube;
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

/// Element-wise cube as a product relation.
pub type CubeRelation = ProductRelation<Cube>;

impl_fused_rescale_proof_api!(Cube, relation CubeRelation);
