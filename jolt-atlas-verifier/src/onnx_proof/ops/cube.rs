use crate::node_view::ComputationNode;
use crate::{
    impl_fused_rescale_proof_api,
    onnx_proof::{ops::mul::ProductRelation, ProofId, ProofType, Verifier},
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_trace::op::Cube;
use atlas_transcript::Transcript;

/// Element-wise cube as a product relation.
pub type CubeRelation = ProductRelation<Cube>;

impl_fused_rescale_proof_api!(Cube, relation CubeRelation);
