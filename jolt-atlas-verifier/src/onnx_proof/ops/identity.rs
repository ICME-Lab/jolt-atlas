use crate::node_view::ComputationNode;
use crate::{
    impl_tie_proof_api,
    onnx_proof::{ops::neg::SignedCopyRelation, Verifier},
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_trace::op::Identity;
use atlas_transcript::Transcript;

impl_tie_proof_api!(Identity, SignedCopyRelation<Identity>);
