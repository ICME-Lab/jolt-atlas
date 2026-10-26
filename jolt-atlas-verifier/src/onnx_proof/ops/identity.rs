use crate::node_view::ComputationNode;
use crate::{
    impl_tie_proof_api,
    onnx_proof::{ops::neg::SignedCopyRelation, Verifier},
};
use atlas_trace::op::Identity;
use joltworks::{field::JoltField, transcripts::Transcript, utils::errors::ProofVerifyError};

impl_tie_proof_api!(Identity, SignedCopyRelation<Identity>);
