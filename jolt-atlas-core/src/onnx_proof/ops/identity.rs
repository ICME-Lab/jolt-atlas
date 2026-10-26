use crate::node_view::ComputationNode;
use crate::{
    impl_tie_proof_api,
    onnx_proof::{ops::neg::SignedCopyRelation, ProofId, Prover},
};
use atlas_trace::op::Identity;
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
};

impl_tie_proof_api!(Identity, SignedCopyRelation<Identity>);
