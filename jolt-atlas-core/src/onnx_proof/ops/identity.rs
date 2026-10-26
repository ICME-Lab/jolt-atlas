use crate::node_view::ComputationNode;
use crate::{
    impl_tie_proof_api,
    onnx_proof::{ops::neg::SignedCopyRelation, ProofId, Prover},
};
use atlas_algebra::field::JoltField;
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_trace::op::Identity;
use atlas_transcript::Transcript;

impl_tie_proof_api!(Identity, SignedCopyRelation<Identity>);
