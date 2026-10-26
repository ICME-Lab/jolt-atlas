use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::ProveOp;
use crate::onnx_proof::{ops::activation_clamped::prove_clamped_activation, ProofId, Prover};
use atlas_trace::op::Erf;
pub use jolt_atlas_verifier::onnx_proof::ops::erf::*;
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
};

impl<F: JoltField, T: Transcript> ProveOp<F, T> for Erf {
    #[tracing::instrument(skip_all, name = "Erf::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        prove_clamped_activation::<F, T, ErfTableMarker>(node, prover)
    }
}
