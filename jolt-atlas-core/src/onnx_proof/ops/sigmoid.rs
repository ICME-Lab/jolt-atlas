use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::ProveOp;
use crate::onnx_proof::{ops::activation_clamped::prove_clamped_activation, ProofId, Prover};
use atlas_algebra::field::JoltField;
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_trace::op::Sigmoid;
use atlas_transcript::Transcript;
pub use jolt_atlas_verifier::onnx_proof::ops::sigmoid::*;

impl<F: JoltField, T: Transcript> ProveOp<F, T> for Sigmoid {
    #[tracing::instrument(skip_all, name = "Sigmoid::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        prove_clamped_activation::<F, T, SigmoidTableMarker>(node, prover)
    }
}
