use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::ProveOp;
use crate::onnx_proof::{ops::activation_clamped::prove_clamped_activation, ProofId, Prover};
use atlas_algebra::field::JoltField;
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_trace::op::Tanh;
use atlas_transcript::Transcript;
pub use jolt_atlas_verifier::onnx_proof::ops::tanh::*;

impl<F: JoltField, T: Transcript> ProveOp<F, T> for Tanh {
    #[tracing::instrument(skip_all, name = "Tanh::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        prove_clamped_activation::<F, T, TanhTableMarker>(node, prover)
    }
}
