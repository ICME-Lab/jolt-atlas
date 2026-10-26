use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::ProveOp;
use crate::onnx_proof::{ops::add::prove_accumulation, ProofId, Prover};
use atlas_algebra::field::JoltField;
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_trace::op::Sub;
use atlas_transcript::Transcript;

impl<F: JoltField, T: Transcript> ProveOp<F, T> for Sub {
    #[tracing::instrument(skip_all, name = "Sub::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        prove_accumulation::<Sub, F, T>(node, prover)
    }
}
