use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::ProveOp;
use crate::onnx_proof::{ops::Prover, relation::prove_tie, ProofId};
use atlas_algebra::field::JoltField;
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_trace::op::Broadcast;
use atlas_transcript::Transcript;
pub use jolt_atlas_verifier::onnx_proof::ops::broadcast::*;

impl<F: JoltField, T: Transcript> ProveOp<F, T> for Broadcast {
    #[tracing::instrument(skip_all, name = "Broadcast::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        prove_tie(
            &BroadcastRelation::for_node(&prover.preprocessing.model, node.clone()),
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        Vec::new()
    }
}
