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
            &BroadcastRelation::for_node(&prover.preprocessing.model, node.clone())
                .expect("the model\'s nodes have the shapes their ops prove"),
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        Vec::new()
    }
}

use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, TieClaims};
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::poly::multilinear_polynomial::EvaluateMle;
use atlas_sumcheck::opening_proof::OpeningAccumulator;
use atlas_trace::Trace;

impl<F: JoltField> TieClaims<F> for BroadcastRelation {
    fn claims(
        &self,
        trace: &Trace,
        setup: &RelationSetup<F>,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        let operand = TraceView::operand_tensors(trace, self.node())[0];
        let (r_input, _) = self
            .split(&setup.incoming_point)
            .expect("the incoming point is a point of the node's padded domain");
        vec![operand.evaluate_mle(&r_input)]
    }
}
