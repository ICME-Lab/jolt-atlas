use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::ProveOp;
use crate::onnx_proof::trace_view::TraceView;
use crate::onnx_proof::{
    ops::constant::{PublicRelation, PublicShape},
    relation::prove_tie,
    ProofId, Prover,
};
use atlas_algebra::field::JoltField;
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_trace::{op::Input, trace::LayerData};
use atlas_transcript::Transcript;
use jolt_claims::SymbolicSumcheck;
use std::sync::Arc;

impl<F: JoltField, T: Transcript> ProveOp<F, T> for Input {
    #[tracing::instrument(skip_all, name = "Input::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        let LayerData { output, .. } = TraceView::layer_data(&prover.trace, node);
        let shape = PublicShape::new(node.clone(), Arc::new(output.clone()));
        prove_tie(
            &PublicRelation::<Input>::new(shape),
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        Vec::new()
    }
}
