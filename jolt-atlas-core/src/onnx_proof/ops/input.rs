use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::{OpLayout, ProveOp, VerifyOp};
use crate::onnx_proof::trace_view::TraceView;
use crate::onnx_proof::{
    ops::constant::{PublicRelation, PublicShape},
    relation::{prove_tie, verify_tie},
    ProofId, Prover, Verifier,
};
use atlas_trace::{op::Input, trace::LayerData};
use jolt_claims::SymbolicSumcheck;
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use std::sync::Arc;

crate::bind_relation!(Input, PublicRelation<Input>);

impl OpLayout for Input {}

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

impl<F: JoltField, T: Transcript> VerifyOp<F, T> for Input {
    #[tracing::instrument(skip_all, name = "Input::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let position = verifier
            .io
            .input_indices
            .iter()
            .position(|&idx| idx == node.idx)
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let shape = PublicShape::new(node.clone(), Arc::new(verifier.io.inputs[position].clone()));
        verify_tie(
            &PublicRelation::<Input>::new(shape),
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )
    }
}
