use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    ops::{
        activation_clamped::{
            clamped_activation_committed_polynomials, prove_clamped_activation,
            verify_clamped_activation, SmallActivationTable,
        },
        OperatorProofTrait,
    },
    ProofId, Prover, Verifier,
};
use atlas_trace::op::Sigmoid;
use common::{
    consts::{ACTIVATION_TABLE_VARS, MODEL_SCALE},
    CommittedPoly,
};
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

/// Marker implementing [`SmallActivationTable`] for Sigmoid's small dense lookup table.
pub(crate) struct SigmoidTableMarker;

impl SmallActivationTable for SigmoidTableMarker {
    fn materialize() -> Vec<i32> {
        crate::onnx_proof::neural_teleport::utils::materialize_signed_activation_table(
            ACTIVATION_TABLE_VARS,
            1,
            MODEL_SCALE as u8,
            atlas_trace::semantics::tables::TableKind::Sigmoid,
        )
    }
}

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Sigmoid {
    #[tracing::instrument(skip_all, name = "Sigmoid::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        prove_clamped_activation::<F, T, SigmoidTableMarker>(node, prover)
    }

    #[tracing::instrument(skip_all, name = "Sigmoid::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        verify_clamped_activation::<F, T, SigmoidTableMarker>(node, verifier)
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        clamped_activation_committed_polynomials(node)
    }
}
