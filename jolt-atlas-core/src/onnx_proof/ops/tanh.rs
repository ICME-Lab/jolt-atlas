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
use atlas_trace::op::Tanh;
use common::{
    consts::{ACTIVATION_TABLE_VARS, MODEL_SCALE},
    CommittedPoly,
};
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

/// Marker implementing [`SmallActivationTable`] for Tanh's small dense lookup table.
pub(crate) struct TanhTableMarker;

impl SmallActivationTable for TanhTableMarker {
    fn materialize() -> Vec<i32> {
        crate::onnx_proof::neural_teleport::utils::materialize_signed_activation_table(
            ACTIVATION_TABLE_VARS,
            1,
            MODEL_SCALE as u8,
            atlas_trace::semantics::tables::TableKind::Tanh,
        )
    }
}

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Tanh {
    #[tracing::instrument(skip_all, name = "Tanh::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        prove_clamped_activation::<F, T, TanhTableMarker>(node, prover)
    }

    #[tracing::instrument(skip_all, name = "Tanh::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        verify_clamped_activation::<F, T, TanhTableMarker>(node, verifier)
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        clamped_activation_committed_polynomials(node)
    }
}
