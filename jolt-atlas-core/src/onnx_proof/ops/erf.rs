use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::{OpLayout, ProveOp, VerifyOp};
use crate::onnx_proof::{
    ops::activation_clamped::{
        clamped_activation_committed_polynomials, prove_clamped_activation,
        verify_clamped_activation, SmallActivationTable,
    },
    ProofId, Prover, Verifier,
};
use atlas_trace::op::Erf;
use common::{
    consts::{ACTIVATION_TABLE_VARS, MODEL_SCALE},
    CommittedPoly,
};
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

/// Marker implementing [`SmallActivationTable`] for Erf's small dense lookup table.
pub(crate) struct ErfTableMarker;

impl SmallActivationTable for ErfTableMarker {
    fn materialize() -> Vec<i32> {
        crate::onnx_proof::neural_teleport::utils::materialize_signed_activation_table(
            ACTIVATION_TABLE_VARS,
            1,
            MODEL_SCALE as u8,
            atlas_trace::semantics::tables::TableKind::Erf,
        )
    }
}

impl OpLayout for Erf {
    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        clamped_activation_committed_polynomials(node)
    }
}

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

impl<F: JoltField, T: Transcript> VerifyOp<F, T> for Erf {
    #[tracing::instrument(skip_all, name = "Erf::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        verify_clamped_activation::<F, T, ErfTableMarker>(node, verifier)
    }
}
