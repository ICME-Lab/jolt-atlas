use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::{OpLayout, VerifyOp};
use crate::onnx_proof::{
    ops::activation_clamped::{
        clamped_activation_committed_polynomials, verify_clamped_activation, SmallActivationTable,
    },
    Verifier,
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_trace::op::Tanh;
use atlas_transcript::Transcript;
use common::{
    consts::{ACTIVATION_TABLE_VARS, MODEL_SCALE},
    CommittedPoly,
};

/// Marker implementing [`SmallActivationTable`] for Tanh's small dense lookup table.
pub struct TanhTableMarker;

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

impl OpLayout for Tanh {
    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        clamped_activation_committed_polynomials(node)
    }
}

impl<F: JoltField, T: Transcript> VerifyOp<F, T> for Tanh {
    #[tracing::instrument(skip_all, name = "Tanh::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        verify_clamped_activation::<F, T, TanhTableMarker>(node, verifier)
    }
}
