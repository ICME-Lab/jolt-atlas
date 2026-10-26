use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::{OpLayout, VerifyOp};
use crate::onnx_proof::{
    ops::activation_clamped::{
        clamped_activation_committed_polynomials, verify_clamped_activation, SmallActivationTable,
    },
    Verifier,
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_trace::op::Sigmoid;
use atlas_transcript::Transcript;
use common::{
    consts::{ACTIVATION_TABLE_VARS, MODEL_SCALE},
    CommittedPoly,
};

/// Marker implementing [`SmallActivationTable`] for Sigmoid's small dense lookup table.
pub struct SigmoidTableMarker;

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

impl OpLayout for Sigmoid {
    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        clamped_activation_committed_polynomials(node)
    }
}

impl<F: JoltField, T: Transcript> VerifyOp<F, T> for Sigmoid {
    #[tracing::instrument(skip_all, name = "Sigmoid::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        verify_clamped_activation::<F, T, SigmoidTableMarker>(node, verifier)
    }
}
