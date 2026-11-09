use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::OpLayout;
use crate::onnx_proof::{
    ops::activation_clamped::{
        clamped_activation_committed_polynomials, clamped_activation_plan, SmallActivationTable,
    },
    plan::{OpActivation, OpPlan, Plan},
};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::{op::Erf, trace::ModelExecutionIO, Model};
use common::{
    consts::{ACTIVATION_TABLE_VARS, MODEL_SCALE},
    CommittedPoly,
};

/// Marker implementing [`SmallActivationTable`] for Erf's small dense lookup table.
pub struct ErfTableMarker;

impl SmallActivationTable for ErfTableMarker {
    fn materialize() -> Result<Vec<i32>, ProofVerifyError> {
        crate::onnx_proof::neural_teleport::utils::materialize_signed_activation_table(
            ACTIVATION_TABLE_VARS,
            1,
            MODEL_SCALE as u8,
            atlas_trace::semantics::tables::TableKind::Erf,
        )
    }
}

impl OpLayout for Erf {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        Ok(clamped_activation_committed_polynomials(node))
    }
}

impl OpPlan for Erf {
    fn plan(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(clamped_activation_plan(node, OpActivation::Erf))
    }
}
