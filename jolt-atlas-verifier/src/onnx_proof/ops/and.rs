use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    ops::mul::product,
    relation::OpRelKind,
    schedule::{OpPlan, OpRelation, ScheduleBuilder},
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_trace::op::And;

/// `And` on booleans is their product ([`product`]).
impl OpPlan for And {
    fn plan<F: JoltField>(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        schedule.reduce();
        schedule.relation(OpRelation::Op(product(node, OpRelKind::And, &[0, 1])));
        Ok(())
    }
}
