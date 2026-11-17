use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    ops::{add::accumulation_plan, OpLayout},
    relation::OpRelKind,
    schedule::{OpPlan, ScheduleBuilder},
};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::{op::Sub, trace::ModelExecutionIO, Model};

impl OpLayout for Sub {}

/// Proof for exact element-wise subtraction: `output = left - right`, where a difference
/// outside `i32` is an execution error.
///
/// Identical in structure to [`Add`](super::add) but with a subtraction tie: the operand tie
/// ([`accumulation`](super::add::accumulation)) checks `left(r) - right(r) == output(r)` at
/// the reduced output point, and the output's range claim is in the packed range buckets. See
/// [`Add`](super::add) for the full breakdown.
impl OpPlan for Sub {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        accumulation_plan(node, OpRelKind::SubAccumulation, schedule);
        Ok(())
    }
}
