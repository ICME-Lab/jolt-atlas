use crate::node_view::ComputationNode;
use crate::onnx_proof::lookup::{PrefixSuffixTable, TableRead};
use crate::onnx_proof::relation::RelationOpening;
use crate::onnx_proof::schedule::{OpPlan, OpRelation, ScheduleBuilder};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::op::ReLU;
use common::{CommittedPoly, VirtualPoly};

/// `ReLU`'s read: the node's output is `ReLU[operand]` at the reduced output point.
pub fn read(node: &ComputationNode) -> TableRead {
    TableRead::of_operand(
        node,
        PrefixSuffixTable::Relu,
        RelationOpening::ReducedNodeOutput(node.idx),
        VirtualPoly::NodeOutputRa,
        CommittedPoly::NodeOutputRaD,
    )
}

impl OpPlan for ReLU {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        let read = read(node);
        let ra = read.ra.clone();
        schedule.reduce();
        schedule.lookup(OpRelation::TableRead(read));
        schedule.one_hot(OpRelation::OneHot(ra));
        Ok(())
    }
}
