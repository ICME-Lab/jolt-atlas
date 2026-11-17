use super::*;
use crate::onnx_proof::lookup::RaEncoding;
use crate::onnx_proof::{
    schedule::{OpPlan, OpRelation, ScheduleBuilder},
    ProofType,
};
use atlas_trace::op::GatherLarge;
use common::CommittedPoly;

/// The plan: the evaluation reduction, the dictionary read, then the one-hot checks of its
/// read addresses, proven in the one-hot sumcheck.
impl OpPlan for GatherLarge {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        model: &Model,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        schedule.reduce();
        schedule.relation(ProofType::Execution, OpRelation::Op(gather(model, node)?));
        schedule.one_hot(OpRelation::OneHot(ra_encoding(node)?));
        Ok(())
    }
}

/// The one-hot encoding of a large gather's read addresses: the cycle point is the index
/// operand's opening's. `Err` if the node is not a large gather with an index operand.
pub fn ra_encoding(node: &ComputationNode) -> Result<RaEncoding, ProofVerifyError> {
    let Op::GatherLarge(gather_op) = &node.operator else {
        return Err(unsupported(node, "expected a GatherLarge operator"));
    };
    if node.inputs.get(1).is_none() {
        return Err(unsupported(node, "no index operand"));
    }
    let ids = OpeningIdBuilder::new(node);
    Ok(RaEncoding::chunked(
        node.idx,
        CommittedPoly::GatherRaD,
        ids.advice(VirtualPoly::NodeOutputRa),
        ids.nodeio(Target::Input(1)),
        gather_op.dict_len.next_power_of_two().log_2(),
    ))
}
