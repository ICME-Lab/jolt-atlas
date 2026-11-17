use super::*;
use crate::onnx_proof::deferred_lookups::DeferredBatch;
use crate::onnx_proof::lookup::RaEncoding;
use crate::onnx_proof::{
    ops::OpLayout,
    plan::{OpPlan, OpRelation, Plan, Step},
    ProofType,
};
use atlas_trace::op::GatherLarge;
use atlas_trace::trace::ModelExecutionIO;
use common::CommittedPoly;

impl OpLayout for GatherLarge {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        Ok(ra_encoding(node)?.committed_polys())
    }
}

/// The plan: the evaluation reduction, the dictionary read, then the one-hot checks of its
/// read addresses, queued for the post-loop one-hot batch.
impl OpPlan for GatherLarge {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan<F>, ProofVerifyError> {
        Ok(Plan(vec![
            Step::Reduce,
            Step::Relation {
                proof: ProofType::Execution,
                relation: OpRelation::Op(gather(model, node)?),
            },
            Step::Defer {
                batch: DeferredBatch::OneHot,
                relation: OpRelation::OneHot(ra_encoding(node)?),
            },
        ]))
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
