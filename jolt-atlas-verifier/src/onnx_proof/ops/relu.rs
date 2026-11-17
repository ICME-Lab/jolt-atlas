use crate::node_view::ComputationNode;
use crate::onnx_proof::deferred_lookups::DeferredBatch;
use crate::onnx_proof::lookup::{PrefixSuffixTable, TableRead};
use crate::onnx_proof::ops::OpLayout;
use crate::onnx_proof::plan::{OpPlan, OpRelation, Plan, Step};
use crate::onnx_proof::relation::RelationOpening;
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

impl OpLayout for ReLU {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        Ok(read(node).ra.committed_polys())
    }
}

impl OpPlan for ReLU {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
    ) -> Result<Plan<F>, ProofVerifyError> {
        let read = read(node);
        let ra = read.ra.clone();
        Ok(Plan(vec![
            Step::Reduce,
            Step::Defer {
                batch: DeferredBatch::Lookups,
                relation: OpRelation::TableRead(read),
            },
            Step::Defer {
                batch: DeferredBatch::OneHot,
                relation: OpRelation::OneHot(ra),
            },
        ]))
    }
}
