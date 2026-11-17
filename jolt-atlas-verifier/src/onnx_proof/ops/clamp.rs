use crate::node_view::ComputationNode;
use crate::onnx_proof::lookup::{PrefixSuffixTable, TableRead};
use crate::onnx_proof::relation::RelationOpening;
use crate::onnx_proof::schedule::{OpPlan, OpRelation, ScheduleBuilder};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_lookup_tables::clamp::CLAMP_BOUND;
use atlas_trace::op::Clamp;
use common::{CommittedPoly, VirtualPoly};

/// `Clamp`'s read: the node's output is its operand read in `atlas_lookup_tables::clamp`'s
/// symmetric clamp table (`[-2^CLAMP_BOUND, 2^CLAMP_BOUND - 1]`) at the reduced output point.
pub fn read(node: &ComputationNode) -> TableRead {
    TableRead::of_operand(
        node,
        PrefixSuffixTable::Clamp,
        RelationOpening::ReducedNodeOutput(node.idx),
        VirtualPoly::SymmetricClampRa,
        CommittedPoly::SymmetricClampRaD,
    )
}

impl OpPlan for Clamp {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        // The clamp table is the symmetric one at this bound.
        if self.bound_log as usize != CLAMP_BOUND {
            return Err(ProofVerifyError::Malformed(format!(
                "model: clamp node {} has bound 2^{}, the table's is 2^{CLAMP_BOUND}",
                node.idx, self.bound_log
            )));
        }
        let read = read(node);
        let ra = read.ra.clone();
        schedule.reduce();
        schedule.lookup(OpRelation::TableRead(read));
        schedule.one_hot(OpRelation::OneHot(ra));
        Ok(())
    }
}
