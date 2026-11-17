use crate::node_view::ComputationNode;
use crate::onnx_proof::schedule::{OpPlan, OpRelation, ScheduleBuilder};
use crate::{
    onnx_proof::{
        ops::OpLayout,
        relation::{OpRel, OpRelKind, RelationDerived, RelationOpening},
        ProofType,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{
    field::JoltField,
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_claims::{constant, derived, opening, Int};
use atlas_trace::op::Iff;

impl OpLayout for Iff {}

impl OpPlan for Iff {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        schedule.reduce();
        schedule.relation(ProofType::Execution, OpRelation::Op(select(node)));
        Ok(())
    }
}

/// Conditional selection `out = mask ? a : b` as a relation:
///
/// ```text
/// input claim   out(r₀)                (the reduced output opening)
/// output claim  eq(r₀, r′) · (mask·a + (1 − mask)·b)
/// ```
///
/// **S ⊆ R.** `Iff`'s semantics require a Boolean mask and compute `m·a + (1 − m)·b` exactly
/// (`atlas_trace::op::Iff`), so the output equals the summand at every hypercube point and its
/// MLE at `r₀` is the sum.
pub fn select<F: JoltField>(node: &ComputationNode) -> OpRel<F> {
    let ids = OpeningIdBuilder::new(node);
    let operands = [0, 1, 2].map(|position| ids.nodeio(Target::Input(position)));
    let [mask, a, b] = operands.map(|id| opening(RelationOpening::Id(id)));
    OpRel {
        rounds: node.pow2_padded_num_output_elements().log_2(),
        degree: 3,
        input: opening(RelationOpening::ReducedNodeOutput(node.idx)),
        output: derived(RelationDerived::Eq) * (mask.clone() * a + (constant(Int(1)) - mask) * b),
        produced: operands.to_vec(),
        ..OpRel::base(OpRelKind::Iff, node)
    }
}
