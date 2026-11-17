use crate::onnx_proof::schedule::{OpPlan, OpRelation, ScheduleBuilder};
use crate::{
    node_view::ComputationNode,
    onnx_proof::{
        relation::{Incoming, OpRel, OpRelKind, RelationDerived, RelationOpening},
        ProofType,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{
    field::JoltField,
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_claims::{derived, opening};
use atlas_trace::op::Div;
use common::{CommittedPoly, VirtualPoly};

impl OpPlan for Div {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        // The division identity runs before the reduction: it adds a claim on the node's own
        // output (the quotient), which the reduction then folds in.
        // The remainder's bound `0 ≤ rem < right` is proven in the packed range buckets
        // after the node loop (`range_bucket::node_remainders`).
        schedule.relation(ProofType::Execution, OpRelation::Op(division(node)));
        schedule.reduce();
        schedule.tie(division_quotient(node));
        Ok(())
    }
}

/// Floor division `q = ⌊left / right⌋` as a relation, its point `r₀` drawn from the transcript:
///
/// ```text
/// input claim   0
/// output claim  eq(r₀, r′) · (right·q + rem − left)
/// ```
///
/// **S ⊆ R.** `atlas_trace::semantics::fixed::floor_div` and its remainder `rem_floor` (the
/// divisor's sign) satisfy `left = right·q + rem` at every element, so the summand vanishes on
/// the hypercube. The bound on `rem` that makes `q` the floor is the range check below,
/// outside the relation.
pub fn division<F: JoltField>(node: &ComputationNode) -> OpRel<F> {
    let ids = OpeningIdBuilder::new(node);
    let produced = [
        ids.nodeio(Target::Input(0)),
        ids.nodeio(Target::Input(1)),
        ids.nodeio(Target::Current),
        ids.advice(VirtualPoly::DivRemainder),
    ];
    let [left, right, quotient, remainder] = produced.map(|id| opening(RelationOpening::Id(id)));
    OpRel {
        rounds: node.pow2_padded_num_output_elements().log_2(),
        degree: 3,
        output: derived(RelationDerived::Eq) * (right * quotient + remainder - left),
        produced: produced.to_vec(),
        incoming: Incoming::Fresh,
        ..OpRel::base(OpRelKind::Div, node)
    }
}

// TODO: Reduce two claims to 1 via 4.5.2 PAZK for Quotient polynomial openings

/// The committed quotient `q` (what the PCS opens) bound to the node's reduced output, a tie:
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  q(r)                   (the committed quotient, at the same point)
/// ```
///
/// **S ⊆ R.** The committed quotient is the node's output tensor.
pub fn division_quotient<F: JoltField>(node: &ComputationNode) -> OpRel<F> {
    let quotient = OpeningIdBuilder::new(node).advice(CommittedPoly::DivNodeQuotient);
    OpRel {
        degree: 1,
        input: opening(RelationOpening::ReducedNodeOutput(node.idx)),
        output: opening(RelationOpening::Id(quotient)),
        produced: vec![quotient],
        ..OpRel::base(OpRelKind::DivQuotient, node)
    }
}
