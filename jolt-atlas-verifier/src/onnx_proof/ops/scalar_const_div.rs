use crate::node_view::ComputationNode;
use crate::onnx_proof::schedule::{OpPlan, OpRelation, ScheduleBuilder};
use crate::{
    onnx_proof::{
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
use atlas_trace::{op::ScalarConstDiv, Op};
use common::VirtualPoly;

impl OpPlan for ScalarConstDiv {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        // The remainder's bound `0 ≤ rem < divisor` is a range claim in the packed remainder
        // buckets, which have no proof for a bound that is not positive.
        if self.divisor < 1 {
            return Err(ProofVerifyError::Malformed(format!(
                "model: ScalarConstDiv node {} has the non-positive divisor {}",
                node.idx, self.divisor
            )));
        }
        schedule.reduce();
        schedule.relation(
            ProofType::Execution,
            OpRelation::Op(constant_division(node)?),
        );
        Ok(())
    }
}

/// Division by a constant `d`, `q = ⌊left / d⌋`, as a relation:
///
/// ```text
/// input claim   d · q(r₀)              (the reduced output opening, scaled)
/// output claim  eq(r₀, r′) · (left − rem)
/// ```
///
/// `Err` if the node is not a `ScalarConstDiv`.
///
/// **S ⊆ R.** `atlas_trace::semantics::fixed::floor_div` and `rem_floor` satisfy
/// `left = d·q + rem` at every element, with `0 ≤ rem < d` for the positive divisor the plan
/// requires. The relation proves the identity; the bound, which pins `q` to the floor, is the
/// remainder's range claim in the packed range buckets (`range_bucket::node_remainders`).
pub fn constant_division<F: JoltField>(
    node: &ComputationNode,
) -> Result<OpRel<F>, ProofVerifyError> {
    let Op::ScalarConstDiv(op) = &node.operator else {
        return Err(ProofVerifyError::Malformed(format!(
            "model: node {} is {}, not a ScalarConstDiv",
            node.idx, node.operator
        )));
    };
    let ids = OpeningIdBuilder::new(node);
    let produced = [
        ids.nodeio(Target::Input(0)),
        ids.advice(VirtualPoly::ScalarConstDivRemainder),
    ];
    let [left, remainder] = produced.map(|id| opening(RelationOpening::Id(id)));
    Ok(OpRel {
        rounds: node.pow2_padded_num_output_elements().log_2(),
        degree: 2,
        input: constant(Int(i128::from(op.divisor)))
            * opening(RelationOpening::ReducedNodeOutput(node.idx)),
        output: derived(RelationDerived::Eq) * (left - remainder),
        produced: produced.to_vec(),
        ..OpRel::base(OpRelKind::ConstantDivision, node)
    })
}
