use crate::node_view::ComputationNode;
use crate::onnx_proof::schedule::{OpPlan, ScheduleBuilder};
use crate::{
    onnx_proof::{
        ops::OpLayout,
        relation::{OpRel, OpRelKind, RelationOpening},
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_claims::{constant, opening, Int};
use atlas_trace::op::Neg;

impl OpLayout for Neg {}

impl OpPlan for Neg {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        schedule.reduce();
        schedule.tie(signed_copy(node, OpRelKind::Neg));
        Ok(())
    }
}

/// The sign of a copy kind: `−1` for `Neg`, `+1` otherwise.
pub fn sign(kind: OpRelKind) -> i64 {
    if kind == OpRelKind::Neg {
        -1
    } else {
        1
    }
}

/// `out = ±operand` as a tie at the output point `r`: no sumcheck, the identity holds at `r`
/// because it holds on the hypercube and both sides are multilinear.
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  ±operand(r)
/// ```
///
/// **S ⊆ R.** `Identity` passes its operand through. `Neg` negates element-wise and its
/// execution fails on `i32::MIN`, whose negation is outside `i32`, so the tie holds at every
/// element of every execution that exists.
pub fn signed_copy<F: JoltField>(node: &ComputationNode, kind: OpRelKind) -> OpRel<F> {
    let operand = OpeningIdBuilder::new(node).nodeio(Target::Input(0));
    OpRel {
        degree: 1,
        input: opening(RelationOpening::ReducedNodeOutput(node.idx)),
        output: constant(Int(i128::from(sign(kind)))) * opening(RelationOpening::Id(operand)),
        produced: vec![operand],
        ..OpRel::base(kind, node)
    }
}
