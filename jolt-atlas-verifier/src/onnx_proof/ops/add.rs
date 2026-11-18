use crate::node_view::ComputationNode;
use crate::{
    onnx_proof::{
        relation::{OpRel, OpRelKind, RelationOpening},
        schedule::{OpPlan, ScheduleBuilder},
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_claims::{constant, opening, Int};
use atlas_trace::{op::Add, Model, NodeId};

/// Proof for exact element-wise addition: `output = left + right`, where a sum outside `i32`
/// is an execution error. The plan ([`accumulation_plan`]) has two parts:
///
/// 1. **Evaluation reduction**: the node's output claims are reduced to one opening at a
///    point `r`.
/// 2. **Operand tie** ([`accumulation`], no sumcheck): the prover opens
///    `left`/`right` at `r`, and the verifier checks `left(r) + right(r) == output(r)`.
///
/// Unless the node is a model output, its output is shown to lie in `i32` by a range claim in
/// the packed range buckets, proven after the nodes' steps
/// ([`range_bucket`](crate::onnx_proof::range_bucket)).
impl OpPlan for Add {
    fn plan<F: JoltField>(
        &self,
        node: &ComputationNode,
        model: &Model,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        accumulation_plan(model, node, OpRelKind::AddAccumulation, schedule)
    }
}

/// The plan of `left ± right`: the evaluation reduction, then the tie of the output to the
/// operands. The output is range-checked in the packed buckets after the nodes' steps.
pub fn accumulation_plan<F: JoltField>(
    model: &Model,
    node: &ComputationNode,
    kind: OpRelKind,
    schedule: &mut ScheduleBuilder<F>,
) -> Result<(), ProofVerifyError> {
    schedule.reduce();
    schedule.tie(accumulation(model, node, kind)?);
    Ok(())
}

/// The bits each operand of `node` is lifted by: `(top − level)·S` for the higher of the two
/// operands' levels `top`. Both are zero when the operands are at one level. `Err` if an
/// operand is not a node of the model.
fn operand_lifts(model: &Model, node: &ComputationNode) -> Result<[usize; 2], ProofVerifyError> {
    let level = |position: usize| -> Result<usize, ProofVerifyError> {
        node.inputs
            .get(position)
            .filter(|&&operand| operand < model.graph().nodes.len())
            .map(|&operand| usize::from(model.level(NodeId(operand as u32))))
            .ok_or_else(|| {
                ProofVerifyError::Malformed(format!(
                    "model: node {} has no operand {position}",
                    node.idx
                ))
            })
    };
    let (left, right) = (level(0)?, level(1)?);
    let top = left.max(right);
    Ok([left, right].map(|level| (top - level) * usize::from(node.scale)))
}

/// `out = left ± right` as a tie at the output point `r`: no sumcheck, the identity holds at
/// `r` because it holds on the hypercube and both sides are multilinear. An operand at a
/// lower level than the other is lifted to it ([`operand_lifts`]), and the output is then
/// wide.
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  2^a·left(r) ± 2^b·right(r)      (a = b = 0 for level-1 operands)
/// ```
///
/// `Err` if an operand is not a node of the model.
///
/// **S ⊆ R.** The op computes `left ± right` exactly, lifting the lower operand, and at
/// level 1 its execution fails when a result leaves `i32`, so the identity holds at every
/// element of every execution that exists.
///
/// **No wrap.** At level 1 the operands are `i32` cells, so `|left ± right| ≤ 2^32`, and
/// `out` lies in `[−2^31, 2^31)` by its range claim (a model output by being a public `i32`
/// tensor): both sides are integers of magnitude below `2^33`. A wide output's magnitude
/// bound, the sum of the lifted operands' bounds, is below `2^125`
/// (`atlas_trace::op::WIDE_LIMIT`, checked when the model compiles and again by
/// `Schedule::compile`), and the output is whatever the relation that consumes it makes it.
/// Either way both sides are integers of magnitude below half the 128-bit field modulus, so
/// their equality in the field is their equality over the integers.
pub fn accumulation<F: JoltField>(
    model: &Model,
    node: &ComputationNode,
    kind: OpRelKind,
) -> Result<OpRel<F>, ProofVerifyError> {
    let [left_lift, right_lift] = operand_lifts(model, node)?;
    let ids = OpeningIdBuilder::new(node);
    let [left, right] = [0, 1].map(|position| ids.nodeio(Target::Input(position)));
    let sign: i128 = if kind == OpRelKind::SubAccumulation {
        -1
    } else {
        1
    };
    Ok(OpRel {
        degree: 1,
        input: opening(RelationOpening::ReducedNodeOutput(node.idx)),
        output: constant(Int(1i128 << left_lift)) * opening(RelationOpening::Id(left))
            + constant(Int(sign * (1i128 << right_lift))) * opening(RelationOpening::Id(right)),
        produced: vec![left, right],
        ..OpRel::base(kind, node)
    })
}
