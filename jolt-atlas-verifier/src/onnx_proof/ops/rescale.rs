//! Rescale: a wide value taken to the model's scale, as a zero-round tie.
//!
//! A wide node's value is an exact accumulation at level `j` (`atlas_trace::Model::level`): no
//! polynomial of it is committed and no range claim bounds it, because it is a function of
//! operands that are checked. `Rescale` is where it becomes an `i32` again: the Euclidean
//! split by `2^((j−1)·S)`, whose quotient is the node's output and whose remainder is advice,
//! both range claims in the packed buckets.

use crate::node_view::ComputationNode;
use crate::onnx_proof::relation::{OpRel, OpRelKind, RelationOpening};
use crate::onnx_proof::schedule::{OpOpen, OpPlan, ScheduleBuilder};
use crate::utils::opening_access::{OpeningIdBuilder, Target};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_claims::{constant, opening, Int};
use atlas_trace::{op::Rescale, Model, NodeId};
use common::VirtualPoly;

impl OpPlan for Rescale {
    fn plan<F: JoltField>(
        &self,
        node: &ComputationNode,
        model: &Model,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        schedule.reduce();
        schedule.open(OpOpen::RescaleRemainder(node.clone()));
        schedule.tie(rescale(model, node)?);
        Ok(())
    }
}

/// The bits a `Rescale` node divides by: `(j − 1)·S` for an operand at level `j`. `Err` if the
/// node has no operand in the model or the operand is not wide.
pub fn rescale_bits(model: &Model, node: &ComputationNode) -> Result<usize, ProofVerifyError> {
    let malformed = |what: &str| {
        ProofVerifyError::Malformed(format!("model: rescale node {}: {what}", node.idx))
    };
    let &operand = node.inputs.first().ok_or_else(|| malformed("no operand"))?;
    if operand >= model.graph().nodes.len() {
        return Err(malformed("its operand is not a node"));
    }
    let level = usize::from(model.level(NodeId(operand as u32)));
    if level < 2 {
        return Err(malformed("its operand is not wide"));
    }
    Ok((level - 1) * usize::from(node.scale))
}

/// A rescale as a tie at the node's output point `r`:
///
/// ```text
/// input claim   out(r)·2^B + rem(r)    (the output, the remainder advice)
/// output claim  wide(r)                (the operand, opened at r)
/// ```
///
/// with `B` the node's [`rescale_bits`]. `Err` if the node's operand is not a wide node of
/// the model.
///
/// **S ⊆ R.** `atlas_trace` splits each entry of the wide operand as `wide = out·2^B + rem`,
/// Euclidean, and fails when the quotient leaves `i32`, so the identity holds between the
/// three extensions at every point of every execution that exists.
///
/// **No wrap.** The range claims give `out ∈ [−2^31, 2^31)` (a model output is a public `i32`
/// tensor) and `rem ∈ [0, 2^B)`, with `B ≤ 3·15`, so `|out·2^B + rem| < 2^77`. The operand is
/// a wide value, whose magnitude bound is below `2^125` (`atlas_trace::op::WIDE_LIMIT`,
/// checked when the model compiles and again by `Schedule::compile`). Both sides are integers
/// of magnitude below half the 128-bit field modulus, so their equality in the field is
/// their equality over the integers, and the quotient and remainder are the Euclidean ones.
pub fn rescale<F: JoltField>(
    model: &Model,
    node: &ComputationNode,
) -> Result<OpRel<F>, ProofVerifyError> {
    let bits = rescale_bits(model, node)?;
    let ids = OpeningIdBuilder::new(node);
    let operand = ids.nodeio(Target::Input(0));
    Ok(OpRel {
        degree: 1,
        input: constant(Int(1i128 << bits)) * opening(RelationOpening::ReducedNodeOutput(node.idx))
            + opening(RelationOpening::Id(
                ids.advice(VirtualPoly::RescaleRemainder),
            )),
        output: opening(RelationOpening::Id(operand)),
        produced: vec![operand],
        ..OpRel::base(OpRelKind::Rescale, node)
    })
}
