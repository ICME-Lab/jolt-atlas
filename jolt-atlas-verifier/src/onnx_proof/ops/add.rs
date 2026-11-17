use crate::node_view::ComputationNode;
use crate::{
    onnx_proof::{
        relation::{OpRel, OpRelKind, RelationOpening},
        schedule::{OpPlan, ScheduleBuilder},
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_claims::opening;
use atlas_trace::{op::Add, Model};

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
        _model: &Model,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        accumulation_plan(node, OpRelKind::AddAccumulation, schedule);
        Ok(())
    }
}

/// The plan of `left ± right`: the evaluation reduction, then the tie of the output to the
/// operands. The output is range-checked in the packed buckets after the nodes' steps.
pub fn accumulation_plan<F: JoltField>(
    node: &ComputationNode,
    kind: OpRelKind,
    schedule: &mut ScheduleBuilder<F>,
) {
    schedule.reduce();
    schedule.tie(accumulation(node, kind));
}

/// `out = left ± right` as a tie at the output point `r`: no sumcheck, the identity holds at
/// `r` because it holds on the hypercube and both sides are multilinear.
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  left(r) ± right(r)
/// ```
///
/// **S ⊆ R.** The op computes `left ± right` exactly and its execution fails when a result
/// leaves `i32`, so the identity holds at every element of every execution that exists.
///
/// **No wrap.** The operands are `i32` cells, so `|left ± right| ≤ 2^32`, and `out` lies in
/// `[−2^31, 2^31)` by its range claim (a model output by being a public `i32` tensor). Both
/// sides are integers of magnitude below `2^33`, far below half the 128-bit field modulus,
/// so their equality in the field is their equality over the integers.
pub fn accumulation<F: JoltField>(node: &ComputationNode, kind: OpRelKind) -> OpRel<F> {
    let ids = OpeningIdBuilder::new(node);
    let [left, right] = [0, 1].map(|position| ids.nodeio(Target::Input(position)));
    let sign = if kind == OpRelKind::SubAccumulation {
        -1
    } else {
        1
    };
    OpRel {
        degree: 1,
        input: opening(RelationOpening::ReducedNodeOutput(node.idx)),
        output: opening(RelationOpening::Id(left)) + opening(RelationOpening::Id(right)) * sign,
        produced: vec![left, right],
        ..OpRel::base(kind, node)
    }
}
