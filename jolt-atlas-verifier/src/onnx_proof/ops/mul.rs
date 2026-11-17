use crate::node_view::ComputationNode;
use crate::{
    onnx_proof::{
        fused_rebase,
        relation::{OpRel, OpRelKind, RelationDerived, RelationOpening},
        schedule::{fused_rescale_plan, OpPlan, OpRelation, ScheduleBuilder},
        ProofType,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{
    field::JoltField,
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_claims::{derived, opening};
use atlas_trace::{op::Mul, trace::ModelExecutionIO, Model};

impl OpPlan for Mul {
    fn plan<F: JoltField>(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        fused_rescale_plan(
            node,
            ProofType::RescaleArith,
            OpRelation::Op(product(node, OpRelKind::Mul, &[0, 1])),
            schedule,
        );
        Ok(())
    }
}

/// An element-wise product of operands (`Mul`, `Square`, `Cube`, and `And` on booleans) as a
/// relation, `Π` running over `factors`, the operand position of each factor (`[0, 1]` for
/// `L·R`, `[0, 0]` for `X²`); each operand read is produced once, in first-factor order:
///
/// ```text
/// input claim   out(r₀)·2^S + rem(r₀)  (fused rescale: the output is the quotient)
///               out(r₀)                (raw product: the reduced output opening)
/// output claim  eq(r₀, r′) · Π operand(r′)
/// ```
///
/// The incoming point is the node output's reduced opening point. The range claims of a fused
/// node's remainder and output are a separate protocol ([`fused_rebase`]) after this one.
///
/// **S ⊆ R.** A raw product's output is `Π operand` exactly, so `Σ eq(r₀,·)·Π operand` is the
/// output's MLE at `r₀`. A fused product splits the accumulation `Π operand` with
/// `atlas_trace::semantics::fixed::rebase` into `out·2^S + rem` (Euclidean, `S` the op's
/// rebase bits: twice the scale for `Cube`), and its execution fails when the quotient leaves
/// `i32`, so the same sum equals `out(r₀)·2^S + rem(r₀)` on every execution that exists.
///
/// **No wrap (fused).** The operands are `i32` cells, so `|Π operand| ≤ 2^62` for `Mul` and
/// `Square` and `≤ 2^93` for `Cube`. The range claims give `out ∈ [−2^31, 2^31)` (a model
/// output is a public `i32` tensor) and `rem ∈ [0, 2^S)`, so `|out·2^S + rem| < 2^(32+S)`,
/// at most `2^62` since `S ≤ 30` (the scale is at most 15, `atlas_trace::Scale`). Both sides
/// are integers of magnitude below `2^94`, far below half the 128-bit field modulus, so
/// their equality in the field is their equality over the integers.
pub fn product<F: JoltField>(
    node: &ComputationNode,
    kind: OpRelKind,
    factors: &[usize],
) -> OpRel<F> {
    let ids = OpeningIdBuilder::new(node);
    let mut operands: Vec<usize> = Vec::new();
    for &position in factors {
        if !operands.contains(&position) {
            operands.push(position);
        }
    }
    OpRel {
        rounds: node.pow2_padded_num_output_elements().log_2(),
        degree: factors.len() + 1,
        input: fused_rebase::input_expression(node),
        output: factors
            .iter()
            .fold(derived(RelationDerived::Eq), |product, &position| {
                product * opening(RelationOpening::Id(ids.nodeio(Target::Input(position))))
            }),
        produced: operands
            .into_iter()
            .map(|position| ids.nodeio(Target::Input(position)))
            .collect(),
        ..OpRel::base(kind, node)
    }
}
