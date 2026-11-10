use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    clamp_lookups::is_scalar,
    ops::sum::axis::SumAxisRelation,
    plan::{clamp_steps, OpOpen, OpPlan, OpRelation, OpTie, Plan, Step},
    ProofType,
};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::{op::Sum, trace::ModelExecutionIO, Model};

crate::impl_plan_op!(Sum, SumAxisRelation);

/// Axis-wise sum implementations for sumcheck protocol.
pub mod axis;

/// Saturating axis-sum proof: `output = SatClamp(acc)`, `acc[k] = Σ_axis operand[k, ·]`.
///
/// Reuses the [`Add`](super::add)/[`Sub`](super::sub) clamp infrastructure: the
/// per-output-element clamp `output = SatClamp(acc)` is identical (an element-wise
/// `i64 → i32` saturation), only the accumulation differs. So the proof is:
///
/// 1. **Clamp** ([`clamp_steps`]): open `acc` ([`VirtualPoly::ClampAcc`]) at the output point
///    and prove `output = SatClamp(acc)` exactly or in the node's packed clamp bucket, as for
///    `Add`.
/// 2. **Axis reduction** ([`SumAxisRelation`], `ProofType::SumReduction`): prove
///    `acc(r) = Σ_axis operand`, reducing to the operand opening. Its input claim
///    is the opened `acc(r)`, not the node output.
///
/// Scalar (`1×1`) outputs skip the packed bucket (its cycle space would be empty): `acc` opens
/// in the clear, so the verifier checks
/// `output == SatClamp(acc)` directly — and the axis reduction still binds `acc`
/// to the operand.
impl OpPlan for Sum {
    fn plan(
        &self,
        node: &ComputationNode,
        model: &Model,
        io: &ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        let mut steps = vec![Step::Reduce];
        // (1+2) `output = SatClamp(acc)`.
        if is_scalar(node) {
            // `acc` opens in the clear: recover it and check the clamp directly.
            steps.push(Step::Open(OpOpen::ClampAcc(node.clone())));
            steps.push(Step::Tie {
                relation: OpTie::ScalarClamp(node.clone()),
            });
        } else {
            steps.extend(clamp_steps(node, model, io));
        }
        // (3) Axis reduction.
        steps.push(Step::Relation {
            proof: ProofType::SumReduction,
            relation: OpRelation::SumAxis(SumAxisRelation::for_node(model, node.clone())?),
        });
        Ok(Plan(steps))
    }
}
