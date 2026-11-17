use crate::node_view::ComputationNode;
use crate::onnx_proof::plan::{OpPlan, Plan, Step};
use crate::{
    onnx_proof::{
        ops::OpLayout,
        relation::{OpRel, OpRelKind, RelationOpening},
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_claims::opening;
use atlas_trace::op::IsNan;

impl OpLayout for IsNan {}

impl OpPlan for IsNan {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
    ) -> Result<Plan<F>, ProofVerifyError> {
        Ok(Plan(vec![
            Step::Reduce,
            Step::Tie {
                relation: zero_output(node),
            },
        ]))
    }
}

/// `IsNan` as a tie at the output point `r`: its output is zero everywhere.
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  0
/// ```
///
/// The operand is still opened at `r` and handed to its producer's reduction, though the
/// output expression does not read it.
///
/// **S ⊆ R.** Fixed-point data is never NaN, so `atlas_trace::op::IsNan` outputs zeros.
pub fn zero_output<F: JoltField>(node: &ComputationNode) -> OpRel<F> {
    OpRel {
        degree: 1,
        input: opening(RelationOpening::ReducedNodeOutput(node.idx)),
        produced: vec![OpeningIdBuilder::new(node).nodeio(Target::Input(0))],
        ..OpRel::base(OpRelKind::ZeroOutput, node)
    }
}
