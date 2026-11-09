use crate::node_view::ComputationNode;
use crate::onnx_proof::plan::{OpPlan, OpTie, Plan, Step};
use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};
use crate::{
    atlas_relation_types,
    onnx_proof::relation::{
        reduced_output_setup, AtlasRelation, RelationExpr, RelationId, RelationOpening,
        RelationSetup, TieRelation,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{
    field::JoltField,
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_trace::{op::MoveAxis, Op};
use atlas_transcript::Transcript;
use jolt_claims::{opening, SymbolicSumcheck};
use jolt_field::Ring;

crate::impl_plan_op!(MoveAxis, AxisPermutationRelation);

impl OpPlan for MoveAxis {
    fn plan(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(Plan(vec![
            Step::Reduce,
            Step::Tie {
                relation: OpTie::AxisPermutation(AxisPermutationRelation::new(node.clone())),
            },
        ]))
    }
}

/// `MoveAxis` as a tie: the output at `r` is the operand at `r` with its per-axis coordinate
/// groups permuted back.
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  operand(π⁻¹(r))
/// ```
///
/// **S ⊆ R.** Moving an axis reorders elements without changing them, so the output's MLE is
/// the operand's with its variable groups permuted.
pub struct AxisPermutationRelation {
    node: ComputationNode,
}

impl AxisPermutationRelation {
    fn input(&self) -> OpeningId {
        OpeningIdBuilder::new(&self.node).nodeio(Target::Input(0))
    }
}

impl SymbolicSumcheck for AxisPermutationRelation {
    atlas_relation_types!();
    type Shape = ComputationNode;

    fn new(node: ComputationNode) -> Self {
        Self { node }
    }

    fn id() -> RelationId {
        RelationId::AxisPermutation
    }

    fn rounds(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::ReducedNodeOutput(self.node.idx))
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::Id(self.input()))
    }
}

impl Produces for AxisPermutationRelation {
    fn produced(&self) -> Vec<OpeningId> {
        vec![self.input()]
    }
}

impl<F: JoltField> Relation<F> for AxisPermutationRelation {
    fn opening_points(
        &self,
        setup: &RelationSetup<F>,
        _challenges: &[F],
    ) -> Result<Vec<Vec<F>>, ProofVerifyError> {
        Ok(vec![permute_challenge_groups(
            &self.node.output_dims,
            &setup.incoming_point,
            &self.node.operator,
        )?])
    }
}

impl<F: JoltField> SetupRelation<F> for AxisPermutationRelation {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> Result<RelationSetup<F>, ProofVerifyError> {
        reduced_output_setup(accumulator, &self.node)
    }
}

impl AtlasRelation for AxisPermutationRelation {
    type Spec = MoveAxis;

    fn node(&self) -> &ComputationNode {
        &self.node
    }
}

impl<F: JoltField> TieRelation<F> for AxisPermutationRelation {}

/// Permutes challenge variable groups to reverse the moveaxis transformation. `Err` if the
/// operator is not a move of two of `output_dims`' axes, or `r_output` is not a point of their
/// padded domain.
fn permute_challenge_groups<F: JoltField>(
    output_dims: &[usize],
    r_output: &[F],
    operator: &Op,
) -> Result<Vec<F>, ProofVerifyError> {
    let unsupported = |what: &str| ProofVerifyError::Malformed(format!("model: moveaxis: {what}"));
    let Op::MoveAxis(op) = operator else {
        return Err(unsupported("expected a MoveAxis operator"));
    };
    let (source, destination) = (op.source, op.destination);
    if source >= output_dims.len() || destination >= output_dims.len() {
        return Err(unsupported("an axis out of range"));
    }

    // Split r_output into groups, one for each axis in output_dims
    let mut challenge_groups: Vec<Vec<F>> = Vec::new();
    let mut offset = 0;

    for &dim in output_dims.iter() {
        let num_vars = dim.log_2();
        let group = r_output
            .get(offset..offset + num_vars)
            .ok_or_else(|| unsupported("a point shorter than the padded output domain"))?;
        challenge_groups.push(group.to_vec());
        offset += num_vars;
    }

    // We need to do the opposite, since we're going from output to input
    // Hence we take the group at 'destination' and put it at 'source'
    let dst_group = challenge_groups.remove(destination);
    challenge_groups.insert(source, dst_group);

    // Flatten the groups back into a single vector
    Ok(challenge_groups.into_iter().flatten().collect())
}
