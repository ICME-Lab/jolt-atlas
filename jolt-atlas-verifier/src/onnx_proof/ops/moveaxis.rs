use crate::node_view::ComputationNode;
use crate::onnx_proof::schedule::{OpPlan, ScheduleBuilder};
use crate::{
    onnx_proof::relation::{Custom, OpRel, OpRelKind, RelationOpening, RelationSetup},
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{
    field::JoltField,
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_claims::opening;
use atlas_trace::{op::MoveAxis, Op};

impl OpPlan for MoveAxis {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        schedule.reduce();
        schedule.tie(axis_permutation(node));
        Ok(())
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
pub fn axis_permutation<F: JoltField>(node: &ComputationNode) -> OpRel<F> {
    let operand = OpeningIdBuilder::new(node).nodeio(Target::Input(0));
    OpRel {
        degree: 1,
        input: opening(RelationOpening::ReducedNodeOutput(node.idx)),
        output: opening(RelationOpening::Id(operand)),
        produced: vec![operand],
        custom: Custom {
            points: Some(permuted_point),
            ..Custom::default()
        },
        ..OpRel::base(OpRelKind::AxisPermutation, node)
    }
}

/// The operand's point of an [`axis_permutation`]: the incoming point with its axis groups
/// permuted back.
fn permuted_point<F: JoltField>(
    relation: &OpRel<F>,
    setup: &RelationSetup<F>,
    _challenges: &[F],
) -> Result<Vec<Vec<F>>, ProofVerifyError> {
    Ok(vec![permute_challenge_groups(
        &relation.node.output_dims,
        &setup.incoming_point,
        &relation.node.operator,
    )?])
}

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
