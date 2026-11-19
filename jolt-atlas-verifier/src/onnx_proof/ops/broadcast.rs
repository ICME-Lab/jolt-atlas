use crate::node_view::{ComputationNode, ModelView};
use crate::onnx_proof::schedule::{OpPlan, ScheduleBuilder};
use crate::{
    onnx_proof::relation::{
        not_derived, Custom, OpRel, OpRelKind, RelationDerived, RelationOpening, RelationSetup,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{
    field::JoltField,
    poly::multilinear::EvaluateMle,
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_claims::{derived, opening};
use atlas_trace::{op::Broadcast, Model, Tensor};

impl OpPlan for Broadcast {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        model: &atlas_trace::Model,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        schedule.reduce();
        schedule.tie(broadcast(model, node)?);
        Ok(())
    }
}

/// `Broadcast` as a tie: the output at `r` is the operand at the non-broadcast coordinates of
/// `r`, times the replication tensor at the broadcast coordinates: one on the lanes of each
/// broadcast axis below its extent (the axis's length before padding), zero on its padded
/// lanes.
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  ones(r_broadcast) · operand(r_input)
/// ```
///
/// The relation's [`dims`](OpRel::dims) are the operand's. `Err` if the node's operand is not
/// a node of the model.
///
/// **S ⊆ R.** Broadcasting replicates each operand element along the broadcast axes and
/// leaves their padded lanes zero, so the output tensor is the operand's tensor product with
/// the replication tensor.
pub fn broadcast<F: JoltField>(
    model: &Model,
    node: &ComputationNode,
) -> Result<OpRel<F>, ProofVerifyError> {
    let input = node
        .inputs
        .first()
        .copied()
        .filter(|&input| input < model.num_nodes())
        .ok_or_else(|| unsupported("the operand is not a node"))?;
    let operand = OpeningIdBuilder::new(node).nodeio(Target::Input(0));
    Ok(OpRel {
        degree: 1,
        input: opening(RelationOpening::ReducedNodeOutput(node.idx)),
        output: derived(RelationDerived::BroadcastOnes) * opening(RelationOpening::Id(operand)),
        produced: vec![operand],
        dims: vec![model.view(input).output_dims],
        custom: Custom {
            points: Some(operand_point),
            derive: Some(ones_at_broadcast_coordinates),
            ..Custom::default()
        },
        ..OpRel::base(OpRelKind::Broadcast, node)
    })
}

/// `(r_input, r_broadcast)`: the incoming point of a [`broadcast`] split into the operand's
/// coordinates and the broadcast ones. `Err` if the operand's dims do not broadcast to the
/// node's, or `r_output` is not a point of the node's padded domain.
pub fn split<F: JoltField>(
    relation: &OpRel<F>,
    r_output: &[F],
) -> Result<(Vec<F>, Vec<F>), ProofVerifyError> {
    let output_dims = &relation.node.output_dims;
    let broadcast_dims = get_broadcast_dims(relation.dims_at(0)?, output_dims)?;
    split_broadcast_vars(output_dims, &broadcast_dims, r_output)
}

/// The operand's point of a [`broadcast`]: the non-broadcast coordinates of the incoming point.
fn operand_point<F: JoltField>(
    relation: &OpRel<F>,
    setup: &RelationSetup<F>,
    _challenges: &[F],
) -> Result<Vec<Vec<F>>, ProofVerifyError> {
    Ok(vec![split(relation, &setup.incoming_point)?.0])
}

/// [`RelationDerived::BroadcastOnes`] of a [`broadcast`]: the replication tensor's MLE at the
/// broadcast coordinates of the incoming point.
fn ones_at_broadcast_coordinates<F: JoltField>(
    relation: &OpRel<F>,
    id: &RelationDerived,
    setup: &RelationSetup<F>,
    _challenges: &[F],
) -> Result<F, ProofVerifyError> {
    Ok(match id {
        RelationDerived::BroadcastOnes => {
            let atlas_trace::Op::Broadcast(op) = &relation.node.operator else {
                return Err(unsupported("the node is not a broadcast"));
            };
            let ones = build_broadcast_tensor(
                relation.dims_at(0)?,
                &relation.node.output_dims,
                &op.extent,
            )?;
            ones.evaluate_mle(&split(relation, &setup.incoming_point)?.1)
        }
        other => return Err(not_derived(other)),
    })
}

fn unsupported(what: impl std::fmt::Display) -> ProofVerifyError {
    ProofVerifyError::Malformed(format!("model: broadcast: {what}"))
}

/// The replication tensor of a broadcast: over the broadcast dimensions, one at the lanes
/// below each broadcast axis's `extent` and zero past it. `Err` if the input dims do not
/// broadcast to the target dims or `extent` is not of the target's rank.
fn build_broadcast_tensor(
    input_dims: &[usize],
    target_dims: &[usize],
    extent: &[usize],
) -> Result<Tensor<i32>, ProofVerifyError> {
    let bc_dims = get_broadcast_dims(input_dims, target_dims)?;
    if extent.len() != bc_dims.len() {
        return Err(unsupported("the extent is not of the output's rank"));
    }
    // Row-major: entry `flat` is live iff every broadcast coordinate is below its extent.
    let num_elems: usize = bc_dims.iter().product();
    let mut values = vec![1i32; num_elems];
    let mut stride = num_elems;
    for (&dim, &live) in bc_dims.iter().zip(extent) {
        stride /= dim.max(1);
        if dim > 1 && live < dim {
            for (flat, value) in values.iter_mut().enumerate() {
                if flat / stride % dim >= live {
                    *value = 0;
                }
            }
        }
    }
    Tensor::from_slice(&values, &bc_dims).map_err(unsupported)
}

/// Computes the broadcast dimensions
///
/// # Returns
/// An array of dimensions, where each dimensions is either 1 if no broadcast is needed in that dimension,
/// or the target dimension otherwise.
///
fn get_broadcast_dims(
    input_dims: &[usize],
    target_dims: &[usize],
) -> Result<Vec<usize>, ProofVerifyError> {
    if input_dims.len() > target_dims.len() {
        return Err(unsupported("the operand has a higher rank than the output"));
    }

    let mut broadcast_dims = target_dims.to_vec();
    for ((broadcast_dim, &target_dim), &input_dim) in broadcast_dims
        .iter_mut()
        .zip(target_dims)
        .rev()
        .zip(input_dims.iter().rev())
    {
        if input_dim == target_dim {
            *broadcast_dim = 1;
        } else if input_dim != 1 {
            return Err(unsupported(format!(
                "input dimension {input_dim} is not broadcastable to target dimension {target_dim}"
            )));
        }
    }
    Ok(broadcast_dims)
}

/// Splits the opening point r_output into two parts:
/// - r_input: the variables corresponding to the input polynomial (non-broadcasted dimensions)
/// - r_broadcast: the variables corresponding to the broadcast polynomial (broadcasted dimensions)
fn split_broadcast_vars<F: JoltField>(
    output_dims: &[usize],
    broadcast_dims: &[usize],
    r_output: &[F],
) -> Result<(Vec<F>, Vec<F>), ProofVerifyError> {
    let mut r_input: Vec<F> = Vec::new();
    let mut r_broadcast: Vec<F> = Vec::new();
    let mut idx = 0;
    for (&output_dim, &broadcast_dim) in output_dims.iter().zip(broadcast_dims.iter()) {
        let dim_vars = output_dim.log_2();
        let vars = r_output.get(idx..idx + dim_vars).ok_or_else(|| {
            unsupported(format!(
                "a point of {} coordinates for output dims {output_dims:?}",
                r_output.len()
            ))
        })?;

        // Select which variables correspond to broadcasted dimensions
        if broadcast_dim == 1 {
            // This dimension is not broadcasted, evaluating the input polynomial on associated variables
            r_input.extend(vars);
        } else {
            // This dimension is broadcasted, we evaluate the broadcast polynomial on associated variables
            r_broadcast.extend(vars);
        }
        idx += dim_vars;
    }
    Ok((r_input, r_broadcast))
}
