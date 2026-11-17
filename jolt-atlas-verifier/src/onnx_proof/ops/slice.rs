//! Slice as a zero-round tie.
//!
//! A slice whose extent is a power of two and whose start is a multiple of it selects a subcube
//! of the operand: the cells whose high axis bits are the bits of `start / extent`. The tie
//! re-points the node's reduced output claim at the operand, at the output point with those
//! bits prefixed to the sliced axis's coordinates. `atlas_trace`'s `pad_graph` accepts only
//! such slices; any other is rejected here.

use crate::node_view::{ComputationNode, ModelView};
use crate::onnx_proof::schedule::{OpPlan, ScheduleBuilder};
use crate::utils::dims::{axis_groups, bits_msb_first, pow2_log};
use crate::{
    onnx_proof::relation::{Custom, OpRel, OpRelKind, RelationOpening, RelationSetup},
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_claims::opening;
use atlas_trace::{op::Slice, Model, Op};

impl OpPlan for Slice {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        schedule.reduce();
        schedule.tie(subcube_select(model, node)?);
        Ok(())
    }
}

/// An aligned slice: its axis, and the subcube of the operand it selects.
struct SliceShape {
    axis: usize,
    /// `start / extent`: which extent-sized block of the operand's axis the slice is.
    block: usize,
    /// `log₂(operand extent / extent)`: the axis coordinates the block index fixes.
    block_vars: usize,
}

/// `Slice` as a tie: the output at `r` is the operand at `r` with the bits of the slice's block
/// index prefixed to the sliced axis's coordinates.
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  operand(r with bits(start / extent) ‖ r_axis on the axis)
/// ```
///
/// The relation's [`dims`](OpRel::dims) are the operand's. `Err` if the node is not a slice
/// the tie holds for ([`check_dims`]).
///
/// **S ⊆ R.** Every dim is a power of two and `start` is a multiple of the extent, so output
/// cell `j` on the axis is operand cell `start + j`, whose axis index is the block index's bits
/// followed by `j`'s. Fixing those high bits of the operand's extension to the block's leaves
/// the extension of that subcube, which is the output's.
pub fn subcube_select<F: JoltField>(
    model: &Model,
    node: &ComputationNode,
) -> Result<OpRel<F>, ProofVerifyError> {
    let input_dims = input_dims(model, node)?;
    slice_shape(node, &input_dims)?;
    let operand = OpeningIdBuilder::new(node).nodeio(Target::Input(0));
    Ok(OpRel {
        degree: 1,
        input: opening(RelationOpening::ReducedNodeOutput(node.idx)),
        output: opening(RelationOpening::Id(operand)),
        produced: vec![operand],
        dims: vec![input_dims],
        custom: Custom {
            points: Some(subcube_point),
            ..Custom::default()
        },
        ..OpRel::base(OpRelKind::Slice, node)
    })
}

/// Check that a slice `node` of `model` is one its tie holds for. `Err` if the node's operand
/// is not a node of the model, a dim is not a power of two, the node's dims are not the
/// operand's with the slice's extent on its axis, or the slice's start is not a multiple of its
/// extent.
pub fn check_dims(model: &Model, node: &ComputationNode) -> Result<(), ProofVerifyError> {
    slice_shape(node, &input_dims(model, node)?).map(|_| ())
}

/// The dims of a slice `node`'s one operand.
fn input_dims(model: &Model, node: &ComputationNode) -> Result<Vec<usize>, ProofVerifyError> {
    let [input_dims] = operand_dims(model, node)?
        .try_into()
        .map_err(|_| unsupported(node, "a slice has one operand"))?;
    Ok(input_dims)
}

/// The subcube a slice `node` selects of an operand of `input_dims`.
fn slice_shape(
    node: &ComputationNode,
    input_dims: &[usize],
) -> Result<SliceShape, ProofVerifyError> {
    let Op::Slice(op) = &node.operator else {
        return Err(unsupported(node, "not a slice"));
    };
    let (axis, start, end) = (op.axis, op.start, op.end);
    let Some(&input_extent) = input_dims.get(axis) else {
        return Err(unsupported(node, "the axis is out of range"));
    };
    if start >= end || end > input_extent {
        return Err(unsupported(
            node,
            format!("{start}..{end} of {input_extent}"),
        ));
    }
    let extent = end - start;
    let mut expected = input_dims.to_vec();
    if let Some(dim) = expected.get_mut(axis) {
        *dim = extent;
    }
    if node.output_dims != expected {
        return Err(unsupported(
            node,
            format!("dims {:?}, the slice's are {expected:?}", node.output_dims),
        ));
    }
    for &dim in input_dims.iter().chain(&node.output_dims) {
        pow2_log(dim)?;
    }
    if !start.is_multiple_of(extent) {
        return Err(unsupported(
            node,
            format!("start {start} is not a multiple of the extent {extent}"),
        ));
    }
    Ok(SliceShape {
        block: start / extent,
        block_vars: pow2_log(input_extent)? - pow2_log(extent)?,
        axis,
    })
}

/// The operand's point of a [`subcube_select`] for the output point `r`.
pub fn operand_point<F: JoltField>(
    relation: &OpRel<F>,
    r: &[F],
) -> Result<Vec<F>, ProofVerifyError> {
    let node = &relation.node;
    let shape = slice_shape(node, relation.dims_at(0)?)?;
    let mut point = Vec::with_capacity(r.len() + shape.block_vars);
    for (axis, group) in axis_groups(&node.output_dims, r)?.into_iter().enumerate() {
        if axis == shape.axis {
            point.extend(bits_msb_first::<F>(shape.block, shape.block_vars));
        }
        point.extend_from_slice(group);
    }
    Ok(point)
}

/// The opening point of a [`subcube_select`]: the operand's point for the incoming point.
fn subcube_point<F: JoltField>(
    relation: &OpRel<F>,
    setup: &RelationSetup<F>,
    _challenges: &[F],
) -> Result<Vec<Vec<F>>, ProofVerifyError> {
    Ok(vec![operand_point(relation, &setup.incoming_point)?])
}

/// The dims of `node`'s operands. `Err` if an operand is not a node of `model`.
fn operand_dims(
    model: &Model,
    node: &ComputationNode,
) -> Result<Vec<Vec<usize>>, ProofVerifyError> {
    node.inputs
        .iter()
        .map(|&input| {
            if input >= model.num_nodes() {
                return Err(unsupported(node, "an operand is not a node"));
            }
            Ok(model.view(input).output_dims)
        })
        .collect()
}

fn unsupported(node: &ComputationNode, what: impl std::fmt::Display) -> ProofVerifyError {
    ProofVerifyError::Malformed(format!("model: slice node {}: {what}", node.idx))
}
