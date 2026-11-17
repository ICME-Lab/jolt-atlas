//! Concat as a zero-round tie.
//!
//! When every operand's extent on the axis is a power of two and each operand starts at a
//! multiple of its own extent, the operands tile the output's axis with subcubes: operand `i`
//! is the block whose high axis bits are the bits of `offset_i / extent_i`. The output's
//! extension is then the sum, over operands, of the block's indicator at the high axis
//! coordinates times the operand's extension at the low ones. `atlas_trace`'s `pad_graph`
//! accepts only such concats; any other is rejected here.

use crate::node_view::{ComputationNode, ModelView};
use crate::onnx_proof::schedule::{OpPlan, ScheduleBuilder};
use crate::utils::dims::{axis_groups, bits_msb_first, pow2_log};
use crate::{
    onnx_proof::relation::{
        not_derived, Custom, OpExpr, OpRel, OpRelKind, RelationDerived, RelationOpening,
        RelationSetup,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_claims::{derived, opening};
use atlas_trace::{op::Concat, Model, Op};

impl OpPlan for Concat {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        schedule.reduce();
        schedule.tie(subcube_tiling(model, node)?);
        Ok(())
    }
}

/// An operand of an aligned concat: the block of the output's axis it fills.
struct Block {
    /// `offset / extent`: the block's index among extent-sized blocks of the axis.
    index: usize,
    /// `log₂(output extent / extent)`: the axis coordinates the block index fixes.
    index_vars: usize,
}

/// An aligned concat: its axis and its operands' blocks, in operand order.
struct ConcatShape {
    axis: usize,
    blocks: Vec<Block>,
}

/// `Concat` as a tie: the output at `r` is the sum over operands of the operand's block
/// indicator at the high coordinates of the axis times the operand at the remaining ones.
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  Σ_i eq(bits(offset_i / extent_i), r_axis,high) · operand_i(r with r_axis,low on the axis)
/// ```
///
/// The relation's [`dims`](OpRel::dims) are the operands', in operand order. `Err` if the node
/// is not a concat the tie holds for ([`check_dims`]).
///
/// **S ⊆ R.** Every dim is a power of two, the operands' extents sum to the output's, and each
/// operand starts at a multiple of its extent, so the operands' blocks partition the axis and
/// output cell `(block_i, j)` is operand `i`'s cell `j`. The right-hand side is multilinear in
/// `r` and equals the output on every Boolean point (exactly one indicator is one there), so it
/// is the output's extension.
pub fn subcube_tiling<F: JoltField>(
    model: &Model,
    node: &ComputationNode,
) -> Result<OpRel<F>, ProofVerifyError> {
    let operand_dims = operand_dims(model, node)?;
    let operands = concat_shape(node, &operand_dims)?.blocks.len();
    let ids = OpeningIdBuilder::new(node);
    let produced: Vec<_> = (0..operands)
        .map(|operand| ids.nodeio(Target::Input(operand)))
        .collect();
    Ok(OpRel {
        degree: 1,
        input: opening(RelationOpening::ReducedNodeOutput(node.idx)),
        output: produced
            .iter()
            .enumerate()
            .map(|(operand, id)| {
                derived(RelationDerived::BlockEq(operand)) * opening(RelationOpening::Id(*id))
            })
            .fold(OpExpr::zero(), |sum, term| sum + term),
        produced,
        dims: operand_dims,
        custom: Custom {
            points: Some(block_points),
            derive: Some(block_indicator),
            ..Custom::default()
        },
        ..OpRel::base(OpRelKind::Concat, node)
    })
}

/// Check that a concat `node` of `model` is one its tie holds for. `Err` if an operand is not
/// a node of the model, a dim is not a power of two, an operand's dims differ from the node's
/// off the axis, the extents on the axis do not sum to the node's, or an operand's offset is
/// not a multiple of its extent.
pub fn check_dims(model: &Model, node: &ComputationNode) -> Result<(), ProofVerifyError> {
    concat_shape(node, &operand_dims(model, node)?).map(|_| ())
}

/// The blocks the operands of a concat `node`, of dims `operand_dims`, fill of its axis.
fn concat_shape(
    node: &ComputationNode,
    operand_dims: &[Vec<usize>],
) -> Result<ConcatShape, ProofVerifyError> {
    let Op::Concat(op) = &node.operator else {
        return Err(unsupported(node, "not a concat"));
    };
    let axis = op.axis;
    let Some(&output_extent) = node.output_dims.get(axis) else {
        return Err(unsupported(node, "the axis is out of range"));
    };
    for &dim in &node.output_dims {
        pow2_log(dim)?;
    }
    let output_vars = pow2_log(output_extent)?;
    let mut blocks = Vec::new();
    let mut offset = 0usize;
    for input_dims in operand_dims {
        let mut expected = node.output_dims.clone();
        let extent = match (input_dims.get(axis), expected.get_mut(axis)) {
            (Some(&extent), Some(dim)) => {
                *dim = extent;
                extent
            }
            _ => return Err(unsupported(node, "an operand of another rank")),
        };
        if *input_dims != expected {
            return Err(unsupported(
                node,
                format!(
                    "operand dims {input_dims:?} in an output of {:?}",
                    node.output_dims
                ),
            ));
        }
        let extent_vars = pow2_log(extent)?;
        if extent > output_extent || !offset.is_multiple_of(extent) {
            return Err(unsupported(
                node,
                format!("offset {offset} is not a multiple of the operand extent {extent}"),
            ));
        }
        blocks.push(Block {
            index: offset / extent,
            index_vars: output_vars - extent_vars,
        });
        offset += extent;
    }
    if offset != output_extent {
        return Err(unsupported(
            node,
            format!("operand extents sum to {offset}, the output's is {output_extent}"),
        ));
    }
    Ok(ConcatShape { axis, blocks })
}

/// The output point `r` of `node` split for `block` on `axis`: the high coordinates of the
/// axis its index fixes, and the operand's point (`r` with the axis's low coordinates on the
/// axis).
fn split<F: JoltField>(
    node: &ComputationNode,
    axis: usize,
    block: &Block,
    r: &[F],
) -> Result<(Vec<F>, Vec<F>), ProofVerifyError> {
    let mut high = Vec::new();
    let mut point = Vec::with_capacity(r.len());
    for (group_axis, group) in axis_groups(&node.output_dims, r)?.into_iter().enumerate() {
        if group_axis == axis {
            let (group_high, group_low) = group.split_at(block.index_vars.min(group.len()));
            high.extend_from_slice(group_high);
            point.extend_from_slice(group_low);
        } else {
            point.extend_from_slice(group);
        }
    }
    Ok((high, point))
}

/// Each operand's point of a [`subcube_tiling`] for the output point `r`, in operand order.
pub fn operand_points<F: JoltField>(
    relation: &OpRel<F>,
    r: &[F],
) -> Result<Vec<Vec<F>>, ProofVerifyError> {
    let node = &relation.node;
    let shape = concat_shape(node, &relation.dims)?;
    shape
        .blocks
        .iter()
        .map(|block| Ok(split(node, shape.axis, block, r)?.1))
        .collect()
}

/// The opening points of a [`subcube_tiling`]: the operands' points for the incoming point.
fn block_points<F: JoltField>(
    relation: &OpRel<F>,
    setup: &RelationSetup<F>,
    _challenges: &[F],
) -> Result<Vec<Vec<F>>, ProofVerifyError> {
    operand_points(relation, &setup.incoming_point)
}

/// [`RelationDerived::BlockEq`] of a [`subcube_tiling`]: the indicator of the operand's block
/// at the high coordinates of the axis in the incoming point.
fn block_indicator<F: JoltField>(
    relation: &OpRel<F>,
    id: &RelationDerived,
    setup: &RelationSetup<F>,
    _challenges: &[F],
) -> Result<F, ProofVerifyError> {
    let RelationDerived::BlockEq(operand) = id else {
        return Err(not_derived(id));
    };
    let node = &relation.node;
    let shape = concat_shape(node, &relation.dims)?;
    let block = shape
        .blocks
        .get(*operand)
        .ok_or_else(|| unsupported(node, format!("no operand {operand}")))?;
    let (high, _) = split(node, shape.axis, block, &setup.incoming_point)?;
    Ok(bits_msb_first::<F>(block.index, block.index_vars)
        .into_iter()
        .zip(high)
        .map(|(bit, r)| bit * r + (F::one() - bit) * (F::one() - r))
        .product())
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
    ProofVerifyError::Malformed(format!("model: concat node {}: {what}", node.idx))
}
