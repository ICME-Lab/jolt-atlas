//! Slice as a zero-round tie.
//!
//! A slice whose extent is a power of two and whose start is a multiple of it selects a subcube
//! of the operand: the cells whose high axis bits are the bits of `start / extent`. The tie
//! re-points the node's reduced output claim at the operand, at the output point with those
//! bits prefixed to the sliced axis's coordinates. `atlas_trace`'s `pad_graph` accepts only
//! such slices; any other is rejected here.

use crate::node_view::{ComputationNode, ModelView};
use crate::onnx_proof::plan::{OpPlan, OpTie, Plan, Step};
use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};
use crate::utils::dims::{axis_groups, bits_msb_first, pow2_log};
use crate::{
    atlas_relation_types,
    onnx_proof::relation::{
        reduced_output_setup, AtlasRelation, RelationExpr, RelationId, RelationOpening,
        RelationSetup, TieRelation,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::field::Ring;
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_claims::{opening, SymbolicSumcheck};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_trace::{op::Slice, Model};
use atlas_transcript::Transcript;

crate::impl_plan_op!(Slice, SubcubeSelectRelation);

impl OpPlan for Slice {
    fn plan(
        &self,
        node: &ComputationNode,
        model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(Plan(vec![
            Step::Reduce,
            Step::Tie {
                relation: OpTie::Slice(SubcubeSelectRelation::for_node(model, node.clone(), self)?),
            },
        ]))
    }
}

/// An aligned slice node: its axis, and the subcube of the operand it selects.
pub struct SliceShape {
    node: ComputationNode,
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
/// **S ⊆ R.** Every dim is a power of two and `start` is a multiple of the extent, so output
/// cell `j` on the axis is operand cell `start + j`, whose axis index is the block index's bits
/// followed by `j`'s. Fixing those high bits of the operand's extension to the block's leaves
/// the extension of that subcube, which is the output's.
pub struct SubcubeSelectRelation {
    shape: SliceShape,
}

impl SubcubeSelectRelation {
    /// The relation of a slice `node` of `model`. `Err` if the node's operand is not a node of
    /// the model, a dim is not a power of two, the node's dims are not the operand's with the
    /// slice's extent on its axis, or the slice's start is not a multiple of its extent.
    pub fn for_node(
        model: &Model,
        node: ComputationNode,
        op: &Slice,
    ) -> Result<Self, ProofVerifyError> {
        let [input_dims] = operand_dims(model, &node)?
            .try_into()
            .map_err(|_| unsupported(&node, "a slice has one operand"))?;
        let (axis, start, end) = (op.axis, op.start, op.end);
        let Some(&input_extent) = input_dims.get(axis) else {
            return Err(unsupported(&node, "the axis is out of range"));
        };
        if start >= end || end > input_extent {
            return Err(unsupported(
                &node,
                format!("{start}..{end} of {input_extent}"),
            ));
        }
        let extent = end - start;
        let mut expected = input_dims.clone();
        if let Some(dim) = expected.get_mut(axis) {
            *dim = extent;
        }
        if node.output_dims != expected {
            return Err(unsupported(
                &node,
                format!("dims {:?}, the slice's are {expected:?}", node.output_dims),
            ));
        }
        for &dim in input_dims.iter().chain(&node.output_dims) {
            pow2_log(dim)?;
        }
        if !start.is_multiple_of(extent) {
            return Err(unsupported(
                &node,
                format!("start {start} is not a multiple of the extent {extent}"),
            ));
        }
        Ok(Self::new(SliceShape {
            block: start / extent,
            block_vars: pow2_log(input_extent)? - pow2_log(extent)?,
            node,
            axis,
        }))
    }

    fn input(&self) -> OpeningId {
        OpeningIdBuilder::new(&self.shape.node).nodeio(Target::Input(0))
    }

    /// The operand's point for the output point `r`.
    pub fn operand_point<F: JoltField>(&self, r: &[F]) -> Result<Vec<F>, ProofVerifyError> {
        let mut point = Vec::with_capacity(r.len() + self.shape.block_vars);
        for (axis, group) in axis_groups(&self.shape.node.output_dims, r)?
            .into_iter()
            .enumerate()
        {
            if axis == self.shape.axis {
                point.extend(bits_msb_first::<F>(self.shape.block, self.shape.block_vars));
            }
            point.extend_from_slice(group);
        }
        Ok(point)
    }
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

impl SymbolicSumcheck for SubcubeSelectRelation {
    atlas_relation_types!();
    type Shape = SliceShape;

    fn new(shape: SliceShape) -> Self {
        Self { shape }
    }

    fn id() -> RelationId {
        RelationId::Slice
    }

    fn rounds(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::ReducedNodeOutput(self.shape.node.idx))
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::Id(self.input()))
    }
}

impl Produces for SubcubeSelectRelation {
    fn produced(&self) -> Vec<OpeningId> {
        vec![self.input()]
    }
}

impl<F: JoltField> Relation<F> for SubcubeSelectRelation {
    fn opening_points(
        &self,
        setup: &RelationSetup<F>,
        _challenges: &[F],
    ) -> Result<Vec<Vec<F>>, ProofVerifyError> {
        Ok(vec![self.operand_point(&setup.incoming_point)?])
    }
}

impl<F: JoltField> SetupRelation<F> for SubcubeSelectRelation {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> Result<RelationSetup<F>, ProofVerifyError> {
        reduced_output_setup(accumulator, &self.shape.node)
    }
}

impl AtlasRelation for SubcubeSelectRelation {
    type Spec = Slice;

    fn node(&self) -> &ComputationNode {
        &self.shape.node
    }
}

impl<F: JoltField> TieRelation<F> for SubcubeSelectRelation {}
