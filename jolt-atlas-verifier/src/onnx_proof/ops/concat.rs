//! Concat as a zero-round tie.
//!
//! When every operand's extent on the axis is a power of two and each operand starts at a
//! multiple of its own extent, the operands tile the output's axis with subcubes: operand `i`
//! is the block whose high axis bits are the bits of `offset_i / extent_i`. The output's
//! extension is then the sum, over operands, of the block's indicator at the high axis
//! coordinates times the operand's extension at the low ones. `atlas_trace`'s `pad_graph`
//! accepts only such concats; any other is rejected here.

use crate::node_view::{ComputationNode, ModelView};
use crate::onnx_proof::plan::{OpPlan, OpTie, Plan, Step};
use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};
use crate::utils::dims::{axis_groups, bits_msb_first, pow2_log};
use crate::{
    atlas_relation_types,
    onnx_proof::relation::{
        not_derived, reduced_output_setup, AtlasRelation, RelationDerived, RelationExpr,
        RelationId, RelationOpening, RelationSetup, TieRelation,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::field::Ring;
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_claims::{derived, opening, SymbolicSumcheck};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_trace::{op::Concat, Model};
use atlas_transcript::Transcript;

crate::impl_plan_op!(Concat, SubcubeTilingRelation);

impl OpPlan for Concat {
    fn plan(
        &self,
        node: &ComputationNode,
        model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(Plan(vec![
            Step::Reduce,
            Step::Tie {
                relation: OpTie::Concat(SubcubeTilingRelation::for_node(
                    model,
                    node.clone(),
                    self,
                )?),
            },
        ]))
    }
}

/// An operand of an aligned concat: the block of the output's axis it fills.
struct Block {
    /// `offset / extent`: the block's index among extent-sized blocks of the axis.
    index: usize,
    /// `log₂(output extent / extent)`: the axis coordinates the block index fixes.
    index_vars: usize,
}

/// An aligned concat node: its axis and its operands' blocks, in operand order.
pub struct ConcatShape {
    node: ComputationNode,
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
/// **S ⊆ R.** Every dim is a power of two, the operands' extents sum to the output's, and each
/// operand starts at a multiple of its extent, so the operands' blocks partition the axis and
/// output cell `(block_i, j)` is operand `i`'s cell `j`. The right-hand side is multilinear in
/// `r` and equals the output on every Boolean point (exactly one indicator is one there), so it
/// is the output's extension.
pub struct SubcubeTilingRelation {
    shape: ConcatShape,
}

impl SubcubeTilingRelation {
    /// The relation of a concat `node` of `model`. `Err` if an operand is not a node of the
    /// model, a dim is not a power of two, an operand's dims differ from the node's off the
    /// axis, the extents on the axis do not sum to the node's, or an operand's offset is not a
    /// multiple of its extent.
    pub fn for_node(
        model: &Model,
        node: ComputationNode,
        op: &Concat,
    ) -> Result<Self, ProofVerifyError> {
        let axis = op.axis;
        let Some(&output_extent) = node.output_dims.get(axis) else {
            return Err(unsupported(&node, "the axis is out of range"));
        };
        for &dim in &node.output_dims {
            pow2_log(dim)?;
        }
        let output_vars = pow2_log(output_extent)?;
        let mut blocks = Vec::new();
        let mut offset = 0usize;
        for input_dims in operand_dims(model, &node)? {
            let mut expected = node.output_dims.clone();
            let extent = match (input_dims.get(axis), expected.get_mut(axis)) {
                (Some(&extent), Some(dim)) => {
                    *dim = extent;
                    extent
                }
                _ => return Err(unsupported(&node, "an operand of another rank")),
            };
            if input_dims != expected {
                return Err(unsupported(
                    &node,
                    format!(
                        "operand dims {input_dims:?} in an output of {:?}",
                        node.output_dims
                    ),
                ));
            }
            let extent_vars = pow2_log(extent)?;
            if extent > output_extent || !offset.is_multiple_of(extent) {
                return Err(unsupported(
                    &node,
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
                &node,
                format!("operand extents sum to {offset}, the output's is {output_extent}"),
            ));
        }
        Ok(Self::new(ConcatShape { node, axis, blocks }))
    }

    fn input(&self, operand: usize) -> OpeningId {
        OpeningIdBuilder::new(&self.shape.node).nodeio(Target::Input(operand))
    }

    /// The output point `r` split for `block`: the high coordinates of the axis its index
    /// fixes, and the operand's point (`r` with the axis's low coordinates on the axis).
    fn split<F: JoltField>(
        &self,
        block: &Block,
        r: &[F],
    ) -> Result<(Vec<F>, Vec<F>), ProofVerifyError> {
        let mut high = Vec::new();
        let mut point = Vec::with_capacity(r.len());
        for (axis, group) in axis_groups(&self.shape.node.output_dims, r)?
            .into_iter()
            .enumerate()
        {
            if axis == self.shape.axis {
                let (group_high, group_low) = group.split_at(block.index_vars.min(group.len()));
                high.extend_from_slice(group_high);
                point.extend_from_slice(group_low);
            } else {
                point.extend_from_slice(group);
            }
        }
        Ok((high, point))
    }

    /// Each operand's point for the output point `r`, in operand order.
    pub fn operand_points<F: JoltField>(&self, r: &[F]) -> Result<Vec<Vec<F>>, ProofVerifyError> {
        self.shape
            .blocks
            .iter()
            .map(|block| Ok(self.split(block, r)?.1))
            .collect()
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
    ProofVerifyError::Malformed(format!("model: concat node {}: {what}", node.idx))
}

impl SymbolicSumcheck for SubcubeTilingRelation {
    atlas_relation_types!();

    fn id(&self) -> RelationId {
        RelationId::Concat
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
        (0..self.shape.blocks.len())
            .map(|operand| {
                derived(RelationDerived::BlockEq(operand))
                    * opening(RelationOpening::Id(self.input(operand)))
            })
            .fold(RelationExpr::zero(), |sum, term| sum + term)
    }
}

impl SubcubeTilingRelation {
    /// The relation for the given shape.
    pub fn new(shape: ConcatShape) -> Self {
        Self { shape }
    }
}

impl Produces for SubcubeTilingRelation {
    fn produced(&self) -> Vec<OpeningId> {
        (0..self.shape.blocks.len())
            .map(|operand| self.input(operand))
            .collect()
    }
}

impl<F: JoltField> Relation<F> for SubcubeTilingRelation {
    fn opening_points(
        &self,
        setup: &RelationSetup<F>,
        _challenges: &[F],
    ) -> Result<Vec<Vec<F>>, ProofVerifyError> {
        self.operand_points(&setup.incoming_point)
    }

    fn derive(
        &self,
        id: &RelationDerived,
        setup: &RelationSetup<F>,
        _challenges: &[F],
    ) -> Result<F, ProofVerifyError> {
        let RelationDerived::BlockEq(operand) = id else {
            return Err(not_derived(id));
        };
        let block = self
            .shape
            .blocks
            .get(*operand)
            .ok_or_else(|| unsupported(&self.shape.node, format!("no operand {operand}")))?;
        let (high, _) = self.split(block, &setup.incoming_point)?;
        Ok(bits_msb_first::<F>(block.index, block.index_vars)
            .into_iter()
            .zip(high)
            .map(|(bit, r)| bit * r + (F::one() - bit) * (F::one() - r))
            .product())
    }
}

impl<F: JoltField> SetupRelation<F> for SubcubeTilingRelation {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> Result<RelationSetup<F>, ProofVerifyError> {
        reduced_output_setup(accumulator, &self.shape.node)
    }
}

impl AtlasRelation for SubcubeTilingRelation {
    type Spec = Concat;

    fn node(&self) -> &ComputationNode {
        &self.shape.node
    }
}

impl<F: JoltField> TieRelation<F> for SubcubeTilingRelation {}
