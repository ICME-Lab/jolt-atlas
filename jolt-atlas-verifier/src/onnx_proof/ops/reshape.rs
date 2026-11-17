//! Reshape as a zero-round tie.
//!
//! On power-of-two dims a reshape keeps every element at its flat row-major index, so the
//! output's multilinear extension is the operand's: the tie re-points the node's reduced output
//! claim at the operand, at the same point. A model compiled with padding has such dims
//! (`atlas_trace`'s `pad_graph` rejects a reshape whose layout changes under padding); a model
//! with any other dims on a reshape is rejected here.

use crate::node_view::{ComputationNode, ModelView};
use crate::onnx_proof::plan::{OpPlan, OpTie, Plan, Step};
use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};
use crate::utils::dims::pow2_log;
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
use atlas_trace::{op::Reshape, Model};
use atlas_transcript::Transcript;

crate::impl_plan_op!(Reshape, FlatCopyRelation);

impl OpPlan for Reshape {
    fn plan(
        &self,
        node: &ComputationNode,
        model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(Plan(vec![
            Step::Reduce,
            Step::Tie {
                relation: OpTie::Reshape(FlatCopyRelation::for_node(model, node.clone())?),
            },
        ]))
    }
}

/// A reshape node whose dims the tie holds for.
pub struct ReshapeShape {
    node: ComputationNode,
}

/// `Reshape` as a tie: the output at `r` is the operand at `r`.
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  operand(r)
/// ```
///
/// **S ⊆ R.** Every dim of the operand and of the output is a power of two with equal
/// products, so neither tensor is padded and both are the same flat row-major vector: a reshape
/// moves no element off its flat index. Their multilinear extensions over that index are equal.
pub struct FlatCopyRelation {
    shape: ReshapeShape,
}

impl FlatCopyRelation {
    /// The relation of a reshape `node` of `model`. `Err` if the node's operand is not a node
    /// of the model, a dim of either is not a power of two, or their element counts differ.
    pub fn for_node(model: &Model, node: ComputationNode) -> Result<Self, ProofVerifyError> {
        let [input_dims] = operand_dims(model, &node)?
            .try_into()
            .map_err(|_| unsupported(&node, "a reshape has one operand"))?;
        let num_vars = |dims: &[usize]| -> Result<usize, ProofVerifyError> {
            dims.iter().map(|&dim| pow2_log(dim)).sum()
        };
        if num_vars(&input_dims)? != num_vars(&node.output_dims)? {
            return Err(unsupported(
                &node,
                format!(
                    "{input_dims:?} and {:?} differ in element count",
                    node.output_dims
                ),
            ));
        }
        Ok(Self::new(ReshapeShape { node }))
    }

    fn input(&self) -> OpeningId {
        OpeningIdBuilder::new(&self.shape.node).nodeio(Target::Input(0))
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
    ProofVerifyError::Malformed(format!("model: reshape node {}: {what}", node.idx))
}

impl SymbolicSumcheck for FlatCopyRelation {
    atlas_relation_types!();

    fn id(&self) -> RelationId {
        RelationId::Reshape
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

impl FlatCopyRelation {
    /// The relation for the given shape.
    pub fn new(shape: ReshapeShape) -> Self {
        Self { shape }
    }
}

impl Produces for FlatCopyRelation {
    fn produced(&self) -> Vec<OpeningId> {
        vec![self.input()]
    }
}

impl<F: JoltField> Relation<F> for FlatCopyRelation {}

impl<F: JoltField> SetupRelation<F> for FlatCopyRelation {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> Result<RelationSetup<F>, ProofVerifyError> {
        reduced_output_setup(accumulator, &self.shape.node)
    }
}

impl AtlasRelation for FlatCopyRelation {
    type Spec = Reshape;

    fn node(&self) -> &ComputationNode {
        &self.shape.node
    }
}

impl<F: JoltField> TieRelation<F> for FlatCopyRelation {}
