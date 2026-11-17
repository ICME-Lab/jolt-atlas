//! Reshape as a zero-round tie.
//!
//! On power-of-two dims a reshape keeps every element at its flat row-major index, so the
//! output's multilinear extension is the operand's: the tie re-points the node's reduced output
//! claim at the operand, at the same point. A model compiled with padding has such dims
//! (`atlas_trace`'s `pad_graph` rejects a reshape whose layout changes under padding); a model
//! with any other dims on a reshape is rejected here.

use crate::node_view::{ComputationNode, ModelView};
use crate::onnx_proof::schedule::{OpPlan, ScheduleBuilder};
use crate::utils::dims::pow2_log;
use crate::{
    onnx_proof::relation::{OpRel, OpRelKind, RelationOpening},
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_claims::opening;
use atlas_trace::{op::Reshape, Model};

impl OpPlan for Reshape {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        schedule.reduce();
        schedule.tie(flat_copy(model, node)?);
        Ok(())
    }
}

/// `Reshape` as a tie: the output at `r` is the operand at `r`.
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  operand(r)
/// ```
///
/// `Err` if the node's dims are not ones the tie holds for ([`check_dims`]).
///
/// **S ⊆ R.** Every dim of the operand and of the output is a power of two with equal
/// products, so neither tensor is padded and both are the same flat row-major vector: a reshape
/// moves no element off its flat index. Their multilinear extensions over that index are equal.
pub fn flat_copy<F: JoltField>(
    model: &Model,
    node: &ComputationNode,
) -> Result<OpRel<F>, ProofVerifyError> {
    check_dims(model, node)?;
    let operand = OpeningIdBuilder::new(node).nodeio(Target::Input(0));
    Ok(OpRel {
        degree: 1,
        input: opening(RelationOpening::ReducedNodeOutput(node.idx)),
        output: opening(RelationOpening::Id(operand)),
        produced: vec![operand],
        ..OpRel::base(OpRelKind::Reshape, node)
    })
}

/// Check that a reshape `node` of `model` has the dims its tie holds for. `Err` if the node's
/// operand is not a node of the model, a dim of either is not a power of two, or their element
/// counts differ.
pub fn check_dims(model: &Model, node: &ComputationNode) -> Result<(), ProofVerifyError> {
    let [input_dims] = operand_dims(model, node)?
        .try_into()
        .map_err(|_| unsupported(node, "a reshape has one operand"))?;
    let num_vars = |dims: &[usize]| -> Result<usize, ProofVerifyError> {
        dims.iter().map(|&dim| pow2_log(dim)).sum()
    };
    if num_vars(&input_dims)? != num_vars(&node.output_dims)? {
        return Err(unsupported(
            node,
            format!(
                "{input_dims:?} and {:?} differ in element count",
                node.output_dims
            ),
        ));
    }
    Ok(())
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
