use crate::node_view::{ComputationNode, ModelView};
use crate::onnx_proof::ops::{OpLayout, VerifyOp};
use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};
use crate::{
    atlas_relation_types,
    onnx_proof::{
        ops::Verifier,
        relation::{
            not_derived, reduced_output_setup, verify_tie, AtlasRelation, RelationDerived,
            RelationExpr, RelationId, RelationOpening, RelationSetup, TieRelation,
        },
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{
    field::JoltField,
    poly::multilinear_polynomial::EvaluateMle,
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_trace::{op::Broadcast, Model, Tensor};
use atlas_transcript::Transcript;
use jolt_claims::{derived, opening, SymbolicSumcheck};
use jolt_field::Ring;

crate::bind_relation!(Broadcast, BroadcastRelation);

impl OpLayout for Broadcast {}

impl<F: JoltField, T: Transcript> VerifyOp<F, T> for Broadcast {
    #[tracing::instrument(skip_all, name = "Broadcast::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        verify_tie(
            &BroadcastRelation::for_node(&verifier.preprocessing.model, node.clone())?,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )
    }
}

/// A broadcast node and its operand's dims.
pub struct BroadcastShape {
    node: ComputationNode,
    input_dims: Vec<usize>,
}

/// `Broadcast` as a tie: the output at `r` is the operand at the non-broadcast coordinates of
/// `r`, times the all-ones replication tensor at the broadcast coordinates.
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  ones(r_broadcast) · operand(r_input)
/// ```
///
/// **S ⊆ R.** Broadcasting replicates each operand element along the broadcast axes, so the
/// output tensor is the operand's tensor product with an all-ones tensor.
pub struct BroadcastRelation {
    shape: BroadcastShape,
}

impl BroadcastRelation {
    /// The relation of a broadcast `node` of `model`. `Err` if the node's operand is not a node
    /// of the model.
    pub fn for_node(model: &Model, node: ComputationNode) -> Result<Self, ProofVerifyError> {
        let input = node
            .inputs
            .first()
            .copied()
            .filter(|&input| input < model.num_nodes())
            .ok_or_else(|| unsupported("the operand is not a node"))?;
        let input_dims = model.view(input).output_dims;
        Ok(Self::new(BroadcastShape { node, input_dims }))
    }

    /// `(r_input, r_broadcast)`: the incoming point split into the operand's coordinates and the
    /// broadcast ones. `Err` if the operand's dims do not broadcast to the node's, or
    /// `r_output` is not a point of the node's padded domain.
    pub fn split<F: JoltField>(
        &self,
        r_output: &[F],
    ) -> Result<(Vec<F>, Vec<F>), ProofVerifyError> {
        let output_dims = &self.shape.node.output_dims;
        let ones = build_broadcast_tensor(&self.shape.input_dims, output_dims)?;
        split_broadcast_vars(output_dims, ones.dims(), r_output)
    }

    fn input(&self) -> OpeningId {
        OpeningIdBuilder::new(&self.shape.node).nodeio(Target::Input(0))
    }
}

fn unsupported(what: impl std::fmt::Display) -> ProofVerifyError {
    ProofVerifyError::Malformed(format!("model: broadcast: {what}"))
}

impl SymbolicSumcheck for BroadcastRelation {
    atlas_relation_types!();
    type Shape = BroadcastShape;

    fn new(shape: BroadcastShape) -> Self {
        Self { shape }
    }

    fn id() -> RelationId {
        RelationId::Broadcast
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
        derived(RelationDerived::BroadcastOnes) * opening(RelationOpening::Id(self.input()))
    }
}

impl Produces for BroadcastRelation {
    fn produced(&self) -> Vec<OpeningId> {
        vec![self.input()]
    }
}

impl<F: JoltField> Relation<F> for BroadcastRelation {
    fn opening_points(
        &self,
        setup: &RelationSetup<F>,
        _challenges: &[F],
    ) -> Result<Vec<Vec<F>>, ProofVerifyError> {
        Ok(vec![self.split(&setup.incoming_point)?.0])
    }

    fn derive(
        &self,
        id: &RelationDerived,
        setup: &RelationSetup<F>,
        _challenges: &[F],
    ) -> Result<F, ProofVerifyError> {
        Ok(match id {
            RelationDerived::BroadcastOnes => {
                let ones =
                    build_broadcast_tensor(&self.shape.input_dims, &self.shape.node.output_dims)?;
                ones.evaluate_mle(&self.split(&setup.incoming_point)?.1)
            }
            other => return Err(not_derived(other)),
        })
    }
}

impl<F: JoltField> SetupRelation<F> for BroadcastRelation {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> Result<RelationSetup<F>, ProofVerifyError> {
        reduced_output_setup(accumulator, &self.shape.node)
    }
}

impl AtlasRelation for BroadcastRelation {
    type Spec = Broadcast;

    fn node(&self) -> &ComputationNode {
        &self.shape.node
    }
}

impl<F: JoltField> TieRelation<F> for BroadcastRelation {}

/// Builds a unit tensor used for broadcast operation
///
/// # Returns
/// A tensor of dimensions equal to the broadcasted dimensions, filled with ones; `Err` if the
/// input dims do not broadcast to the target dims.
fn build_broadcast_tensor(
    input_dims: &[usize],
    target_dims: &[usize],
) -> Result<Tensor<i32>, ProofVerifyError> {
    let bc_dims = get_broadcast_dims(input_dims, target_dims)?;
    let num_elems: usize = bc_dims.iter().product();
    Tensor::from_slice(&vec![1i32; num_elems], &bc_dims).map_err(unsupported)
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
