use crate::node_view::{ComputationNode, ModelView};
use crate::onnx_proof::ops::{OpLayout, ProveOp, VerifyOp};
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    atlas_relation_types,
    onnx_proof::{
        ops::{Prover, Verifier},
        relation::{
            prove_tie, reduced_output_setup, verify_tie, AtlasRelation, RelationDerived,
            RelationExpr, RelationId, RelationOpening, RelationSetup, TieRelation,
        },
        ProofId,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_trace::{op::Broadcast, Model, Tensor, Trace};
use core::panic;
use jolt_claims::{derived, opening, SymbolicSumcheck};
use jolt_field::Ring;
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::EvaluateMle,
        opening_proof::{OpeningAccumulator, OpeningId},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

crate::bind_relation!(Broadcast, BroadcastRelation);

impl OpLayout for Broadcast {}

impl<F: JoltField, T: Transcript> ProveOp<F, T> for Broadcast {
    #[tracing::instrument(skip_all, name = "Broadcast::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        prove_tie(
            &BroadcastRelation::for_node(&prover.preprocessing.model, node.clone()),
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        Vec::new()
    }
}

impl<F: JoltField, T: Transcript> VerifyOp<F, T> for Broadcast {
    #[tracing::instrument(skip_all, name = "Broadcast::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        verify_tie(
            &BroadcastRelation::for_node(&verifier.preprocessing.model, node.clone()),
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
    /// The relation of a broadcast `node` of `model`.
    pub fn for_node(model: &Model, node: ComputationNode) -> Self {
        let input_dims = model.view(node.inputs[0]).output_dims;
        Self::new(BroadcastShape { node, input_dims })
    }

    /// `(r_input, r_broadcast)`: the incoming point split into the operand's coordinates and the
    /// broadcast ones.
    fn split<F: JoltField>(&self, r_output: &[F]) -> (Vec<F>, Vec<F>) {
        let output_dims = &self.shape.node.output_dims;
        let ones = build_broadcast_tensor(&self.shape.input_dims, output_dims);
        split_broadcast_vars(output_dims, ones.dims(), r_output)
    }
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
        derived(RelationDerived::BroadcastOnes) * opening(RelationOpening::Id(self.produced()[0]))
    }
}

impl AtlasRelation for BroadcastRelation {
    type Spec = Broadcast;

    fn node(&self) -> &ComputationNode {
        &self.shape.node
    }

    fn setup<F: JoltField, T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> RelationSetup<F> {
        reduced_output_setup(accumulator, &self.shape.node)
    }

    fn produced(&self) -> Vec<OpeningId> {
        vec![OpeningIdBuilder::new(&self.shape.node).nodeio(Target::Input(0))]
    }

    fn opening_points<F: JoltField>(
        &self,
        setup: &RelationSetup<F>,
        _challenges: &[F],
    ) -> Vec<Vec<F>> {
        vec![self.split(&setup.incoming_point).0]
    }

    fn derive<F: JoltField>(
        &self,
        id: &RelationDerived,
        setup: &RelationSetup<F>,
        _challenges: &[F],
    ) -> F {
        match id {
            RelationDerived::BroadcastOnes => {
                let ones =
                    build_broadcast_tensor(&self.shape.input_dims, &self.shape.node.output_dims);
                ones.evaluate_mle(&self.split(&setup.incoming_point).1)
            }
            other => panic!("a broadcast derives no {other:?}"),
        }
    }
}

impl TieRelation for BroadcastRelation {
    fn claims<F: JoltField>(
        &self,
        trace: &Trace,
        setup: &RelationSetup<F>,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        let operand = TraceView::operand_tensors(trace, &self.shape.node)[0];
        vec![operand.evaluate_mle(&self.split(&setup.incoming_point).0)]
    }
}

/// Builds a unit tensor used for broadcast operation
///
/// # Returns
/// A tensor of dimensions equal to the broadcasted dimensions, filled with ones.
fn build_broadcast_tensor(input_dims: &[usize], target_dims: &[usize]) -> Tensor<i32> {
    let bc_dims = get_broadcast_dims(input_dims, target_dims);
    let num_elems: usize = bc_dims.iter().product();
    Tensor::from_slice(&vec![1i32; num_elems], &bc_dims).unwrap()
}

/// Computes the broadcast dimensions
///
/// # Returns
/// An array of dimensions, where each dimensions is either 1 if no broadcast is needed in that dimension,
/// or the target dimension otherwise.
///
fn get_broadcast_dims(input_dims: &[usize], target_dims: &[usize]) -> Vec<usize> {
    assert!(input_dims.len() <= target_dims.len());

    let mut broadcast_dims = target_dims.to_vec();
    for ((i, &target_dim), &input_dim) in target_dims
        .iter()
        .enumerate()
        .rev()
        .zip(input_dims.iter().rev())
    {
        if input_dim == target_dim {
            broadcast_dims[i] = 1;
        } else if input_dim != 1 {
            panic!(
                "Input dimension {} is not broadcastable to target dimension {}",
                input_dim, target_dim
            );
        }
    }
    broadcast_dims
}

/// Splits the opening point r_output into two parts:
/// - r_input: the variables corresponding to the input polynomial (non-broadcasted dimensions)
/// - r_broadcast: the variables corresponding to the broadcast polynomial (broadcasted dimensions)
fn split_broadcast_vars<F: JoltField>(
    output_dims: &[usize],
    broadcast_dims: &[usize],
    r_output: &[F],
) -> (Vec<F>, Vec<F>) {
    let mut r_input: Vec<F> = Vec::new();
    let mut r_broadcast: Vec<F> = Vec::new();
    let mut idx = 0;
    for (&output_dim, &broadcast_dim) in output_dims.iter().zip(broadcast_dims.iter()) {
        let dim_vars = output_dim.log_2();

        // Select which variables correspond to broadcasted dimensions
        if broadcast_dim == 1 {
            // This dimension is not broadcasted, evaluating the input polynomial on associated variables
            r_input.extend(&r_output[idx..idx + dim_vars]);
        } else {
            // This dimension is broadcasted, we evaluate the broadcast polynomial on associated variables
            r_broadcast.extend(&r_output[idx..idx + dim_vars]);
        }
        idx += dim_vars;
    }
    (r_input, r_broadcast)
}
