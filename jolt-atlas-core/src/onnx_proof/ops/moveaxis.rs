use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    atlas_relation_types, impl_tie_proof_api,
    onnx_proof::{
        ops::{OperatorProofTrait, Prover, Verifier},
        relation::{
            reduced_output_setup, AtlasRelation, EngineRelation, RelationExpr, RelationId,
            RelationOpening, RelationSetup,
        },
        ProofId,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_trace::{op::MoveAxis, Op, Trace};
use jolt_claims::{opening, SymbolicSumcheck};
use jolt_field::Ring;
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{OpeningAccumulator, OpeningId},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

impl_tie_proof_api!(MoveAxis, AxisPermutationRelation);

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
pub struct AxisPermutationRelation {
    node: ComputationNode,
}

impl SymbolicSumcheck for AxisPermutationRelation {
    atlas_relation_types!();
    type Shape = ComputationNode;

    fn new(node: ComputationNode) -> Self {
        Self { node }
    }

    fn id() -> RelationId {
        RelationId::AxisPermutation
    }

    fn rounds(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::ReducedNodeOutput(self.node.idx))
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::Id(self.produced()[0]))
    }
}

impl AtlasRelation for AxisPermutationRelation {
    type Spec = MoveAxis;

    fn node(&self) -> &ComputationNode {
        &self.node
    }

    fn setup<F: JoltField, T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> RelationSetup<F> {
        reduced_output_setup(accumulator, &self.node)
    }

    fn produced(&self) -> Vec<OpeningId> {
        vec![OpeningIdBuilder::new(&self.node).nodeio(Target::Input(0))]
    }

    fn opening_points<F: JoltField>(
        &self,
        setup: &RelationSetup<F>,
        _challenges: &[F],
    ) -> Vec<Vec<F>> {
        vec![permute_challenge_groups(
            &self.node.output_dims,
            &setup.incoming_point,
            &self.node.operator,
        )]
    }
}

impl EngineRelation for AxisPermutationRelation {
    fn witnesses<F: JoltField>(&self, trace: &Trace) -> Vec<MultilinearPolynomial<F>> {
        let operand = TraceView::operand_tensors(trace, &self.node)[0];
        vec![MultilinearPolynomial::from(operand.padded_pow2())]
    }
}

/// Permutes challenge variable groups to reverse the moveaxis transformation
fn permute_challenge_groups<F: JoltField>(
    output_dims: &[usize],
    r_output: &[F],
    operator: &Op,
) -> Vec<F> {
    let (source, destination) = match operator {
        Op::MoveAxis(op) => (op.source, op.destination),
        _ => panic!("Expected MoveAxis operator"),
    };

    // Split r_output into groups, one for each axis in output_dims
    let mut challenge_groups: Vec<Vec<F>> = Vec::new();
    let mut offset = 0;

    for &dim in output_dims.iter() {
        let num_vars = dim.log_2();
        challenge_groups.push(r_output[offset..offset + num_vars].to_vec());
        offset += num_vars;
    }

    // We need to do the opposite, since we're going from output to input
    // Hence we take the group at 'destination' and put it at 'source'
    let dst_group = challenge_groups.remove(destination);
    challenge_groups.insert(source, dst_group);

    // Flatten the groups back into a single vector
    challenge_groups.into_iter().flatten().collect()
}
