use atlas_onnx_tracer::{
    node::ComputationNode,
    ops::{MoveAxis, Operator},
};
use joltworks::{
    field::JoltField,
    poly::opening_proof::{
        OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

use crate::{
    onnx_proof::{
        ops::{OperatorProofTrait, Prover, Verifier},
        ProofId,
    },
    utils::opening_id_builder::{AccOpeningAccessor, Target},
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for MoveAxis {
    #[tracing::instrument(skip_all, name = "MoveAxis::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let params = MoveAxisParams::<F>::new(node.clone(), &prover.accumulator);
        let moveaxis_prover = MoveAxisProver::initialize(params);
        moveaxis_prover.prove(&mut prover.accumulator, &mut prover.transcript);
        vec![]
    }

    #[tracing::instrument(skip_all, name = "MoveAxis::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let moveaxis_verifier = MoveAxisVerifier::new(node.clone(), &verifier.accumulator);
        moveaxis_verifier.verify(&mut verifier.accumulator, &mut verifier.transcript)
    }
}

/// Parameters for proving moveaxis (transpose) operations.
///
/// MoveAxis reorders tensor dimensions (axes) without changing the underlying data.
#[derive(Clone)]
pub struct MoveAxisParams<F: JoltField> {
    r_output: Vec<F>,
    computation_node: ComputationNode,
}

impl<F: JoltField> MoveAxisParams<F> {
    /// Create new moveaxis parameters from a computation node and opening accumulator.
    pub fn new(computation_node: ComputationNode, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);
        let r_output = accessor.get_reduced_opening().0.r;
        Self {
            r_output,
            computation_node,
        }
    }
}

/// Prover state for moveaxis (transpose) operations.
///
/// Maintains permuted input challenge variables that correspond to the reordered axes.
pub struct MoveAxisProver<F: JoltField> {
    params: MoveAxisParams<F>,
    r_input: Vec<F>,
}

impl<F: JoltField> MoveAxisProver<F> {
    /// Initialize the prover with parameters, computing the permuted input challenges.
    pub fn initialize(params: MoveAxisParams<F>) -> Self {
        let r_input = permute_challenge_groups::<F>(
            &params.computation_node.output_dims,
            &params.r_output,
            &params.computation_node.operator,
        );

        Self { params, r_input }
    }

    /// Generate the proof for the moveaxis operation.
    pub fn prove(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) {
        let opening_point = OpeningPoint::new(self.r_input.clone());
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .to_provider(transcript, opening_point);
        // For MoveAxis, claim_A == claim_O since the data doesn't change.
        let claim_O = provider.get_reduced_opening().1;
        provider.append_node_io(Target::Input(0), claim_O);
    }
}

/// Verifier for moveaxis (transpose) operations.
///
/// Verifies that input and output claims match with appropriately permuted challenge variables.
pub struct MoveAxisVerifier<F: JoltField> {
    params: MoveAxisParams<F>,
    r_input: Vec<F>,
}

impl<F: JoltField> MoveAxisVerifier<F> {
    /// Create a new verifier for the moveaxis operation.
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = MoveAxisParams::new(computation_node, accumulator);

        let r_input = permute_challenge_groups::<F>(
            &params.computation_node.output_dims,
            &params.r_output,
            &params.computation_node.operator,
        );

        Self { params, r_input }
    }

    /// Verify the moveaxis operation.
    pub fn verify(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Result<(), ProofVerifyError> {
        let opening_point = OpeningPoint::new(self.r_input.clone());
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .to_provider(transcript, opening_point);

        // Cache and retrieve the claim for the input node.
        provider.append_node_io(Target::Input(0));
        let operand_claim = provider.get_node_io(Target::Input(0)).1;

        // For MoveAxis, the input claim should equal the output claim.
        let claim_O = provider.get_reduced_opening().1;

        if operand_claim != claim_O {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "MoveAxis claim does not match expected claim".to_string(),
            ));
        }

        Ok(())
    }
}

/// Permutes challenge variable groups to reverse the moveaxis transformation
fn permute_challenge_groups<F: JoltField>(
    output_dims: &[usize],
    r_output: &[F],
    operator: &Operator,
) -> Vec<F> {
    let (source, destination) = match operator {
        Operator::MoveAxis(op) => (op.source, op.destination),
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

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{model::test::ModelBuilder, model::Model, tensor::Tensor};
    use rand::{rngs::StdRng, SeedableRng};

    fn moveaxis_model(input_shape: &[usize], source: usize, destination: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(input_shape.to_vec());
        let res = b.moveaxis(i, source, destination);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_moveaxis() {
        let mut rng = StdRng::seed_from_u64(0x777);
        let test_cases = vec![
            // (input_shape, source, destination)
            (vec![4, 8], 0, 1),
            (vec![4, 8], 1, 0),
            (vec![2, 4, 8], 0, 1),
            (vec![2, 4, 8], 0, 2),
            (vec![2, 4, 8], 1, 2),
        ];

        for (input_shape, source, destination) in test_cases {
            let input = Tensor::<i32>::random_small(&mut rng, &input_shape);
            let model = moveaxis_model(&input_shape, source, destination);
            unit_test_op(model, &[input]);
        }
    }

    #[test]
    #[ignore = "TODO: non-power-of-two moveaxis path not fully validated yet"]
    fn test_moveaxis_non_power_of_two_input_len() {
        let mut rng = StdRng::seed_from_u64(0x778);
        let input_shape = vec![5, 7];
        let input = Tensor::<i32>::random_small(&mut rng, &input_shape);
        let model = moveaxis_model(&input_shape, 0, 1);
        unit_test_op(model, &[input]);
    }
}
