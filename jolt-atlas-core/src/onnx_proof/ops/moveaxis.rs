use atlas_onnx_tracer::{node::ComputationNode, ops::MoveAxis};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::opening_proof::{
        OpeningAccumulator, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

use crate::onnx_proof::{
    ops::{OperatorProofTrait, Prover, Verifier},
    ProofId,
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
    r_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
}

impl<F: JoltField> MoveAxisParams<F> {
    /// Create new moveaxis parameters from a computation node and opening accumulator.
    pub fn new(computation_node: ComputationNode, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let r_output = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(computation_node.idx),
                SumcheckId::Execution,
            )
            .0
            .r;
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
    r_input: Vec<F::Challenge>,
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
        // For MoveAxis, claim_A == claim_O since the data doesn't change
        let claim_O = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            self.r_input.clone().into(),
            claim_O,
        );
        accumulator.cache_virtual_operand_claims(transcript, &self.params.computation_node);
    }
}

/// Verifier for moveaxis (transpose) operations.
///
/// Verifies that input and output claims match with appropriately permuted challenge variables.
pub struct MoveAxisVerifier<F: JoltField> {
    params: MoveAxisParams<F>,
    r_input: Vec<F::Challenge>,
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
        // Cache the opening point for the input node
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            self.r_input.clone().into(),
        );
        accumulator.append_operand_claims(transcript, self.params.computation_node.idx);

        // Retrieve the claim for the input node
        let [operand_claim] = accumulator.get_operand_claims::<1>(self.params.computation_node.idx);

        // For MoveAxis, the input claim should equal the output claim
        let claim_O = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;

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
    r_output: &[F::Challenge],
    operator: &atlas_onnx_tracer::ops::Operator,
) -> Vec<F::Challenge> {
    use atlas_onnx_tracer::ops::Operator;

    let (source, destination) = match operator {
        Operator::MoveAxis(op) => (op.source, op.destination),
        _ => panic!("Expected MoveAxis operator"),
    };

    // Split r_output into groups, one for each axis in output_dims
    let mut challenge_groups: Vec<Vec<F::Challenge>> = Vec::new();
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
}
