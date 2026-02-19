use crate::onnx_proof::{
    ops::{OperatorProofTrait, Prover, Verifier},
    ProofId,
};
use atlas_onnx_tracer::{node::ComputationNode, ops::Reshape};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::opening_proof::{
        OpeningAccumulator, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Reshape {
    #[tracing::instrument(skip_all, name = "Reshape::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let params = ReshapeParams::<F>::new(node.clone(), &prover.accumulator);
        let reshape_prover = ReshapeProver::initialize(params);
        reshape_prover.prove(&mut prover.accumulator, &mut prover.transcript);
        vec![]
    }

    #[tracing::instrument(skip_all, name = "Reshape::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let reshape_verifier = ReshapeVerifier::new(node.clone(), &verifier.accumulator);
        reshape_verifier.verify(&mut verifier.accumulator, &mut verifier.transcript)
    }
}

/// Parameters for proving reshape operations.
///
/// Reshape changes tensor dimensions without modifying data layout in memory.
#[derive(Clone)]
pub struct ReshapeParams<F: JoltField> {
    r_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
}

impl<F: JoltField> ReshapeParams<F> {
    /// Create new reshape parameters from a computation node and opening accumulator.
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

/// Prover state for reshape operations.
///
/// Since reshape doesn't change data, proving simply involves showing that input and output
/// evaluate to the same claim at the same opening point.
pub struct ReshapeProver<F: JoltField> {
    params: ReshapeParams<F>,
}

impl<F: JoltField> ReshapeProver<F> {
    /// Initialize the prover with parameters.
    pub fn initialize(params: ReshapeParams<F>) -> Self {
        Self { params }
    }

    /// Generate the proof for the reshape operation.
    pub fn prove(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) {
        // For reshape, the opening point is identical to the output's opening point
        // since the multilinear polynomial representation is the same.
        // Also, claim_A == claim_O since reshape doesn't change the data.
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
            self.params.r_output.clone().into(),
            claim_O,
        );
        accumulator.cache_virtual_operand_claims(transcript, &self.params.computation_node);
    }
}

/// Verifier for reshape operations.
///
/// Verifies that input and output claims match for the reshape operation.
pub struct ReshapeVerifier<F: JoltField> {
    params: ReshapeParams<F>,
}

impl<F: JoltField> ReshapeVerifier<F> {
    /// Create a new verifier for the reshape operation.
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = ReshapeParams::new(computation_node, accumulator);
        Self { params }
    }

    /// Verify the reshape operation.
    pub fn verify(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Result<(), ProofVerifyError> {
        // Cache the opening point for the input node
        // For reshape, the opening point is identical to the output's opening point
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            self.params.r_output.clone().into(),
        );
        accumulator.append_operand_claims(transcript, self.params.computation_node.idx);

        // Retrieve the claim for the input node
        let operand_claim =
            accumulator.get_operand_claims::<1>(self.params.computation_node.idx)[0];

        // For reshape, the input claim should equal the output claim
        let claim_O = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;

        if operand_claim != claim_O {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Reshape claim does not match expected claim".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, Model},
        tensor::Tensor,
    };
    use rand::{rngs::StdRng, SeedableRng};

    fn reshape_model(input_shape: &[usize], output_shape: &[usize]) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(input_shape.to_vec());
        let res = b.reshape(i, output_shape.to_vec());
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_reshape() {
        let mut rng = StdRng::seed_from_u64(0x999);
        let test_cases = vec![
            // (input_shape, output_shape)
            (vec![16], vec![4, 4]),
            (vec![4, 8], vec![8, 4]),
            (vec![2, 4, 8], vec![64]),
            (vec![2, 4, 8], vec![8, 8]),
        ];

        for (input_shape, output_shape) in test_cases {
            let input = Tensor::<i32>::random_small(&mut rng, &input_shape);
            let model = reshape_model(&input_shape, &output_shape);
            unit_test_op(model, &[input]);
        }
    }
}
