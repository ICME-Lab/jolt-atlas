use crate::{
    onnx_proof::{ops::OperatorProofTrait, ProofId, Prover, Verifier},
    utils::opening_access::AccOpeningAccessor,
};
use atlas_onnx_tracer::{node::ComputationNode, ops::Input};
use joltworks::{
    field::JoltField,
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Input {
    #[tracing::instrument(skip_all, name = "Input::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        // Assert claim is already cached
        let _opening = AccOpeningAccessor::new(&prover.accumulator, node).get_reduced_opening();
        vec![]
    }

    #[tracing::instrument(skip_all, name = "Input::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // Check input_claim == IO.evaluate_input(r_input)
        let (r_node_input, input_claim) =
            AccOpeningAccessor::new(&verifier.accumulator, node).get_reduced_opening();
        let input = verifier.io.inputs[verifier
            .io
            .input_indices
            .iter()
            .position(|&idx| idx == node.idx)
            .unwrap()]
        .padded_next_power_of_two();
        let expected_claim = MultilinearPolynomial::from(input).evaluate(&r_node_input.r);
        if expected_claim != input_claim {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Input claim does not match expected claim".to_string(),
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

    fn input_only_model(input_shape: &[usize]) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(input_shape.to_vec());
        b.mark_output(i);
        b.build()
    }

    #[test]
    fn test_input_non_power_of_two_input_len() {
        let t = 1000;
        let mut rng = StdRng::seed_from_u64(0x991);
        let input = Tensor::<i32>::random_small(&mut rng, &[t]);
        let model = input_only_model(&[t]);
        unit_test_op(model, &[input]);
    }
}
