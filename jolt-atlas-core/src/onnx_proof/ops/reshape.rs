use atlas_onnx_tracer::node::ComputationNode;
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::opening_proof::{
        OpeningAccumulator, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

#[derive(Clone)]
pub struct ReshapeParams<F: JoltField> {
    r_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
}

impl<F: JoltField> ReshapeParams<F> {
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

pub struct ReshapeProver<F: JoltField> {
    params: ReshapeParams<F>,
}

impl<F: JoltField> ReshapeProver<F> {
    pub fn initialize(params: ReshapeParams<F>) -> Self {
        Self { params }
    }

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
    }
}

pub struct ReshapeVerifier<F: JoltField> {
    params: ReshapeParams<F>,
}

impl<F: JoltField> ReshapeVerifier<F> {
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = ReshapeParams::new(computation_node, accumulator);
        Self { params }
    }

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

        // Retrieve the claim for the input node
        let claim_A = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
                SumcheckId::Execution,
            )
            .1;

        // For reshape, the input claim should equal the output claim
        let claim_O = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;

        if claim_A != claim_O {
            return Err(ProofVerifyError::InvalidOpeningProof);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use atlas_onnx_tracer::{model, tensor::Tensor};
    use common::VirtualPolynomial;
    use joltworks::{
        field::JoltField,
        poly::{
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            opening_proof::{
                OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
                BIG_ENDIAN,
            },
        },
        transcripts::{Blake2bTranscript, Transcript},
    };
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_reshape() {
        let mut rng = StdRng::seed_from_u64(0x999);
        let test_cases = vec![
            // (input_shape, output_shape)
            (vec![12], vec![3, 4]),
            (vec![2, 6], vec![3, 4]),
            (vec![2, 3, 4], vec![24]),
            (vec![2, 3, 4], vec![6, 4]),
        ];

        for (input_shape, output_shape) in test_cases {
            let input = Tensor::<i32>::random_small(&mut rng, &input_shape);
            let model = model::test::reshape_model(&input_shape, &output_shape);

            let output_index = model.outputs()[0];
            let computation_node = &model[output_index];

            let mut input_padded = input.clone();
            input_padded.pad_next_power_of_two();
            let max_vars = input_padded
                .data()
                .len()
                .next_power_of_two()
                .trailing_zeros() as usize;

            let prover_transcript = &mut Blake2bTranscript::new(&[]);
            let mut prover_opening_accumulator: ProverOpeningAccumulator<Fr> =
                ProverOpeningAccumulator::new(max_vars);
            let verifier_transcript = &mut Blake2bTranscript::new(&[]);
            let mut verifier_opening_accumulator: VerifierOpeningAccumulator<Fr> =
                VerifierOpeningAccumulator::new(max_vars);

            let r_node_output: Vec<<Fr as JoltField>::Challenge> =
                prover_transcript.challenge_vector_optimized::<Fr>(max_vars);
            let _r_node_output: Vec<<Fr as JoltField>::Challenge> =
                verifier_transcript.challenge_vector_optimized::<Fr>(max_vars);

            let reshape_claim =
                MultilinearPolynomial::from(input_padded.clone()).evaluate(&r_node_output);
            prover_opening_accumulator.append_virtual(
                prover_transcript,
                VirtualPolynomial::NodeOutput(output_index),
                SumcheckId::Execution,
                r_node_output.clone().into(),
                reshape_claim,
            );

            let params: ReshapeParams<Fr> =
                ReshapeParams::new(computation_node.clone(), &prover_opening_accumulator);
            let reshape_prover = ReshapeProver::initialize(params);

            reshape_prover.prove(&mut prover_opening_accumulator, prover_transcript);

            // Take claims
            for (key, (_, value)) in &prover_opening_accumulator.openings {
                let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
                verifier_opening_accumulator
                    .openings
                    .insert(*key, (empty_point, *value));
            }

            verifier_opening_accumulator.append_virtual(
                verifier_transcript,
                VirtualPolynomial::NodeOutput(output_index),
                SumcheckId::Execution,
                r_node_output.into(),
            );

            let reshape_verifier =
                ReshapeVerifier::new(computation_node.clone(), &verifier_opening_accumulator);

            let res =
                reshape_verifier.verify(&mut verifier_opening_accumulator, verifier_transcript);

            prover_transcript.compare_to(verifier_transcript.clone());
            assert!(res.is_ok(), "Reshape verification failed");
        }
    }
}
