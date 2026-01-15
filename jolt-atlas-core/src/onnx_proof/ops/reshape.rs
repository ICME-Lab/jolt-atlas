use atlas_onnx_tracer::{model::ComputationGraph, node::ComputationNode};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::{
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
};

#[derive(Clone)]
pub struct ReshapeParams<F: JoltField> {
    claim_A: F,
    r_node_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
}

impl<F: JoltField> ReshapeParams<F> {
    pub fn new(computation_node: ComputationNode, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let (r_output, claim_A) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(computation_node.idx),
            SumcheckId::Execution,
        );
        Self {
            claim_A,
            r_node_output: r_output.r,
            computation_node,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ReshapeParams<F> {
    fn degree(&self) -> usize {
        0
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.claim_A
    }

    fn normalize_opening_point(&self, _challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::default()
    }

    fn num_rounds(&self) -> usize {
        0
    }
}

pub struct ReshapeProver<F: JoltField> {
    params: ReshapeParams<F>,
}

impl<F: JoltField> ReshapeProver<F> {
    pub fn new(params: ReshapeParams<F>) -> Self {
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ReshapeProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UniPoly<F> {
        // This dummy sumcheck has no rounds, hence this method should never be called
        unimplemented!()
    }

    fn ingest_challenge(&mut self, _r_j: F::Challenge, _round: usize) {
        unimplemented!()
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // For reshape, the opening point is identical to the output's opening point
        // since the multilinear polynomial representation is the same
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            self.params.r_node_output.clone().into(),
            self.params.claim_A,
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
        _graph: &ComputationGraph,
    ) -> Self {
        let params = ReshapeParams::new(computation_node, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ReshapeVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // For reshape, the input claim equals the output claim
        accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
                SumcheckId::Execution,
            )
            .1
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // Same opening point as the output
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            self.params.r_node_output.clone().into(),
        );
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
        subprotocols::sumcheck::Sumcheck,
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
            let mut prover_sumcheck = ReshapeProver::new(params);

            let (proof, r_sumcheck) = Sumcheck::prove(
                &mut prover_sumcheck,
                &mut prover_opening_accumulator,
                prover_transcript,
            );

            // Transfer claims to verifier
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

            let verifier_sumcheck = ReshapeVerifier::new(
                computation_node.clone(),
                &verifier_opening_accumulator,
                &model.graph,
            );
            let res = Sumcheck::verify(
                &proof,
                &verifier_sumcheck,
                &mut verifier_opening_accumulator,
                verifier_transcript,
            );
            prover_transcript.compare_to(verifier_transcript.clone());
            let r_sumcheck_verif = res.unwrap();
            assert_eq!(r_sumcheck, r_sumcheck_verif);
        }
    }
}
