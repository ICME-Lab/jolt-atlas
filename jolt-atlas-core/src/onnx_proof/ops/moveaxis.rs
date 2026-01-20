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

    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let moveaxis_verifier = MoveAxisVerifier::new(node.clone(), &verifier.accumulator);
        moveaxis_verifier.verify(&mut verifier.accumulator, &mut verifier.transcript)
    }
}

#[derive(Clone)]
pub struct MoveAxisParams<F: JoltField> {
    r_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
}

impl<F: JoltField> MoveAxisParams<F> {
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

pub struct MoveAxisProver<F: JoltField> {
    params: MoveAxisParams<F>,
    r_input: Vec<F::Challenge>,
}

impl<F: JoltField> MoveAxisProver<F> {
    pub fn initialize(params: MoveAxisParams<F>) -> Self {
        let r_input = permute_challenge_groups::<F>(
            &params.computation_node.output_dims,
            &params.r_output,
            &params.computation_node.operator,
        );

        Self { params, r_input }
    }

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
    }
}

pub struct MoveAxisVerifier<F: JoltField> {
    params: MoveAxisParams<F>,
    r_input: Vec<F::Challenge>,
}

impl<F: JoltField> MoveAxisVerifier<F> {
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

        // Retrieve the claim for the input node
        let claim_A = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
                SumcheckId::Execution,
            )
            .1;

        // For MoveAxis, the input claim should equal the output claim
        let claim_O = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;

        if claim_A != claim_O {
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
    use super::*;
    use ark_bn254::Fr;
    use atlas_onnx_tracer::{
        model::{
            self,
            trace::{LayerData, Trace},
        },
        tensor::Tensor,
    };
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
            let model = model::test::moveaxis_model(&input_shape, source, destination);
            let trace = model.trace(&[input.clone()]);

            let output_index = model.outputs()[0];
            let computation_node = &model[output_index];
            let LayerData {
                operands: _,
                output,
            } = Trace::layer_data(&trace, computation_node);

            let mut output = output.clone();
            output.pad_next_power_of_two();
            let max_vars: usize = output.dims().iter().map(|&d| d.log_2()).sum();

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

            let moveaxis_claim =
                MultilinearPolynomial::from(output.clone()).evaluate(&r_node_output);
            prover_opening_accumulator.append_virtual(
                prover_transcript,
                VirtualPolynomial::NodeOutput(output_index),
                SumcheckId::Execution,
                r_node_output.clone().into(),
                moveaxis_claim,
            );

            let params: MoveAxisParams<Fr> =
                MoveAxisParams::new(computation_node.clone(), &prover_opening_accumulator);
            let moveaxis_prover = MoveAxisProver::initialize(params);

            moveaxis_prover.prove(&mut prover_opening_accumulator, prover_transcript);

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

            let moveaxis_verifier =
                MoveAxisVerifier::new(computation_node.clone(), &verifier_opening_accumulator);

            let res =
                moveaxis_verifier.verify(&mut verifier_opening_accumulator, verifier_transcript);

            prover_transcript.compare_to(verifier_transcript.clone());
            assert!(res.is_ok(), "MoveAxis verification failed");
        }
    }
}
