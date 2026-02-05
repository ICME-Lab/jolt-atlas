use core::panic;

use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, Trace},
        ComputationGraph,
    },
    node::ComputationNode,
    ops::Broadcast,
    tensor::Tensor,
};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
        },
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

use crate::onnx_proof::{
    ops::{OperatorProofTrait, Prover, Verifier},
    ProofId,
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Broadcast {
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let params = BroadcastParams::new(node.clone(), &prover.accumulator);
        let broadcast_prover = BroadcastProver::initialize(&prover.trace, params);
        broadcast_prover.prove(&mut prover.accumulator, &mut prover.transcript);
        // Broadcast doesn't produce a sumcheck proof
        vec![]
    }

    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let broadcast_verifier = BroadcastVerifier::new(
            node.clone(),
            &verifier.accumulator,
            &verifier.preprocessing.model.graph,
        );
        broadcast_verifier.verify(&mut verifier.accumulator, &mut verifier.transcript)
    }
}

#[derive(Clone)]
pub struct BroadcastParams<F: JoltField> {
    r_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
}

impl<F: JoltField> BroadcastParams<F> {
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

pub struct BroadcastProver<F: JoltField> {
    params: BroadcastParams<F>,
    r_input: Vec<F::Challenge>,
    claim_A: F,
}

impl<F: JoltField> BroadcastProver<F> {
    pub fn initialize(trace: &Trace, params: BroadcastParams<F>) -> Self {
        let LayerData { operands, output } = Trace::layer_data(trace, &params.computation_node);
        let [operand] = operands[..] else {
            panic!("Expected one operand for Broadcast operation")
        };

        let input_dims = operand.dims();
        let output_dims = output.dims();
        let broadcast_tensor = build_broadcast_tensor(input_dims, output_dims);

        let (r_input, _r_broadcast) =
            split_broadcast_vars::<F>(output_dims, broadcast_tensor.dims(), &params.r_output);

        let mut operand = operand.clone();
        operand.pad_next_power_of_two();
        let claim_A = MultilinearPolynomial::from(operand.clone()).evaluate(&r_input);

        #[cfg(test)]
        {
            // Ensure the broadcast tensor is correctly built,
            // Tensors are correctly padded, and the spliting of r_input/r_broadcast is correct
            let mut output = output.clone();
            output.pad_next_power_of_two();
            let claim_O = MultilinearPolynomial::from(output.clone()).evaluate(&params.r_output);
            let mut broadcast_tensor = broadcast_tensor;
            broadcast_tensor.pad_next_power_of_two();
            let eval_I = MultilinearPolynomial::from(broadcast_tensor).evaluate(&_r_broadcast);
            assert_eq!(claim_O, claim_A * eval_I);
        }

        Self {
            params,
            r_input,
            claim_A,
        }
    }

    pub fn prove(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) {
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            self.r_input.clone().into(),
            self.claim_A,
        );
    }
}

pub struct BroadcastVerifier<F: JoltField> {
    params: BroadcastParams<F>,
    r_input: Vec<F::Challenge>,
    eval_I: F,
}

impl<F: JoltField> BroadcastVerifier<F> {
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
        graph: &ComputationGraph,
    ) -> Self {
        let params = BroadcastParams::new(computation_node, accumulator);
        let input_dims = &graph
            .nodes
            .get(&params.computation_node.inputs[0])
            .expect("Broadcast node should have an input")
            .output_dims;
        let output_dims = &params.computation_node.output_dims;

        let mut broadcast_tensor = build_broadcast_tensor(input_dims, output_dims);

        let (r_input, r_broadcast) =
            split_broadcast_vars::<F>(output_dims, broadcast_tensor.dims(), &params.r_output);

        broadcast_tensor.pad_next_power_of_two();
        let eval_I = MultilinearPolynomial::from(broadcast_tensor).evaluate(&r_broadcast);

        Self {
            params,
            r_input,
            eval_I,
        }
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

        let expected_claim_O = claim_A * self.eval_I;

        let claim_O = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;

        if expected_claim_O != claim_O {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Broadcast claim does not match expected claim".to_string(),
            ));
        }

        Ok(())
    }
}

/// Builds a unit tensor used for broadcast operation
///
/// # Returns
/// A tensor of dimensions equal to the broadcasted dimensions, filled with ones.
fn build_broadcast_tensor(input_dims: &[usize], target_dims: &[usize]) -> Tensor<i32> {
    let bc_dims = get_broadcast_dims(input_dims, target_dims);
    let num_elems: usize = bc_dims.iter().product();
    Tensor::new(Some(&vec![1i32; num_elems]), &bc_dims).unwrap()
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
    r_output: &[F::Challenge],
) -> (Vec<F::Challenge>, Vec<F::Challenge>) {
    let mut r_input: Vec<F::Challenge> = Vec::new();
    let mut r_broadcast: Vec<F::Challenge> = Vec::new();
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
    fn test_broadcast() {
        let mut rng = StdRng::seed_from_u64(0x888);
        let test_IO = vec![
            //------Input Shape | Output Shape
            /*-------*/ (vec![4], vec![8, 4]),
            /*----*/ (vec![1, 4], vec![4, 4]),
            /*----*/ (vec![4, 1], vec![4, 8]),
            /*-*/ (vec![1, 1, 4], vec![2, 4, 4]),
            /*-*/ (vec![1, 4, 1], vec![2, 4, 8]),
        ];

        for (input_shape, output_shape) in test_IO {
            let input = Tensor::<i32>::random_small(&mut rng, &input_shape);
            let model = model::test::broadcast_model(&input_shape, &output_shape);
            let trace = model.trace(&[input]);

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
                ProverOpeningAccumulator::new();
            let verifier_transcript = &mut Blake2bTranscript::new(&[]);
            let mut verifier_opening_accumulator: VerifierOpeningAccumulator<Fr> =
                VerifierOpeningAccumulator::new();

            let r_node_output: Vec<<Fr as JoltField>::Challenge> =
                prover_transcript.challenge_vector_optimized::<Fr>(max_vars);
            let _r_node_output: Vec<<Fr as JoltField>::Challenge> =
                verifier_transcript.challenge_vector_optimized::<Fr>(max_vars);

            let broadcast_claim =
                MultilinearPolynomial::from(output.clone()).evaluate(&r_node_output);
            prover_opening_accumulator.append_virtual(
                prover_transcript,
                VirtualPolynomial::NodeOutput(output_index),
                SumcheckId::Execution,
                r_node_output.clone().into(),
                broadcast_claim,
            );

            let params: BroadcastParams<Fr> =
                BroadcastParams::new(computation_node.clone(), &prover_opening_accumulator);
            let broadcast_prover = BroadcastProver::initialize(&trace, params);

            broadcast_prover.prove(&mut prover_opening_accumulator, prover_transcript);

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

            let broadcast_verifier = BroadcastVerifier::new(
                computation_node.clone(),
                &verifier_opening_accumulator,
                &model.graph,
            );

            let res =
                broadcast_verifier.verify(&mut verifier_opening_accumulator, verifier_transcript);

            prover_transcript.compare_to(verifier_transcript.clone());
            assert!(res.is_ok(), "Broadcast verification failed");
        }
    }
}
