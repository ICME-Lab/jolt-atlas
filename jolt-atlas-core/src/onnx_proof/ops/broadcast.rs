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
    #[tracing::instrument(skip_all, name = "Broadcast::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let params = BroadcastParams::new(
            node.clone(),
            &prover.accumulator,
            &prover.preprocessing.model.graph,
        );
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

/// Parameters for proving broadcast operations.
///
/// Broadcasting expands tensors to larger shapes by replicating elements.
#[derive(Clone)]
pub struct BroadcastParams<F: JoltField> {
    r_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
    input_raw_dims: Vec<usize>,
    output_raw_dims: Vec<usize>,
}

impl<F: JoltField> BroadcastParams<F> {
    /// Create new broadcast parameters from a computation node and opening accumulator.
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
        graph: &ComputationGraph,
    ) -> Self {
        let r_output = accumulator
            .get_node_output_opening(computation_node.idx)
            .0
            .r;
        let input_raw_dims = graph
            .nodes
            .get(&computation_node.inputs[0])
            .expect("Broadcast node should have an input")
            .output_dims
            .clone();
        let output_raw_dims = computation_node.output_dims.clone();
        Self {
            r_output,
            computation_node,
            input_raw_dims,
            output_raw_dims,
        }
    }
}

/// Prover state for broadcast sumcheck protocol.
///
/// Maintains transformed input variables and claims for proving the broadcast relation.
pub struct BroadcastProver<F: JoltField> {
    params: BroadcastParams<F>,
    r_input: Vec<F::Challenge>,
    claim_A: F,
}

impl<F: JoltField> BroadcastProver<F> {
    /// Initialize the prover with trace data and parameters.
    #[tracing::instrument(skip_all)]
    pub fn initialize(trace: &Trace, params: BroadcastParams<F>) -> Self {
        let LayerData { operands, output } = Trace::layer_data(trace, &params.computation_node);
        let [operand] = operands[..] else {
            panic!("Expected one operand for Broadcast operation")
        };

        // let input_dims = operand.dims();
        let output_dims = output.dims();
        let broadcast_tensor =
            build_broadcast_tensor(&params.input_raw_dims, &params.output_raw_dims);

        let (r_input, _r_broadcast) =
            split_broadcast_vars::<F>(output_dims, broadcast_tensor.dims(), &params.r_output);

        let mut operand = operand.clone();
        // operand.pad_next_power_of_two();
        let claim_A = MultilinearPolynomial::from(operand.clone()).evaluate(&r_input);

        #[cfg(test)]
        {
            // Ensure the broadcast tensor is correctly built,
            // Tensors are correctly padded, and the spliting of r_input/r_broadcast is correct
            let mut output = output.clone();
            // output.pad_next_power_of_two();
            let claim_O = MultilinearPolynomial::from(output.clone()).evaluate(&params.r_output);
            let broadcast_tensor = broadcast_tensor;
            let eval_I = MultilinearPolynomial::from(broadcast_tensor).evaluate(&_r_broadcast);
            assert_eq!(claim_O, claim_A * eval_I);
        }

        Self {
            params,
            r_input,
            claim_A,
        }
    }

    /// Prove the broadcast operation by adding the input polynomial to the opening accumulator.
    #[tracing::instrument(skip_all)]
    pub fn prove(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) {
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            self.r_input.clone().into(),
            self.claim_A,
        );
    }
}

/// Verifier for broadcast operation sumcheck protocol.
pub struct BroadcastVerifier<F: JoltField> {
    params: BroadcastParams<F>,
    r_input: Vec<F::Challenge>,
    eval_I: F,
}

impl<F: JoltField> BroadcastVerifier<F> {
    /// Create new broadcast verifier and compute the broadcast tensor evaluation.
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
        graph: &ComputationGraph,
    ) -> Self {
        let params = BroadcastParams::new(computation_node, accumulator, graph);
        let output_dims = params.computation_node.pow2_padded_output_dims();
        let broadcast_tensor = build_broadcast_tensor(
            &params.input_raw_dims,
            &params.output_raw_dims,
        );

        let (r_input, r_broadcast) =
            split_broadcast_vars::<F>(&output_dims, broadcast_tensor.dims(), &params.r_output);

        let eval_I = MultilinearPolynomial::from(broadcast_tensor).evaluate(&r_broadcast);

        Self {
            params,
            r_input,
            eval_I,
        }
    }

    /// Verify the broadcast operation by checking the polynomial openings.
    pub fn verify(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Result<(), ProofVerifyError> {
        // Cache the opening point for the input node
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            self.r_input.clone().into(),
        );

        // Retrieve the claim for the input node
        let operand_claim = accumulator.get_node_output_claim(
            self.params.computation_node.inputs[0],
            self.params.computation_node.idx,
        );

        let expected_claim_O = operand_claim * self.eval_I;

        let claim_O = accumulator
            .get_node_output_opening(self.params.computation_node.idx)
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
fn build_broadcast_tensor(
    input_raw_dims: &[usize],
    output_raw_dim: &[usize],
) -> Tensor<i32> {
    let bc_raw_dims = get_broadcast_dims(input_raw_dims, output_raw_dim);
    let bc_padded_dims: Vec<usize> = bc_raw_dims.iter().map(|d| d.next_power_of_two()).collect();
    let total_elems: usize = bc_padded_dims.iter().product();
    let mut bc_elements = Vec::with_capacity(total_elems);
    for linear_idx in 0..total_elems {
        // Convert row-major linear index into per-axis coordinates,
        // then mark 1 iff the coordinate stays inside the raw (unpadded) box.
        let mut rem = linear_idx;
        let mut inside_raw = true;
        for axis in (0..bc_padded_dims.len()).rev() {
            let dim = bc_padded_dims[axis];
            let coord = rem % dim;
            rem /= dim;
            if coord >= bc_raw_dims[axis] {
                inside_raw = false;
                break;
            }
        }
        bc_elements.push(if inside_raw { 1 } else { 0 });
    }

    Tensor::new(Some(&bc_elements), &bc_padded_dims).unwrap()
}

/// Computes the broadcast dimensions
///
/// # Returns
/// An array of dimensions, where each dimensions is either 1 if no broadcast is needed in that dimension,
/// or the target dimension otherwise.
///
fn get_broadcast_dims(input_dims: &[usize], output_dims: &[usize]) -> Vec<usize> {
    assert!(input_dims.len() <= output_dims.len());

    let mut broadcast_dims = output_dims.to_vec();
    for ((i, &target_dim), &input_dim) in output_dims
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
    use crate::onnx_proof::Fr;
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{model::test::ModelBuilder, model::Model, tensor::Tensor};
    use joltworks::field::JoltField;
    use rand::{rngs::StdRng, SeedableRng};

    fn broadcast_model(input_shape: &[usize], output_shape: &[usize]) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(input_shape.to_vec());
        let res = b.broadcast(i, output_shape.to_vec());
        b.mark_output(res);
        b.build()
    }

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
            let model = broadcast_model(&input_shape, &output_shape);
            unit_test_op(model, &[input]);
        }
    }

    #[test]
    fn test_build_broadcast_tensor_pads_last_dim_to_pow2_with_zero_tail() {
        let t = super::build_broadcast_tensor(&[1, 16, 1], &[1, 16, 768]);
        assert_eq!(t.dims(), &[1, 1, 1024]);
        assert_eq!(t.len(), 1024);
        assert!(t.data()[..768].iter().all(|&x| x == 1));
        assert!(t.data()[768..].iter().all(|&x| x == 0));
    }

    #[test]
    fn test_build_broadcast_tensor_collapses_non_broadcast_axes_to_one() {
        let t = super::build_broadcast_tensor(&[4, 1], &[4, 8]);
        assert_eq!(t.dims(), &[1, 8]);
        assert_eq!(t.len(), 8);
        assert!(t.data().iter().all(|&x| x == 1));
    }

    #[test]
    fn test_split_broadcast_vars_with_raw_output_and_padded_mask_still_partitions_vars() {
        // Even when broadcast tensor dims are pow2-padded (1024) while raw output uses 768,
        // split_broadcast_vars itself still partitions all output challenges.
        let output_raw_dims = vec![1, 16, 768];
        let bc = super::build_broadcast_tensor(&[1, 16, 1], &output_raw_dims);
        assert_eq!(bc.dims(), &[1, 1, 1024]);

        // Challenges are sampled for the padded output space: 1*16*1024 => 14 vars.
        let r_output: Vec<<Fr as JoltField>::Challenge> = vec![Default::default(); 14];
        let (r_input, r_broadcast) =
            super::split_broadcast_vars::<Fr>(&output_raw_dims, bc.dims(), &r_output);

        assert_eq!(r_input.len() + r_broadcast.len(), r_output.len());
    }
}
