use crate::utils::dims::{coord_to_linear, linear_to_coord};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::{Concat, Operator},
    utils::dims::Pad,
};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        opening_proof::{OpeningAccumulator, SumcheckId},
    },
    subprotocols::{
        gamma_fold::{GammaFoldProver, GammaFoldVerifier},
        sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, index_to_field_bitvector},
};

use crate::onnx_proof::{
    ops::{eval_reduction::NodeEvalReduction, OperatorProofTrait, ReductionFlow},
    ProofId, ProofType, Prover, Verifier,
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Concat {
    fn reduction_flow(&self) -> ReductionFlow {
        ReductionFlow::Custom
    }

    #[tracing::instrument(skip_all, name = "Concat::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let Operator::Concat(concat_op) = &node.operator else {
            panic!("Expected Concat operator")
        };

        let r_output = prover.accumulator.get_node_output_opening(node.idx).0;

        let raw_inputs_dims = {
            let LayerData { operands, .. } = Trace::layer_data(&prover.trace, node);
            operands
                .iter()
                .map(|tensor| tensor.dims())
                .collect::<Vec<_>>()
        };
        let input_count = raw_inputs_dims.len();
        assert!(input_count > 0, "Concat expects at least one operand");

        let axis = normalize_axis(concat_op.axis, node.output_dims.len());
        let ctx =
            ConcatGammaWeightsContext::new(&raw_inputs_dims, &node.output_dims, axis, &r_output.r);

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            Vec::with_capacity(input_count);
        for input_idx in 0..input_count {
            instances.push(Box::new(initialize_input_prover(
                node, input_idx, &ctx, prover,
            )));
        }
        let (proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );

        vec![(ProofId(node.idx, ProofType::Execution), proof)]
    }

    fn prove_with_reduction(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> (
        joltworks::subprotocols::evaluation_reduction::EvalReductionProof<F>,
        Vec<(ProofId, SumcheckInstanceProof<F, T>)>,
    ) {
        let eval_reduction_proof = NodeEvalReduction::prove(prover, node);
        let execution_proofs = self.prove(node, prover);
        (eval_reduction_proof, execution_proofs)
    }

    #[tracing::instrument(skip_all, name = "Concat::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;

        let Operator::Concat(concat_op) = &node.operator else {
            panic!("Expected Concat operator")
        };

        let r_output = verifier.accumulator.get_node_output_opening(node.idx).0;

        let raw_inputs_dims = {
            let graph = &verifier.preprocessing.model.graph;
            let input_nodes = graph.get_input_nodes(node);
            input_nodes
                .iter()
                .map(|input_node| input_node.output_dims.as_slice())
                .collect::<Vec<&[usize]>>()
        };
        let input_count = raw_inputs_dims.len();
        assert!(input_count > 0, "Concat expects at least one input node");

        let axis = normalize_axis(concat_op.axis, node.output_dims.len());
        let ctx =
            ConcatGammaWeightsContext::new(&raw_inputs_dims, &node.output_dims, axis, &r_output.r);

        let mut verifiers = Vec::with_capacity(input_count);
        for input_idx in 0..input_count {
            verifiers.push(initialize_input_verifier(node, input_idx, &ctx, verifier));
        }

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, T>> = verifiers
            .iter()
            .map(|instance| instance as &dyn SumcheckInstanceVerifier<F, T>)
            .collect();
        let r_sumcheck = BatchedSumcheck::verify(
            proof,
            instances,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;
        let _ = r_sumcheck;

        let output_claim = verifier.accumulator.get_node_output_opening(node.idx).1;
        let expected_output_claim = (0..input_count).fold(F::zero(), |running, input_idx| {
            let input_claim = verifier
                .accumulator
                .get_virtual_polynomial_opening(
                    claim_poly_input(node, input_idx),
                    SumcheckId::RLC(node.idx),
                )
                .1;
            running + input_claim
        });
        if output_claim != expected_output_claim {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Concat output-claim relation failed".to_string(),
            ));
        }

        Ok(())
    }

    fn verify_with_reduction(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
        eval_reduction_proof: &joltworks::subprotocols::evaluation_reduction::EvalReductionProof<F>,
    ) -> Result<(), ProofVerifyError> {
        NodeEvalReduction::verify(verifier, node, eval_reduction_proof)?;
        self.verify(node, verifier)
    }
}

fn initialize_input_prover<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    input_idx: usize,
    ctx: &ConcatGammaWeightsContext<F>,
    prover: &mut Prover<F, T>,
) -> GammaFoldProver<F> {
    let LayerData { operands, .. } = Trace::layer_data(&prover.trace, node);
    let input_tensor = operands
        .get(input_idx)
        .unwrap_or_else(|| panic!("Concat input_idx {input_idx} out of range"));

    GammaFoldProver::initialize(
        node.idx,
        claim_poly_input(node, input_idx),
        input_tensor,
        ctx.gamma_weights(input_idx),
        &mut prover.accumulator,
        &mut prover.transcript,
    )
}

fn initialize_input_verifier<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    input_idx: usize,
    ctx: &ConcatGammaWeightsContext<F>,
    verifier: &mut Verifier<'_, F, T>,
) -> GammaFoldVerifier<F> {
    let graph = &verifier.preprocessing.model.graph;
    let input_nodes = graph.get_input_nodes(node);
    let input_num_elements = input_nodes[input_idx].pow2_padded_num_output_elements();

    GammaFoldVerifier::initialize(
        node.idx,
        claim_poly_input(node, input_idx),
        input_num_elements,
        ctx.gamma_weights(input_idx),
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )
}

/// Normalizes an ONNX-style axis (possibly negative) to a canonical `[0, rank)` axis.
fn normalize_axis(axis: isize, rank: usize) -> usize {
    let rank_i = rank as isize;
    let normalized = if axis < 0 { axis + rank_i } else { axis };
    assert!(
        normalized >= 0 && normalized < rank_i,
        "Concat axis {axis} out of range for rank {rank}"
    );
    normalized as usize
}

/// Validates concat shape rules.
///
/// All inputs must share rank with output. Non-concat dimensions must match output,
/// and concat dimension in output must equal the sum of the same dimension across inputs.
///
/// Note:
/// Requires non-padded shapes.
fn validate_concat_shapes(inputs_dims: &[&[usize]], out_dims: &[usize], axis: usize) {
    assert!(!inputs_dims.is_empty(), "Concat expects at least one input");
    let rank = out_dims.len();
    for dims in inputs_dims {
        assert_eq!(dims.len(), rank, "Concat input rank mismatch");
    }

    for dim in 0..out_dims.len() {
        if dim == axis {
            let sum: usize = inputs_dims.iter().map(|dims| dims[dim]).sum();
            assert_eq!(
                out_dims[dim], sum,
                "Concat output axis dimension must equal sum of input dimensions"
            );
        } else {
            let out_dim = out_dims[dim];
            for dims in inputs_dims {
                assert_eq!(
                    dims[dim], out_dim,
                    "Concat non-axis dimensions must match between inputs"
                );
            }
        }
    }
}

/// Context for generating concat-aware gamma weights for each input polynomial.
struct ConcatGammaWeightsContext<F: JoltField> {
    /// Raw input dims, the source of truth for concat indexing.
    inputs_dims: Vec<Vec<usize>>,
    /// Padded output dims are used for stride computation.
    output_dims: Vec<usize>,
    /// Concat axis index.
    axis: usize,
    /// Eq evaluations over raw output linear indices.
    output_eq_evals: Vec<F>,
}

impl<F: JoltField> ConcatGammaWeightsContext<F> {
    fn new(raw_inputs: &[&[usize]], raw_output_dims: &[usize], axis: usize, r_output: &[F]) -> Self {
        validate_concat_shapes(raw_inputs, raw_output_dims, axis);

        let raw_inputs = raw_inputs.iter().map(|dims| dims.to_vec()).collect();
        let output_padded_dims = raw_output_dims
            .iter()
            .map(|dim| dim.next_power_of_two())
            .collect::<Vec<_>>();
        let output_padded_len: usize = output_padded_dims.iter().product();
        let output_num_vars = output_padded_len.trailing_zeros() as usize;
        assert_eq!(
            output_num_vars,
            r_output.len(),
            "Concat output opening length must match padded output domain"
        );
        let output_eq_evals = (0..output_padded_len)
            .map(|i| {
                let output_bits = index_to_field_bitvector::<F>(i as u64, output_num_vars);
                EqPolynomial::mle(&output_bits, r_output)
            })
            .collect();

        Self {
            inputs_dims: raw_inputs,
            output_dims: output_padded_dims,
            axis,
            output_eq_evals,
        }
    }

    /// Returns the linearized gamma weights for the concat output polynomial.
    fn output_gamma_weights(&self) -> Vec<F> {
        self.output_eq_evals.clone()
    }

    fn input_dims(&self, input_idx: usize) -> &[usize] {
        self.inputs_dims
            .get(input_idx)
            .unwrap_or_else(|| panic!("Concat input_idx {input_idx} out of range"))
    }

    /// Returns the axis offset (in output coordinates) for the given input.
    fn axis_offset(&self, input_idx: usize) -> usize {
        self.inputs_dims[..input_idx]
            .iter()
            .map(|dims| dims[self.axis])
            .sum()
    }

    /// Builds concat-aware gamma weights for one input tensor.
    ///
    /// The produced vector aligns with row-major linearization of the padded input.
    ///
    /// This used to recurse dimension-by-dimension for `gamma^i` weights. That worked because
    /// gamma powers factor cleanly across row-major strides. For `eq(r, i)` weights we instead
    /// need the final output linear index, so we map each input position directly to its
    /// corresponding output index and read the precomputed evaluation from `output_eq_evals`.
    ///
    /// Example
    /// ```ignore
    /// // For concat of [2,2] and [2,1] on axis 1.
    /// // Position of input 1, index [0,0] in output is [0, 2]
    /// // i.e. 2 in row-major order, so it's weight is gamma^2.
    /// // Complete weight vector for input 1 is [gamma^2, gamma^5]
    /// ```
    ///
    fn gamma_weights(&self, input_idx: usize) -> Vec<F> {
        let input_dims = self.input_dims(input_idx);
        let poly_len: usize = input_dims.iter().product();

        if poly_len == 0 {
            return Vec::new();
        }

        let axis_offset = self.axis_offset(input_idx);

        // This loop is index-independent and could be upgraded to `into_par_iter()` if selector
        // construction becomes a measurable hotspot for large concat inputs.
        let raw_weights = (0..poly_len)
            .map(|linear_idx| {
                let coord = linear_to_coord(linear_idx, input_dims);
                let mut output_coord = coord.clone();
                output_coord[self.axis] += axis_offset;
                let output_index = coord_to_linear(&output_coord, &self.output_dims);
                self.output_eq_evals[output_index]
            })
            .collect::<Vec<_>>();
        raw_weights.pad_next_power_of_two(input_dims)
    }
}

fn claim_poly_input(node: &ComputationNode, input_idx: usize) -> VirtualPolynomial {
    let input_node_idx = *node
        .inputs
        .get(input_idx)
        .unwrap_or_else(|| panic!("Concat input_idx {input_idx} out of range"));
    VirtualPolynomial::NodeOutput(input_node_idx)
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use ark_bn254::Fr;
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, Model},
        tensor::Tensor,
    };
    use joltworks::{field::JoltField, poly::eq_poly::EqPolynomial, utils::index_to_field_bitvector};
    use num_traits::Zero;
    use rand::{rngs::StdRng, SeedableRng};

    fn concat_model(input_shapes: Vec<Vec<usize>>, axis: isize) -> Model {
        let mut b = ModelBuilder::new();
        let inputs: Vec<_> = input_shapes
            .into_iter()
            .map(|shape| b.input(shape))
            .collect();
        let out = b.concat(&inputs, axis);
        b.mark_output(out);
        b.build()
    }

    fn eq_eval<F: JoltField>(r: &[F], index: usize) -> F {
        let bits = index_to_field_bitvector::<F>(index as u64, r.len());
        EqPolynomial::mle(&bits, r)
    }

    #[test]
    fn test_concat_power_of_two_shapes() {
        let cases: Vec<(Vec<usize>, Vec<usize>, isize)> = vec![
            // Rank 2, concat on positive axis.
            (vec![4, 2], vec![4, 2], 1),
            // Rank 2, concat on axis 0.
            (vec![2, 4], vec![2, 4], 0),
            // Rank 3, concat on last axis.
            (vec![2, 4, 2], vec![2, 4, 2], 2),
            // Rank 3, concat on negative axis (-2 => axis 1).
            (vec![2, 2, 4], vec![2, 2, 4], -2),
            // Rank 4, concat on last axis.
            (vec![2, 2, 2, 2], vec![2, 2, 2, 2], 3),
        ];

        for (idx, (a_shape, b_shape, axis)) in cases.into_iter().enumerate() {
            let mut rng = StdRng::seed_from_u64(0xC0D2 + idx as u64);
            let a = Tensor::<i32>::random_small(&mut rng, &a_shape);
            let b = Tensor::<i32>::random_small(&mut rng, &b_shape);
            let model = concat_model(vec![a_shape, b_shape], axis);
            unit_test_op(model, &[a, b]);
        }
    }

    #[test]
    fn test_concat_arbitrary_shapes() {
        let cases: Vec<(Vec<usize>, Vec<usize>, isize)> = vec![
            // Rank 2, concat on positive axis.
            (vec![3, 2], vec![3, 3], 1),
            // Rank 2, concat on axis 0.
            (vec![2, 3], vec![4, 3], 0),
            // Rank 3, concat on last axis.
            (vec![2, 3, 3], vec![2, 3, 2], 2),
            // Rank 3, concat on negative axis (-2 => axis 1).
            (vec![2, 1, 3], vec![2, 2, 3], -2),
            // Rank 4, concat on last axis.
            (vec![1, 2, 3, 1], vec![1, 2, 3, 2], 3),
        ];

        for (idx, (a_shape, b_shape, axis)) in cases.into_iter().enumerate() {
            let mut rng = StdRng::seed_from_u64(0xC0C2 + idx as u64);
            let a = Tensor::<i32>::random_small(&mut rng, &a_shape);
            let b = Tensor::<i32>::random_small(&mut rng, &b_shape);
            let model = concat_model(vec![a_shape, b_shape], axis);
            unit_test_op(model, &[a, b]);
        }
    }

    #[test]
    fn test_concat_multi_input_shapes() {
        let cases: Vec<(Vec<Vec<usize>>, isize)> = vec![
            (vec![vec![2, 3], vec![2, 1], vec![2, 4]], 1),
            (vec![vec![1, 2, 4], vec![3, 2, 4], vec![2, 2, 4]], 0),
            (
                vec![vec![2, 2, 1], vec![2, 2, 3], vec![2, 2, 2], vec![2, 2, 2]],
                -1,
            ),
        ];

        for (idx, (shapes, axis)) in cases.into_iter().enumerate() {
            let mut rng = StdRng::seed_from_u64(0xC0F0 + idx as u64);
            let inputs: Vec<Tensor<i32>> = shapes
                .iter()
                .map(|shape| Tensor::<i32>::random_small(&mut rng, shape))
                .collect();
            let model = concat_model(shapes, axis);
            unit_test_op(model, &inputs);
        }
    }

    #[test]
    fn test_concat_single_input_identity() {
        let mut rng = StdRng::seed_from_u64(0xC011);
        let shape = vec![3, 5, 2];
        let input = Tensor::<i32>::random_small(&mut rng, &shape);
        let model = concat_model(vec![shape], 1);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_concat_gamma_weights_visual_layout_rank3_axis1() {
        let r_output = vec![
            Fr::from(2u64),
            Fr::from(3u64),
            Fr::from(5u64),
            Fr::from(7u64),
        ];
        // Concat inputs [2,3,2] and [2,1,2] along axis 1.
        // Raw output [2,4,2] (3+1=4, already pow2).
        // Padded input A [2,4,2], padded input B [2,1,2], padded output [2,4,2].
        let ctx = super::ConcatGammaWeightsContext::new(
            &[&[2, 3, 2], &[2, 1, 2]],
            &[2, 4, 2],
            1,
            &r_output,
        );

        let output_weights = ctx.output_gamma_weights();
        let a_weights = ctx.gamma_weights(0);
        let b_weights = ctx.gamma_weights(1);

        // Output [2,4,2] is already pow2 (len=16), so all 16 entries are eq(r, i).
        let expected_output = (0..16)
            .map(|i| eq_eval(&r_output, i))
            .collect::<Vec<_>>();
        assert_eq!(output_weights, expected_output.as_slice());

        let mut expected_a = vec![Fr::zero(); 16];
        for (dst, output_index) in [
            (0usize, 0usize),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (8, 8),
            (9, 9),
            (10, 10),
            (11, 11),
            (12, 12),
            (13, 13),
        ] {
            expected_a[dst] = eq_eval(&r_output, output_index);
        }
        assert_eq!(a_weights, expected_a);

        // Input B raw [2,1,2] padded to [2,1,2], axis-1 offset 3.
        // d0=0: output indices [6, 7]
        // d0=1: output indices [14, 15]
        let expected_b = vec![
            eq_eval(&r_output, 6),
            eq_eval(&r_output, 7),
            eq_eval(&r_output, 14),
            eq_eval(&r_output, 15),
        ];
        assert_eq!(b_weights, expected_b);
    }
}
