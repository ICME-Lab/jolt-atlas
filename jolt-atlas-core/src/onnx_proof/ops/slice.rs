use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::{Operator, Slice},
    tensor::Tensor,
    utils::dims::Pad,
};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::onnx_proof::{ops::OperatorProofTrait, ProofId, ProofType, Prover, Verifier};

const DEGREE_BOUND: usize = 2;

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Slice {
    #[tracing::instrument(skip_all, name = "Slice::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let Operator::Slice(slice_op) = &node.operator else {
            panic!("Expected Slice operator")
        };

        let gamma: F = prover.transcript.challenge_scalar();

        let input_dims = {
            let LayerData { operands, .. } = Trace::layer_data(&prover.trace, node);
            let [input_tensor] = operands[..] else {
                panic!("Slice expects exactly one operand")
            };
            input_tensor.dims().to_vec()
        };

        let ctx =
            SliceGammaWeightsContext::new(&input_dims, &node.output_dims, slice_op.clone(), gamma);

        // TODO(#138): Implement N-to-1 reduction to constrain both claims coming from consumer nodes and this claim
        let output_prover = GammaFoldProver::initialize_output(node, &ctx, prover);
        let input_prover = GammaFoldProver::initialize_input(node, &ctx, prover);

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![Box::new(output_prover), Box::new(input_prover)];

        let (proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );

        vec![(ProofId(node.idx, ProofType::Execution), proof)]
    }

    #[tracing::instrument(skip_all, name = "Slice::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;

        let Operator::Slice(slice_op) = &node.operator else {
            panic!("Expected Slice operator")
        };

        let gamma: F = verifier.transcript.challenge_scalar();

        let input_dims = {
            let graph = &verifier.preprocessing.model.graph;
            let input_nodes = graph.get_input_nodes(node);
            let [input_node] = input_nodes[..] else {
                panic!("Slice expects exactly one input node")
            };
            input_node.output_dims.clone()
        };

        let ctx =
            SliceGammaWeightsContext::new(&input_dims, &node.output_dims, slice_op.clone(), gamma);

        let verifiers: Vec<GammaFoldVerifier<F>> = vec![
            GammaFoldVerifier::initialize_output(node, &ctx, verifier),
            GammaFoldVerifier::initialize_input(node, &ctx, verifier),
        ];
        let instances: Vec<&dyn SumcheckInstanceVerifier<F, T>> = verifiers
            .iter()
            .map(|instance| instance as &dyn SumcheckInstanceVerifier<F, T>)
            .collect();
        BatchedSumcheck::verify(
            proof,
            instances,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        let output_claim = verifier
            .accumulator
            .get_virtual_polynomial_opening(claim_poly_output(node), SumcheckId::RLC(node.idx))
            .1;
        let input_claim = verifier
            .accumulator
            .get_virtual_polynomial_opening(claim_poly_input(node), SumcheckId::RLC(node.idx))
            .1;
        if output_claim != input_claim {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Slice folded-claim relation failed".to_string(),
            ));
        }

        Ok(())
    }
}

#[derive(Clone)]
/// Parameters for proving a gamma-folded claim over a tensor polynomial.
pub struct GammaFoldParams<F: JoltField> {
    node_exec_idx: usize,
    claim_poly: VirtualPolynomial,
    num_rounds: usize,
    _marker: core::marker::PhantomData<F>,
}

impl<F: JoltField> GammaFoldParams<F> {
    fn new(node_exec_idx: usize, claim_poly: VirtualPolynomial, num_rounds: usize) -> Self {
        Self {
            node_exec_idx,
            claim_poly,
            num_rounds,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for GammaFoldParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        accumulator
            .get_virtual_polynomial_opening(self.claim_poly, SumcheckId::RLC(self.node_exec_idx))
            .1
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }
}

/// Prover state for the Slice gamma-folding sumcheck instances.
pub struct GammaFoldProver<F: JoltField> {
    params: GammaFoldParams<F>,
    tensor: MultilinearPolynomial<F>,
    weights: MultilinearPolynomial<F>,
}

impl<F: JoltField> GammaFoldProver<F> {
    fn initialize<T: Transcript>(
        node: &ComputationNode,
        claim_poly: VirtualPolynomial,
        tensor_values: &Tensor<i32>,
        weight_values: Vec<F>,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        let tensor = MultilinearPolynomial::from(tensor_values.padded_next_power_of_two());
        let claim = tensor.dot_product(&weight_values);
        let weights = MultilinearPolynomial::from(weight_values);
        accumulator.append_virtual(
            transcript,
            claim_poly,
            SumcheckId::RLC(node.idx),
            vec![].into(),
            claim,
        );

        let params = GammaFoldParams::new(node.idx, claim_poly, tensor.len().log_2());
        Self::new(params, tensor, weights)
    }

    fn initialize_output<T: Transcript>(
        node: &ComputationNode,
        ctx: &SliceGammaWeightsContext<F>,
        prover: &mut Prover<F, T>,
    ) -> Self {
        let LayerData { output, .. } = Trace::layer_data(&prover.trace, node);

        Self::initialize(
            node,
            claim_poly_output(node),
            output,
            ctx.output_gamma_weights(),
            &mut prover.accumulator,
            &mut prover.transcript,
        )
    }

    fn initialize_input<T: Transcript>(
        node: &ComputationNode,
        ctx: &SliceGammaWeightsContext<F>,
        prover: &mut Prover<F, T>,
    ) -> Self {
        let LayerData { operands, .. } = Trace::layer_data(&prover.trace, node);
        let [input_tensor] = operands[..] else {
            panic!("Slice expects exactly one operand")
        };

        Self::initialize(
            node,
            claim_poly_input(node),
            input_tensor,
            ctx.input_gamma_weights(),
            &mut prover.accumulator,
            &mut prover.transcript,
        )
    }

    fn new(
        params: GammaFoldParams<F>,
        tensor: MultilinearPolynomial<F>,
        weights: MultilinearPolynomial<F>,
    ) -> Self {
        Self {
            params,
            tensor,
            weights,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for GammaFoldProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let univariate_poly_evals: [F; DEGREE_BOUND] = (0..self.tensor.len() / 2)
            .into_par_iter()
            .map(|i| {
                let tensor_evals =
                    self.tensor
                        .sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                let weight_evals =
                    self.weights
                        .sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                [
                    tensor_evals[0] * weight_evals[0],
                    tensor_evals[1] * weight_evals[1],
                ]
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        UniPoly::from_evals_and_hint(previous_claim, &univariate_poly_evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.tensor.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.weights.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            self.params.claim_poly,
            SumcheckId::NodeExecution(self.params.node_exec_idx),
            opening_point,
            self.tensor.final_sumcheck_claim(),
        );
    }
}

/// Verifier state for the Slice gamma-folding sumcheck instances.
pub struct GammaFoldVerifier<F: JoltField> {
    params: GammaFoldParams<F>,
    weights: MultilinearPolynomial<F>,
}

impl<F: JoltField> GammaFoldVerifier<F> {
    fn initialize<T: Transcript>(
        node: &ComputationNode,
        claim_poly: VirtualPolynomial,
        num_elements: usize,
        weight_values: Vec<F>,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Self {
        let weights = MultilinearPolynomial::from(weight_values);
        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            claim_poly,
            SumcheckId::RLC(node.idx),
            vec![].into(),
        );

        let params = GammaFoldParams::new(node.idx, claim_poly, num_elements.log_2());
        Self::new(params, weights)
    }

    fn initialize_output<T: Transcript>(
        node: &ComputationNode,
        ctx: &SliceGammaWeightsContext<F>,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Self {
        Self::initialize(
            node,
            claim_poly_output(node),
            node.pow2_padded_num_output_elements(),
            ctx.output_gamma_weights(),
            verifier,
        )
    }

    fn initialize_input<T: Transcript>(
        node: &ComputationNode,
        ctx: &SliceGammaWeightsContext<F>,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Self {
        let graph = &verifier.preprocessing.model.graph;
        let input_nodes = graph.get_input_nodes(node);
        let [input_node] = input_nodes[..] else {
            panic!("Slice expects exactly one input node")
        };

        Self::initialize(
            node,
            claim_poly_input(node),
            input_node.pow2_padded_num_output_elements(),
            ctx.input_gamma_weights(),
            verifier,
        )
    }

    fn new(params: GammaFoldParams<F>, weights: MultilinearPolynomial<F>) -> Self {
        Self { params, weights }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for GammaFoldVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let weight_eval = self.weights.evaluate(&opening_point.r);
        let tensor_claim = accumulator
            .get_virtual_polynomial_opening(
                self.params.claim_poly,
                SumcheckId::NodeExecution(self.params.node_exec_idx),
            )
            .1;
        weight_eval * tensor_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            self.params.claim_poly,
            SumcheckId::NodeExecution(self.params.node_exec_idx),
            opening_point,
        );
    }
}

fn validate_slice_shapes(input_dims: &[usize], output_dims: &[usize], op: &Slice) {
    assert_eq!(input_dims.len(), output_dims.len(), "Slice rank mismatch");
    assert!(op.axis < input_dims.len(), "Slice axis out of range");
    assert!(op.start <= op.end, "Slice start must be <= end");
    assert!(op.end <= input_dims[op.axis], "Slice end out of bounds");

    for dim in 0..input_dims.len() {
        let expected = if dim == op.axis {
            op.end - op.start
        } else {
            input_dims[dim]
        };
        assert_eq!(
            output_dims[dim], expected,
            "Slice output shape mismatch at axis {dim}"
        );
    }
}

fn gamma_powers<F: JoltField>(poly_len: usize, gamma: F) -> Vec<F> {
    if poly_len == 0 {
        return Vec::new();
    }

    let mut weights = vec![F::zero(); poly_len];
    weights[0] = F::one();
    for i in 1..poly_len {
        weights[i] = weights[i - 1] * gamma;
    }
    weights
}

struct SliceGammaWeightsContext<F: JoltField> {
    input_dims: Vec<usize>,
    output_dims: Vec<usize>,
    op: Slice,
    gamma_powers: Vec<F>,
}

impl<F: JoltField> SliceGammaWeightsContext<F> {
    fn new(input_dims: &[usize], output_dims: &[usize], op: Slice, gamma: F) -> Self {
        validate_slice_shapes(input_dims, output_dims, &op);

        let output_len: usize = output_dims.iter().product();
        Self {
            input_dims: input_dims.to_vec(),
            output_dims: output_dims.to_vec(),
            op,
            gamma_powers: gamma_powers(output_len, gamma),
        }
    }

    fn output_gamma_weights(&self) -> Vec<F> {
        self.gamma_powers.pad_next_power_of_two(&self.output_dims)
    }

    fn input_gamma_weights(&self) -> Vec<F> {
        let rank = self.input_dims.len();
        assert!(rank > 0, "Slice rank must be non-zero");

        let raw_weights = self.build_input_weights_recursive(rank - 1, vec![F::one()]);
        debug_assert_eq!(raw_weights.len(), self.input_dims.iter().product::<usize>());
        raw_weights.pad_next_power_of_two(&self.input_dims)
    }

    /// Recursively expands gamma factors from inner to outer dimensions.
    ///
    /// This mirrors concat-style weight construction and keeps all indexing in
    /// slice-native terms: at the sliced axis we only emit non-zero factors for
    /// positions in `[start, start + output_axis_dim)` and zero otherwise.
    fn build_input_weights_recursive(&self, dim: usize, current_slice: Vec<F>) -> Vec<F> {
        let output_linear_factor: usize = self.output_dims.iter().skip(dim + 1).product();
        let mut outer_slice = Vec::with_capacity(self.input_dims[dim] * current_slice.len());

        if dim == self.op.axis {
            outer_slice.extend((0..current_slice.len() * self.op.start).map(|_| F::zero()));
        }

        for i in 0..self.output_dims[dim] {
            let exp = i * output_linear_factor;
            let dim_factor = self.gamma_powers[exp];
            outer_slice.extend(current_slice.iter().map(|w| *w * dim_factor));
        }

        if dim == self.op.axis {
            outer_slice.resize(self.input_dims[dim] * current_slice.len(), F::zero());
        }

        if dim == 0 {
            outer_slice
        } else {
            self.build_input_weights_recursive(dim - 1, outer_slice)
        }
    }
}

fn claim_poly_output(node: &ComputationNode) -> VirtualPolynomial {
    VirtualPolynomial::NodeOutput(node.idx)
}

fn claim_poly_input(node: &ComputationNode) -> VirtualPolynomial {
    let input_node_idx = *node
        .inputs
        .first()
        .expect("Slice node should have one input");
    VirtualPolynomial::NodeOutput(input_node_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_proof::ops::test::unit_test_op;
    use ark_bn254::Fr;
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, Model},
        ops::Slice,
        tensor::{ops::concat, Tensor},
        utils::dims::Pad,
    };
    use joltworks::field::JoltField;
    use rand::{rngs::StdRng, SeedableRng};

    fn slice_model(input_shape: Vec<usize>, axis: usize, start: usize, end: usize) -> Model {
        let mut b = ModelBuilder::new();
        let input = b.input(input_shape);
        let out = b.slice(input, axis, start, end);
        b.mark_output(out);
        b.build()
    }

    fn gamma_pow<F: JoltField>(gamma: F, power: usize) -> F {
        (0..power).fold(F::one(), |acc, _| acc * gamma)
    }

    fn weights_from_exponents<F: JoltField>(gamma: F, exponents: &[usize]) -> Vec<F> {
        exponents.iter().map(|exp| gamma_pow(gamma, *exp)).collect()
    }

    fn exponents_tensor_to_weights(gamma: Fr, exponents: &Tensor<i32>) -> Vec<Fr> {
        exponents
            .data()
            .iter()
            .map(|exp| {
                if *exp < 0 {
                    Fr::from(0u64)
                } else {
                    gamma_pow(gamma, *exp as usize)
                }
            })
            .collect()
    }

    fn output_exponents_2d(dims: &[usize; 2]) -> Tensor<i32> {
        let len = dims[0] * dims[1];
        let data: Vec<i32> = (0..len as i32).collect();
        Tensor::construct(data, vec![dims[0], dims[1]])
    }

    /// Builds expected 2D input-weight exponents using concat of:
    /// zero block(s) + active output exponents block + zero block(s).
    ///
    /// Sentinel `-1` means zero weight.
    fn expected_input_exponents_with_concat_2d(input_dims: &[usize; 2], op: &Slice) -> Tensor<i32> {
        let output_dims = [input_dims[0], input_dims[1]];
        let output_dims = if op.axis == 0 {
            [op.end - op.start, output_dims[1]]
        } else {
            [output_dims[0], op.end - op.start]
        };
        let active = output_exponents_2d(&output_dims);

        if op.axis == 1 {
            let mut blocks: Vec<Tensor<i32>> = Vec::new();
            if op.start > 0 {
                blocks.push(Tensor::construct(
                    vec![-1; input_dims[0] * op.start],
                    vec![input_dims[0], op.start],
                ));
            }
            blocks.push(active);
            let right = input_dims[1] - op.end;
            if right > 0 {
                blocks.push(Tensor::construct(
                    vec![-1; input_dims[0] * right],
                    vec![input_dims[0], right],
                ));
            }
            let block_refs: Vec<&Tensor<i32>> = blocks.iter().collect();
            concat(&block_refs, 1).expect("concat on axis 1 should build expected exponents")
        } else {
            let mut blocks: Vec<Tensor<i32>> = Vec::new();
            if op.start > 0 {
                blocks.push(Tensor::construct(
                    vec![-1; op.start * input_dims[1]],
                    vec![op.start, input_dims[1]],
                ));
            }
            blocks.push(active);
            let bottom = input_dims[0] - op.end;
            if bottom > 0 {
                blocks.push(Tensor::construct(
                    vec![-1; bottom * input_dims[1]],
                    vec![bottom, input_dims[1]],
                ));
            }
            let block_refs: Vec<&Tensor<i32>> = blocks.iter().collect();
            concat(&block_refs, 0).expect("concat on axis 0 should build expected exponents")
        }
    }

    #[test]
    fn test_slice_arbitrary_shapes() {
        let cases = vec![
            (vec![2, 5], 1, 1, 4),
            (vec![3, 4], 0, 1, 3),
            (vec![2, 3, 5], 2, 0, 4),
            (vec![2, 4, 3], 1, 1, 4),
            (vec![4, 6], 1, 2, 5),
            (vec![5, 3], 0, 0, 2),
        ];

        for (idx, (shape, axis, start, end)) in cases.into_iter().enumerate() {
            let mut rng = StdRng::seed_from_u64(0x511CE + idx as u64);
            let input = Tensor::<i32>::random_small(&mut rng, &shape);
            let model = slice_model(shape, axis, start, end);
            unit_test_op(model, &[input]);
        }
    }

    #[test]
    fn test_slice_gamma_weights_layout_rank2_axis1() {
        let gamma = Fr::from(7u64);
        let cases = vec![
            (
                [2, 5],
                Slice {
                    axis: 1,
                    start: 1,
                    end: 4,
                },
            ),
            (
                [3, 6],
                Slice {
                    axis: 1,
                    start: 2,
                    end: 5,
                },
            ),
            (
                [5, 4],
                Slice {
                    axis: 0,
                    start: 1,
                    end: 4,
                },
            ),
            (
                [6, 3],
                Slice {
                    axis: 0,
                    start: 0,
                    end: 2,
                },
            ),
        ];

        for (input_dims, op) in cases {
            let output_dims = if op.axis == 0 {
                vec![op.end - op.start, input_dims[1]]
            } else {
                vec![input_dims[0], op.end - op.start]
            };

            let ctx = SliceGammaWeightsContext::new(&input_dims, &output_dims, op.clone(), gamma);
            let output_weights = ctx.output_gamma_weights();
            let input_weights = ctx.input_gamma_weights();

            let output_len = output_dims.iter().product::<usize>();
            let expected_output =
                weights_from_exponents(gamma, &(0..output_len).collect::<Vec<_>>())
                    .pad_next_power_of_two(&output_dims);
            assert_eq!(output_weights, expected_output);

            let expected_input_exponents =
                expected_input_exponents_with_concat_2d(&input_dims, &op);
            let expected_input = exponents_tensor_to_weights(gamma, &expected_input_exponents)
                .pad_next_power_of_two(&input_dims);
            assert_eq!(input_weights, expected_input);
        }
    }
}
