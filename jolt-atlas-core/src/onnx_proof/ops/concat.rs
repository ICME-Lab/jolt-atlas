use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::{Concat, Operator},
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

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Concat {
    #[tracing::instrument(skip_all, name = "Concat::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let Operator::Concat(concat_op) = &node.operator else {
            panic!("Expected Concat operator")
        };

        let gamma: F = prover.transcript.challenge_scalar();

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
        validate_concat_shapes(&raw_inputs_dims, &node.output_dims, axis);
        let ctx = ConcatGammaWeightsContext::new(&raw_inputs_dims, &node.output_dims, axis, gamma);

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            Vec::with_capacity(1 + input_count);
        // TODO(#138): Implement N-to-1 reduction to constrain both claims coming from consumer nodes and this claim
        instances.push(Box::new(GammaFoldProver::initialize_output(
            node, &ctx, prover,
        )));
        for input_idx in 0..input_count {
            instances.push(Box::new(GammaFoldProver::initialize_input(
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

        let gamma: F = verifier.transcript.challenge_scalar();

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
        validate_concat_shapes(&raw_inputs_dims, &node.output_dims, axis);
        let ctx = ConcatGammaWeightsContext::new(&raw_inputs_dims, &node.output_dims, axis, gamma);

        let mut verifiers = Vec::with_capacity(1 + input_count);
        verifiers.push(GammaFoldVerifier::initialize_output(node, &ctx, verifier));
        for input_idx in 0..input_count {
            verifiers.push(GammaFoldVerifier::initialize_input(
                node, input_idx, &ctx, verifier,
            ));
        }

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

        let claim_c = verifier
            .accumulator
            .get_virtual_polynomial_opening(claim_poly_output(node), SumcheckId::RLC)
            .1;
        let expected_claim_c = (0..input_count).fold(F::zero(), |running, input_idx| {
            let input_claim = verifier
                .accumulator
                .get_virtual_polynomial_opening(claim_poly_input(node, input_idx), SumcheckId::RLC)
                .1;
            running + input_claim
        });
        if claim_c != expected_claim_c {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Concat folded-claim relation failed".to_string(),
            ));
        }

        Ok(())
    }
}

#[derive(Clone)]
/// Parameters for computing the RLC folding of a tensor polynomial.
pub struct GammaFoldParams<F: JoltField> {
    consumer_node_idx: usize,
    claim_poly: VirtualPolynomial,
    num_rounds: usize,
    _marker: core::marker::PhantomData<F>,
}

impl<F: JoltField> GammaFoldParams<F> {
    fn new(consumer_node_idx: usize, claim_poly: VirtualPolynomial, num_rounds: usize) -> Self {
        Self {
            consumer_node_idx,
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
            .get_virtual_polynomial_opening(self.claim_poly, SumcheckId::RLC)
            .1
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }
}

/// Prover state for the gamma-folding sumcheck instance.
///
/// Proves claimed_RLC = sum_i tensor(i) * gamma^i.
pub struct GammaFoldProver<F: JoltField> {
    params: GammaFoldParams<F>,
    tensor: MultilinearPolynomial<F>,
    weights: MultilinearPolynomial<F>,
}

impl<F: JoltField> GammaFoldProver<F> {
    fn initialize<T: Transcript>(
        node: &ComputationNode,
        claim_poly: VirtualPolynomial,
        input_tensor: &Tensor<i32>,
        weight_values: Vec<F>,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        let tensor = MultilinearPolynomial::from(input_tensor.padded_next_power_of_two());
        let claim = fold_poly_with_weights(&tensor, &weight_values);
        accumulator.append_virtual(
            transcript,
            claim_poly,
            SumcheckId::RLC,
            vec![].into(),
            claim,
        );

        let params = GammaFoldParams::new(node.idx, claim_poly, tensor.len().log_2());
        Self::new(params, tensor, weight_values)
    }

    fn initialize_output<T: Transcript>(
        node: &ComputationNode,
        ctx: &ConcatGammaWeightsContext<F>,
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
        input_idx: usize,
        ctx: &ConcatGammaWeightsContext<F>,
        prover: &mut Prover<F, T>,
    ) -> Self {
        let LayerData { operands, .. } = Trace::layer_data(&prover.trace, node);
        let input_tensor = operands
            .get(input_idx)
            .unwrap_or_else(|| panic!("Concat input_idx {input_idx} out of range"));

        Self::initialize(
            node,
            claim_poly_input(node, input_idx),
            input_tensor,
            ctx.gamma_weights(input_idx),
            &mut prover.accumulator,
            &mut prover.transcript,
        )
    }

    fn new(
        params: GammaFoldParams<F>,
        tensor: MultilinearPolynomial<F>,
        weight_values: Vec<F>,
    ) -> Self {
        Self {
            params,
            tensor,
            weights: MultilinearPolynomial::from(weight_values),
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
            SumcheckId::NodeExecution(self.params.consumer_node_idx),
            opening_point,
            self.tensor.final_sumcheck_claim(),
        );
    }
}

/// Verifier state for the gamma-folding sumcheck instance.
///
/// Verifies claimed_RLC = sum_i tensor(i) * gamma^i.
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
        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            claim_poly,
            SumcheckId::RLC,
            vec![].into(),
        );

        let params = GammaFoldParams::new(node.idx, claim_poly, num_elements.log_2());
        Self::new(params, weight_values)
    }

    fn initialize_output<T: Transcript>(
        node: &ComputationNode,
        ctx: &ConcatGammaWeightsContext<F>,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Self {
        let output_num_elements = node.pow2_padded_num_output_elements();

        Self::initialize(
            node,
            claim_poly_output(node),
            output_num_elements,
            ctx.output_gamma_weights(),
            verifier,
        )
    }

    fn initialize_input<T: Transcript>(
        node: &ComputationNode,
        input_idx: usize,
        ctx: &ConcatGammaWeightsContext<F>,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Self {
        let graph = &verifier.preprocessing.model.graph;
        let input_nodes = graph.get_input_nodes(node);
        let input_num_elements = input_nodes[input_idx].pow2_padded_num_output_elements();

        Self::initialize(
            node,
            claim_poly_input(node, input_idx),
            input_num_elements,
            ctx.gamma_weights(input_idx),
            verifier,
        )
    }

    fn new(params: GammaFoldParams<F>, weight_values: Vec<F>) -> Self {
        Self {
            params,
            weights: MultilinearPolynomial::from(weight_values),
        }
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
                SumcheckId::NodeExecution(self.params.consumer_node_idx),
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
            SumcheckId::NodeExecution(self.params.consumer_node_idx),
            opening_point,
        );
    }
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

/// Computes a folded claim `sum_i poly[i] * weights[i]` for an already built polynomial.
fn fold_poly_with_weights<F: JoltField>(poly: &MultilinearPolynomial<F>, weights: &[F]) -> F {
    assert_eq!(
        poly.len(),
        weights.len(),
        "weights length must match polynomial length"
    );
    (0..poly.len())
        .map(|i| poly.get_bound_coeff(i) * weights[i])
        .sum()
}

/// Builds the standard geometric gamma-powers vector `[1, gamma, gamma^2, ...]`.
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

/// Context for generating concat-aware gamma weights for each input polynomial.
struct ConcatGammaWeightsContext<F: JoltField> {
    /// Raw input dims, the source of truth for concat indexing.
    inputs_dims: Vec<Vec<usize>>,
    /// Raw output dims are used for stride computation.
    output_dims: Vec<usize>,
    /// Concat axis index.
    axis: usize,
    /// Gamma powers over raw output linear indices.
    gamma_powers: Vec<F>,
}

impl<F: JoltField> ConcatGammaWeightsContext<F> {
    fn new(raw_inputs: &[&[usize]], raw_output_dims: &[usize], axis: usize, gamma: F) -> Self {
        let raw_inputs = raw_inputs.iter().map(|dims| dims.to_vec()).collect();
        let output_len: usize = raw_output_dims.iter().product();
        let gamma_powers = gamma_powers(output_len, gamma);

        Self {
            inputs_dims: raw_inputs,
            output_dims: raw_output_dims.to_vec(),
            axis,
            gamma_powers,
        }
    }

    /// Returns the linearized gamma weights for the concat output polynomial.
    fn output_gamma_weights(&self) -> Vec<F> {
        self.gamma_powers.pad_next_power_of_two(&self.output_dims)
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
    /// Weights are built directly in unpadded layout using concat recursion,
    /// And then padded with 0s.
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

        let rank = input_dims.len();
        assert!(rank > 0, "Concat input rank must be non-zero");

        let weights_slice = self.build_weights_recursive(rank - 1, vec![F::one()], input_idx);

        debug_assert_eq!(weights_slice.len(), poly_len);

        weights_slice.pad_next_power_of_two(input_dims)
    }

    /// Recursively expands an inner weight slice to outer dimensions.
    ///
    /// At each dimension, each block is multiplied by the appropriate gamma stride,
    /// and by an axis offset when this is the concat dimension.
    fn build_weights_recursive(
        &self,
        dim: usize,
        current_slice: Vec<F>,
        input_idx: usize,
    ) -> Vec<F> {
        let input_dims = self.input_dims(input_idx);
        let dim_offset = if self.axis == dim {
            self.axis_offset(input_idx)
        } else {
            0
        };
        let output_linear_factor: usize = self.output_dims.iter().skip(dim + 1).product();

        let mut outer_slice = Vec::with_capacity(input_dims[dim] * current_slice.len());
        for i in 0..input_dims[dim] {
            let exp = (i + dim_offset) * output_linear_factor;
            assert!(
                exp < self.gamma_powers.len(),
                "Concat gamma exponent index {exp} out of range (len={})",
                self.gamma_powers.len()
            );
            let dim_factor = self.gamma_powers[exp];
            outer_slice.extend(current_slice.iter().map(|w| *w * dim_factor));
        }

        if dim == 0 {
            outer_slice
        } else {
            self.build_weights_recursive(dim - 1, outer_slice, input_idx)
        }
    }
}

fn claim_poly_output(node: &ComputationNode) -> VirtualPolynomial {
    VirtualPolynomial::NodeOutput(node.idx)
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
        utils::dims::Pad,
    };
    use joltworks::field::JoltField;
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

    fn gamma_pow<F: JoltField>(gamma: F, power: usize) -> F {
        (0..power).fold(F::one(), |acc, _| acc * gamma)
    }

    fn weights_from_exponents<F: JoltField>(gamma: F, exponents: &[usize]) -> Vec<F> {
        exponents.iter().map(|exp| gamma_pow(gamma, *exp)).collect()
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
        let gamma = Fr::from(5u64);
        // Concat inputs [2,3,2] and [2,1,2] along axis 1.
        // Raw output [2,4,2] (3+1=4, already pow2).
        // Padded input A [2,4,2], padded input B [2,1,2], padded output [2,4,2].
        let ctx =
            super::ConcatGammaWeightsContext::new(&[&[2, 3, 2], &[2, 1, 2]], &[2, 4, 2], 1, gamma);

        let output_weights = ctx.output_gamma_weights();
        let a_weights = ctx.gamma_weights(0);
        let b_weights = ctx.gamma_weights(1);

        // Output [2,4,2] is already pow2 (len=16), so all 16 entries are γ^0..γ^15.
        let expected_output = weights_from_exponents(
            gamma,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        );
        assert_eq!(output_weights, expected_output.as_slice());

        let expected_a = weights_from_exponents(gamma, &[0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])
            .pad_next_power_of_two(&[2, 3, 2]);
        assert_eq!(a_weights, expected_a);

        // Input B raw [2,1,2] padded to [2,1,2], axis-1 offset 3.
        // d0=0: exp=(0+3)*2=6 → [γ^6, γ^7]
        // d0=1: d0 * γ^8      → [γ^14, γ^15]
        let expected_b = weights_from_exponents(gamma, &[6, 7, 14, 15]);
        assert_eq!(b_weights, expected_b);
    }
}
