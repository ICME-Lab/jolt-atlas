use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::{Concat, Operator},
    utils::dims::UsizeDimsExt,
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

        let LayerData { operands, output } = Trace::layer_data(&prover.trace, node);
        let [a_tensor, b_tensor] = operands[..] else {
            panic!(
                "Concat currently supports exactly 2 operands (N-input support is work in progress)"
            )
        };
        let gamma: F = prover.transcript.challenge_scalar();

        let axis = normalize_axis(concat_op.axis, output.dims().len());
        let raw_inputs_dims: Vec<&[usize]> = vec![a_tensor.dims(), b_tensor.dims()];
        validate_concat_shapes(&raw_inputs_dims, output.dims(), axis);
        let padded_output_dims = output.dims().map_next_power_of_two();

        let ctx =
            ConcatGammaWeightsContext::new(&raw_inputs_dims, &padded_output_dims, axis, gamma);
        let output_weights = ctx.output_gamma_weights().to_vec();
        let poly_c = MultilinearPolynomial::from(output.padded_next_power_of_two());
        let claim_c = fold_poly_with_weights(&poly_c, &output_weights);

        let a_weights = ctx.gamma_weights(0);
        let poly_a = MultilinearPolynomial::from(a_tensor.padded_next_power_of_two());
        let claim_a = fold_poly_with_weights(&poly_a, &a_weights);

        let b_weights = ctx.gamma_weights(1);
        let poly_b = MultilinearPolynomial::from(b_tensor.padded_next_power_of_two());
        let claim_b = fold_poly_with_weights(&poly_b, &b_weights);

        prover.accumulator.append_virtual(
            &mut prover.transcript,
            claim_poly_c(node),
            SumcheckId::RLC,
            vec![].into(),
            claim_c,
        );
        prover.accumulator.append_virtual(
            &mut prover.transcript,
            claim_poly_a(node),
            SumcheckId::RLC,
            vec![].into(),
            claim_a,
        );
        prover.accumulator.append_virtual(
            &mut prover.transcript,
            claim_poly_b(node),
            SumcheckId::RLC,
            vec![].into(),
            claim_b,
        );

        let c_prover = GammaFoldProver::new(
            GammaFoldParams::new(node.idx, claim_poly_c(node), poly_c.len().log_2()),
            poly_c,
            output_weights,
        );
        let a_prover = GammaFoldProver::new(
            GammaFoldParams::new(node.idx, claim_poly_a(node), poly_a.len().log_2()),
            poly_a,
            a_weights,
        );
        let b_prover = GammaFoldProver::new(
            GammaFoldParams::new(node.idx, claim_poly_b(node), poly_b.len().log_2()),
            poly_b,
            b_weights,
        );

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![Box::new(c_prover), Box::new(a_prover), Box::new(b_prover)];
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

        let graph = &verifier.preprocessing.model.graph;
        let [a_node, b_node] = graph.get_input_nodes(node)[..] else {
            panic!("Concat currently supports exactly 2 operands")
        };

        let gamma: F = verifier.transcript.challenge_scalar();

        let axis = normalize_axis(concat_op.axis, node.output_dims.len());
        let raw_inputs_dims: Vec<&[usize]> = vec![&a_node.output_dims, &b_node.output_dims];
        validate_concat_shapes(&raw_inputs_dims, &node.output_dims, axis);
        let padded_output = node.pow2_padded_output_dims();

        let ctx = ConcatGammaWeightsContext::new(&raw_inputs_dims, &padded_output, axis, gamma);

        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            claim_poly_c(node),
            SumcheckId::RLC,
            vec![].into(),
        );
        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            claim_poly_a(node),
            SumcheckId::RLC,
            vec![].into(),
        );
        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            claim_poly_b(node),
            SumcheckId::RLC,
            vec![].into(),
        );

        let len_a: usize = a_node.pow2_padded_num_output_elements();
        let len_b: usize = b_node.pow2_padded_num_output_elements();
        let len_c: usize = node.pow2_padded_num_output_elements();

        let c_verifier = GammaFoldVerifier::new(
            GammaFoldParams::new(node.idx, claim_poly_c(node), len_c.log_2()),
            ctx.output_gamma_weights().to_vec(),
        );
        let a_verifier = GammaFoldVerifier::new(
            GammaFoldParams::new(node.idx, claim_poly_a(node), len_a.log_2()),
            ctx.gamma_weights(0),
        );
        let b_verifier = GammaFoldVerifier::new(
            GammaFoldParams::new(node.idx, claim_poly_b(node), len_b.log_2()),
            ctx.gamma_weights(1),
        );

        let instances: Vec<&dyn SumcheckInstanceVerifier<F, T>> =
            vec![&c_verifier, &a_verifier, &b_verifier];
        BatchedSumcheck::verify(
            proof,
            instances,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        let claim_c = verifier
            .accumulator
            .get_virtual_polynomial_opening(claim_poly_c(node), SumcheckId::RLC)
            .1;
        let claim_a = verifier
            .accumulator
            .get_virtual_polynomial_opening(claim_poly_a(node), SumcheckId::RLC)
            .1;
        let claim_b = verifier
            .accumulator
            .get_virtual_polynomial_opening(claim_poly_b(node), SumcheckId::RLC)
            .1;

        let expected_claim_c = claim_a + claim_b;
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
struct ConcatGammaWeightsContext<'a, F: JoltField> {
    /// Padded input dims, used for weight layout.
    padded_inputs: Vec<Vec<usize>>,
    /// Padded output dims, used for stride computation.
    padded_output_dims: &'a [usize],
    axis: usize,
    /// Raw axis dimension per input, used to compute correct axis offsets.
    dims_at_axis: Vec<usize>,
    gamma_powers: Vec<F>,
}

impl<'a, F: JoltField> ConcatGammaWeightsContext<'a, F> {
    fn new(
        raw_inputs: &[&[usize]],
        padded_output_dims: &'a [usize],
        axis: usize,
        gamma: F,
    ) -> Self {
        let padded_inputs = raw_inputs
            .iter()
            .map(|&dims| dims.map_next_power_of_two())
            .collect();
        let dims_at_axis: Vec<usize> = raw_inputs.iter().map(|dims| dims[axis]).collect();
        let output_len: usize = padded_output_dims.iter().product();
        let gamma_powers = gamma_powers(output_len, gamma);

        Self {
            padded_inputs,
            padded_output_dims,
            axis,
            dims_at_axis,
            gamma_powers,
        }
    }

    /// Returns the linearized gamma weights for the concat output polynomial.
    fn output_gamma_weights(&self) -> &[F] {
        &self.gamma_powers
    }

    fn input_dims(&self, input_idx: usize) -> &[usize] {
        self.padded_inputs
            .get(input_idx)
            .unwrap_or_else(|| panic!("Concat input_idx {input_idx} out of range"))
    }

    /// Returns the axis offset (in output coordinates) for the given input.
    ///
    /// Uses raw axis dims to avoid inflated offsets from padding.
    fn axis_offset(&self, input_idx: usize) -> usize {
        self.dims_at_axis[..input_idx].iter().sum()
    }

    /// Builds concat-aware gamma weights for one input tensor.
    ///
    /// The produced vector aligns with row-major linearization of the (possibly padded) input.
    /// The exponent for each index in the given input corresponds to its position in the concatenated output
    ///
    /// Example
    /// ```ignore
    /// // For concat of [2,2] and [2,1] on axis 1.
    /// // Position of input 1, index [0,0] in output is [0, 2]
    /// // i.e. 2 in row-major order, so it's weight is gamma^2.
    /// // Complete weight vector for input 1 is [gamma^2, gamma^5]
    /// ```
    ///
    /// ### Note:
    /// For non-power of two tensors, the padded input is used for layout, so padding tensor elements will be mapped to a non-zero weight.
    /// However we should ensure that every node's output has all padding value set to 0,
    /// So the non-zero weights are effectively ignored in the proof.
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
        weights_slice
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
        let output_linear_factor: usize = self.padded_output_dims.iter().skip(dim + 1).product();

        let mut outer_slice = Vec::with_capacity(input_dims[dim] * current_slice.len());
        for i in 0..input_dims[dim] {
            let exp = (i + dim_offset) * output_linear_factor;
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

fn claim_poly_c(node: &ComputationNode) -> VirtualPolynomial {
    VirtualPolynomial::NodeOutput(node.idx)
}

fn claim_poly_a(node: &ComputationNode) -> VirtualPolynomial {
    VirtualPolynomial::NodeOutput(node.inputs[0])
}

fn claim_poly_b(node: &ComputationNode) -> VirtualPolynomial {
    VirtualPolynomial::NodeOutput(node.inputs[1])
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use ark_bn254::Fr;
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, Model},
        tensor::Tensor,
    };
    use joltworks::field::JoltField;
    use rand::{rngs::StdRng, SeedableRng};

    fn concat_model(a_shape: Vec<usize>, b_shape: Vec<usize>, axis: isize) -> Model {
        let mut b = ModelBuilder::new();
        let a = b.input(a_shape);
        let c = b.input(b_shape);
        let out = b.concat(a, c, axis);
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
            let model = concat_model(a_shape, b_shape, axis);
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
            let model = concat_model(a_shape, b_shape, axis);
            unit_test_op(model, &[a, b]);
        }
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

        // Input A raw [2,3,2] padded to [2,4,2], axis-1 offset 0.
        // All 4 padded axis-1 slots get sequential output exponents (offset=0 so no shift).
        // d0=0: [γ^0..γ^7], d0=1: [γ^8..γ^15]
        // Note: since inputs values at index 6, 7, 14, 15 are padding and equal 0,
        // the weight 2^6, 2^7, 2^14, 2^15 are multiplied by 0 and disappear.
        let expected_a = weights_from_exponents(
            gamma,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        );
        assert_eq!(a_weights, expected_a);

        // Input B raw [2,1,2] padded to [2,1,2], axis-1 offset 3.
        // d0=0: exp=(0+3)*2=6 → [γ^6, γ^7]
        // d0=1: d0 * γ^8      → [γ^14, γ^15]
        let expected_b = weights_from_exponents(gamma, &[6, 7, 14, 15]);
        assert_eq!(b_weights, expected_b);
    }
}
