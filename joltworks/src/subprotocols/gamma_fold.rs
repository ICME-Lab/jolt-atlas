use atlas_onnx_tracer::tensor::Tensor;
use common::VirtualPolynomial;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    field::{IntoOpening, JoltField},
    poly::{
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, VirtualOpeningId, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::math::Math,
};

const DEGREE_BOUND: usize = 2;

/// Builds the standard geometric gamma-powers vector `[1, gamma, gamma^2, ...]`.
pub fn gamma_powers<F: JoltField>(poly_len: usize, gamma: F) -> Vec<F> {
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
        let id = VirtualOpeningId::new(self.claim_poly, SumcheckId::RLC(self.node_exec_idx));
        accumulator.get_virtual_polynomial_opening(id).1
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }
}

/// Prover state for gamma-folding sumcheck instances.
///
/// Proves `claimed_RLC = sum_i tensor(i) * gamma^i`.
pub struct GammaFoldProver<F: JoltField> {
    params: GammaFoldParams<F>,
    tensor: MultilinearPolynomial<F>,
    weights: MultilinearPolynomial<F>,
}

impl<F: JoltField> GammaFoldProver<F> {
    pub fn initialize<T: Transcript>(
        node_exec_idx: usize,
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
            VirtualOpeningId::new(claim_poly, SumcheckId::RLC(node_exec_idx)),
            (vec![] as Vec<F>).into(),
            claim,
        );

        let params = GammaFoldParams::new(node_exec_idx, claim_poly, tensor.len().log_2());
        Self::new(params, tensor, weights)
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
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let id = VirtualOpeningId::new(
            self.params.claim_poly,
            SumcheckId::NodeExecution(self.params.node_exec_idx),
        );
        accumulator.append_virtual(
            transcript,
            id,
            opening_point,
            self.tensor.final_sumcheck_claim(),
        );
    }
}

/// Verifier state for gamma-folding sumcheck instances.
///
/// Verifies `claimed_RLC = sum_i tensor(i) * gamma^i`.
pub struct GammaFoldVerifier<F: JoltField> {
    params: GammaFoldParams<F>,
    weights: MultilinearPolynomial<F>,
}

impl<F: JoltField> GammaFoldVerifier<F> {
    pub fn initialize<T: Transcript>(
        node_exec_idx: usize,
        claim_poly: VirtualPolynomial,
        num_elements: usize,
        weight_values: Vec<F>,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        let weights = MultilinearPolynomial::from(weight_values);
        accumulator.append_virtual(
            transcript,
            VirtualOpeningId::new(claim_poly, SumcheckId::RLC(node_exec_idx)),
            (vec![] as Vec<F>).into(),
        );

        let params = GammaFoldParams::new(node_exec_idx, claim_poly, num_elements.log_2());
        Self::new(params, weights)
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
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let weight_eval = self.weights.evaluate(&opening_point.r);
        let id = VirtualOpeningId::new(
            self.params.claim_poly,
            SumcheckId::NodeExecution(self.params.node_exec_idx),
        );
        let tensor_claim = accumulator.get_virtual_polynomial_opening(id).1;
        weight_eval * tensor_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let id = VirtualOpeningId::new(
            self.params.claim_poly,
            SumcheckId::NodeExecution(self.params.node_exec_idx),
        );
        accumulator.append_virtual(transcript, id, opening_point);
    }
}
