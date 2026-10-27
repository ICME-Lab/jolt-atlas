//! The prover side of [`super`].

use super::*;
use atlas_trace::Tensor;
use common::parallel::par_enabled;
use common::VirtualPolynomial;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use atlas_algebra::{field::{IntoOpening, JoltField}, poly::{multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation}, opening_point::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN}}, utils::math::Math};
use crate::{opening_proof::{OpeningAccumulator, prover::ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator, VirtualOpeningId}, sumcheck::prover::SumcheckInstanceProver, sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier}};
use atlas_transcript::Transcript;

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
        let tensor = MultilinearPolynomial::from(tensor_values.padded_pow2());
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

impl<F: JoltField> GammaFoldProver<F> {
    fn round_message(&mut self, _round: usize, previous_claim: F) -> UnivariatePoly<F> {
        let univariate_poly_evals: [F; DEGREE_BOUND] = (0..self.tensor.len() / 2)
            .into_par_iter()
            .with_min_len(par_enabled())
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

        UnivariatePoly::from_evals_and_hint(previous_claim, &univariate_poly_evals)
    }


    fn bind(&mut self, r: F, _round: usize) {
        let r_j = F::challenge_from_field(r);
        self.tensor.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.weights.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

}

impl<F: JoltField> ProveRounds<F> for GammaFoldProver<F> {
    fn num_rounds(&self) -> usize {
        SumcheckInstanceParams::num_rounds(&self.params)
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r, round - 1);
        }
        Ok(self.round_message(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind, SumcheckInstanceParams::num_rounds(&self.params) - 1);
        Ok(())
    }
}
