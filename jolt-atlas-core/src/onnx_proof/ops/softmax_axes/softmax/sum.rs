use crate::onnx_proof::ops::softmax_axes::softmax::SoftmaxIndex;
use atlas_onnx_tracer::tensor::ops::nonlinearities::SoftmaxTrace;
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
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
};
use rayon::prelude::*;
use std::marker::PhantomData;

const DEGREE_BOUND: usize = 1;

/// Parameters for proving sum computation in softmax.
#[derive(Clone)]
pub struct SumParams {
    softmax_index: SoftmaxIndex,
    num_rounds: usize,
}

impl SumParams {
    /// Create new parameters for sum computation.
    pub fn new<F: JoltField>(
        softmax_index: SoftmaxIndex,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        Self {
            softmax_index,
            num_rounds: accumulator
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::SoftmaxFeatureOutput(
                        softmax_index.node_idx,
                        softmax_index.feature_idx,
                    ),
                    SumcheckId::Execution,
                )
                .0
                .r
                .len(),
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for SumParams {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, sum_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::SoftmaxSumOutput(
                self.softmax_index.node_idx,
                self.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
        );
        sum_claim
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }
}

/// Prover state for sum computation in softmax.
pub struct SumProver<F: JoltField> {
    params: SumParams,
    operand: MultilinearPolynomial<F>,
}

impl<F: JoltField> SumProver<F> {
    /// Initialize the prover for sum computation.
    pub fn initialize(trace: &SoftmaxTrace, params: SumParams) -> Self {
        let operand = MultilinearPolynomial::from(trace.exp_q_values.clone());
        Self { params, operand }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for SumProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self { operand, .. } = self;
        let half_poly_len = operand.len() / 2;
        let eval_0 = (0..half_poly_len)
            .into_par_iter()
            .map(|i| {
                let evals = operand.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                evals[0]
            })
            .reduce(|| F::zero(), |running, new| running + new);
        UniPoly::from_evals_and_hint(previous_claim, &[eval_0])
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.operand.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            VirtualPolynomial::SoftmaxExponentiationOutput(
                self.params.softmax_index.node_idx,
                self.params.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
            opening_point.clone(),
            self.operand.final_sumcheck_claim(),
        );
    }
}

/// Verifier for sum computation in softmax.
pub struct SumVerifier<F: JoltField> {
    params: SumParams,
    _field: PhantomData<F>,
}

impl<F: JoltField> SumVerifier<F> {
    /// Create new verifier for sum computation.
    pub fn new(softmax_index: SoftmaxIndex, accumulator: &VerifierOpeningAccumulator<F>) -> Self {
        let params = SumParams::new(softmax_index, accumulator);
        Self {
            params,
            _field: PhantomData,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for SumVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExponentiationOutput(
                    self.params.softmax_index.node_idx,
                    self.params.softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
            )
            .1
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
            VirtualPolynomial::SoftmaxExponentiationOutput(
                self.params.softmax_index.node_idx,
                self.params.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
            opening_point.clone(),
        );
    }
}
