use crate::onnx_proof::ops::softmax_axes::softmax::SoftmaxIndex;
use atlas_onnx_tracer::tensor::ops::nonlinearities::SoftmaxTrace;
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
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
    utils::{index_to_field_bitvector, thread::unsafe_allocate_zero_vec},
};
use rayon::prelude::*;
use std::{array, marker::PhantomData};

const DEGREE_BOUND: usize = 2;

#[derive(Clone)]
pub struct IndicatorParams {
    softmax_index: SoftmaxIndex,
    num_rounds: usize,
    max_index: usize,
}

impl IndicatorParams {
    pub fn new<F: JoltField>(
        softmax_index: SoftmaxIndex,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let f_max_index = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxMaxIndex(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
            )
            .1;
        let max_index = f_max_index
            .to_u64()
            .expect("max index should fit in 64 bits") as usize;
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
            max_index,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for IndicatorParams {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, max_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::SoftmaxMaxOutput(
                self.softmax_index.node_idx,
                self.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
        );
        max_claim
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }
}

pub struct IndicatorProver<F: JoltField> {
    params: IndicatorParams,
    softmax_operand: MultilinearPolynomial<F>,
    e: MultilinearPolynomial<F>,
}

impl<F: JoltField> IndicatorProver<F> {
    pub fn initialize(trace: &SoftmaxTrace, params: IndicatorParams) -> Self {
        let softmax_operand = MultilinearPolynomial::from(trace.input_logits.clone());
        let e = {
            let n = softmax_operand.get_num_vars();
            let mut evals = unsafe_allocate_zero_vec(1 << n);
            let max_index = params.max_index;
            evals[max_index] = F::one();
            MultilinearPolynomial::from(evals)
        };
        Self {
            params,
            softmax_operand,
            e,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for IndicatorProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            softmax_operand, e, ..
        } = self;
        let half_poly_len = softmax_operand.len() / 2;
        let uni_poly_evals: [F; 2] = (0..half_poly_len)
            .into_par_iter()
            .map(|i| {
                let so_evals =
                    softmax_operand.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                let e_evals = e.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                [so_evals[0] * e_evals[0], so_evals[1] * e_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| array::from_fn(|i| running[i] + new[i]),
            );
        UniPoly::from_evals_and_hint(previous_claim, &uni_poly_evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.softmax_operand
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.e.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            VirtualPolynomial::SoftmaxInputLogitsOutput(
                self.params.softmax_index.node_idx,
                self.params.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
            opening_point.clone(),
            self.softmax_operand.final_sumcheck_claim(),
        );
    }
}

pub struct IndicatorVerifier<F: JoltField> {
    params: IndicatorParams,
    _field: PhantomData<F>,
}

impl<F: JoltField> IndicatorVerifier<F> {
    pub fn new(softmax_index: SoftmaxIndex, accumulator: &VerifierOpeningAccumulator<F>) -> Self {
        let params = IndicatorParams::new(softmax_index, accumulator);
        Self {
            params,
            _field: PhantomData,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for IndicatorVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let (_, softmax_operand_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::SoftmaxInputLogitsOutput(
                self.params.softmax_index.node_idx,
                self.params.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
        );
        let mut y =
            index_to_field_bitvector::<F>(self.params.max_index as u64, sumcheck_challenges.len());
        y.reverse();
        softmax_operand_claim * EqPolynomial::mle(sumcheck_challenges, &y)
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
            VirtualPolynomial::SoftmaxInputLogitsOutput(
                self.params.softmax_index.node_idx,
                self.params.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
            opening_point.clone(),
        );
    }
}
