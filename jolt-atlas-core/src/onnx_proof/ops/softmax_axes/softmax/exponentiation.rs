use std::array;

use crate::onnx_proof::ops::softmax_axes::softmax::SoftmaxIndex;
use atlas_onnx_tracer::tensor::ops::nonlinearities::{
    SoftmaxTrace, EXP_LUT_SCALE_128, EXP_LUT_SIZE, LOG_EXP_LUT_SIZE,
};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
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
        shout::RaOneHotEncoding,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::thread::unsafe_allocate_zero_vec,
};
use rayon::prelude::*;

const READ_RAF_DEGREE_BOUND: usize = 2;

/// Shared prover/verifier parameters for Shout.
#[derive(Clone)]
pub struct ReadRafParams<F: JoltField> {
    r_exponentiation_output: Vec<F::Challenge>,
    softmax_index: SoftmaxIndex,
    gamma: F,
}

impl<F: JoltField> ReadRafParams<F> {
    /// Create new parameters for exponentiation lookups.
    pub fn new(
        softmax_index: SoftmaxIndex,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma = transcript.challenge_scalar();
        let r_exponentiation_output = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExponentiationOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
            )
            .0
            .r;
        Self {
            r_exponentiation_output,
            softmax_index,
            gamma,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ReadRafParams<F> {
    fn degree(&self) -> usize {
        READ_RAF_DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, read_checking_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::SoftmaxExponentiationOutput(
                self.softmax_index.node_idx,
                self.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
        );
        let (_, raf_checking_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::SoftmaxAbsCenteredLogitsOutput(
                self.softmax_index.node_idx,
                self.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
        );
        read_checking_claim + self.gamma * raf_checking_claim
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        LOG_EXP_LUT_SIZE
    }
}

/// Prover state for exponentiations.
pub struct ReadRafProver<F: JoltField> {
    params: ReadRafParams<F>,
    val: MultilinearPolynomial<F>,
    F: MultilinearPolynomial<F>,
    int: IdentityPolynomial<F>,
}

impl<F: JoltField> ReadRafProver<F> {
    /// Initialize the prover for exponentiation lookups.
    pub fn initialize(
        trace: &SoftmaxTrace,
        params: ReadRafParams<F>,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        // Add raf claim and implicitly sub claim
        let raf_claim = MultilinearPolynomial::from(trace.abs_centered_logits.to_vec())
            .evaluate(&params.r_exponentiation_output);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxAbsCenteredLogitsOutput(
                params.softmax_index.node_idx,
                params.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
            params.r_exponentiation_output.clone().into(),
            raf_claim,
        );

        let E = EqPolynomial::evals(&params.r_exponentiation_output);
        let lookup_indices = &trace.abs_centered_logits;
        let F = lookup_indices
            .data()
            .par_iter()
            .enumerate()
            .fold(
                || unsafe_allocate_zero_vec::<F>(EXP_LUT_SIZE),
                |mut local_F, (j, &lookup_index)| {
                    local_F[lookup_index as usize] += E[j];
                    local_F
                },
            )
            .reduce(
                || unsafe_allocate_zero_vec::<F>(EXP_LUT_SIZE),
                |mut acc, local_F| {
                    for (i, &val) in local_F.iter().enumerate() {
                        acc[i] += val;
                    }
                    acc
                },
            );
        let val = MultilinearPolynomial::from(EXP_LUT_SCALE_128.to_vec());
        Self {
            params,
            val,
            F: MultilinearPolynomial::from(F),
            int: IdentityPolynomial::new(LOG_EXP_LUT_SIZE),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ReadRafProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self { F, val, int, .. } = self;
        let half_poly_len = val.len() / 2;
        let uni_poly_evals: [F; 2] = (0..half_poly_len)
            .into_par_iter()
            .map(|i| {
                let val_evals =
                    val.sumcheck_evals(i, READ_RAF_DEGREE_BOUND, BindingOrder::HighToLow);
                let f_evals = F.sumcheck_evals(i, READ_RAF_DEGREE_BOUND, BindingOrder::HighToLow);
                let int_evals =
                    int.sumcheck_evals(i, READ_RAF_DEGREE_BOUND, BindingOrder::HighToLow);
                [
                    f_evals[0] * (val_evals[0] + self.params.gamma * int_evals[0]),
                    f_evals[1] * (val_evals[1] + self.params.gamma * int_evals[1]),
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| array::from_fn(|i| running[i] + new[i]),
            );
        UniPoly::from_evals_and_hint(previous_claim, &uni_poly_evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.val.bind_parallel(r_j, BindingOrder::HighToLow);
        self.F.bind_parallel(r_j, BindingOrder::HighToLow);
        self.int.bind_parallel(r_j, BindingOrder::HighToLow);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = [
            sumcheck_challenges,
            self.params.r_exponentiation_output.as_slice(),
        ]
        .concat();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxExponentiationRa(
                self.params.softmax_index.node_idx,
                self.params.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
            opening_point.into(),
            self.F.final_sumcheck_claim(),
        );
    }
}

/// Verifier for exponentiation lookups.
pub struct ReadRafVerifier<F: JoltField> {
    params: ReadRafParams<F>,
}

impl<F: JoltField> ReadRafVerifier<F> {
    /// Create new verifier for exponentiation lookups.
    pub fn new(
        softmax_index: SoftmaxIndex,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = ReadRafParams::new(softmax_index, accumulator, transcript);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxAbsCenteredLogitsOutput(
                params.softmax_index.node_idx,
                params.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
            params.r_exponentiation_output.clone().into(),
        );
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ReadRafVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let ra_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExponentiationRa(
                    self.params.softmax_index.node_idx,
                    self.params.softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
            )
            .1;
        let val_claim =
            MultilinearPolynomial::from(EXP_LUT_SCALE_128.to_vec()).evaluate(sumcheck_challenges);
        let int_claim = IdentityPolynomial::new(LOG_EXP_LUT_SIZE).evaluate(sumcheck_challenges);
        ra_claim * (val_claim + self.params.gamma * int_claim)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = [
            sumcheck_challenges,
            self.params.r_exponentiation_output.as_slice(),
        ]
        .concat();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxExponentiationRa(
                self.params.softmax_index.node_idx,
                self.params.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
            opening_point.into(),
        );
    }
}

// ---------------------------------------------------------------------------
// SoftmaxExpRaEncoding — implements RaOneHotEncoding for softmax exponentiation
// ---------------------------------------------------------------------------

/// Encoding for softmax exponentiation read-address one-hot checking.
pub struct SoftmaxExpRaEncoding {
    /// Feature identifier within the softmax operation.
    pub softmax_index: SoftmaxIndex,
}

impl RaOneHotEncoding for SoftmaxExpRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPolynomial {
        CommittedPolynomial::SoftmaxExponentiationRaD(
            self.softmax_index.node_idx,
            self.softmax_index.feature_idx,
            d,
        )
    }

    fn r_cycle_source(&self) -> (VirtualPolynomial, SumcheckId) {
        (
            VirtualPolynomial::SoftmaxExponentiationOutput(
                self.softmax_index.node_idx,
                self.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
        )
    }

    fn ra_source(&self) -> (VirtualPolynomial, SumcheckId) {
        (
            VirtualPolynomial::SoftmaxExponentiationRa(
                self.softmax_index.node_idx,
                self.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
        )
    }

    fn log_k(&self) -> usize {
        LOG_EXP_LUT_SIZE
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), LOG_EXP_LUT_SIZE)
    }
}
