use crate::utils::opening_access::AccOpeningAccessor;
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        split_eq_poly::SplitEqUnreducedFold,
    },
};
use atlas_sumcheck::{
    opening_proof::ProverOpeningAccumulator, sumcheck_prover::SumcheckInstanceProver,
    sumcheck_verifier::SumcheckInstanceParams, upstream_sumcheck::round_check_failed,
};
use atlas_transcript::Transcript;
use common::VirtualPoly;
pub use jolt_atlas_verifier::onnx_proof::ops::softmax_last_axis::exponentiation::mult::*;
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};

/// Prover state for softmax exp multiplication.
pub struct MultProver<F: JoltField> {
    params: MultParams<F>,
    eq: GruenSplitEqPolynomial<F>,
    // log(F * N) variables
    exp_hi: MultilinearPolynomial<F>,
    // log(F * N) variables
    exp_lo: MultilinearPolynomial<F>,
}

impl<F: JoltField> MultProver<F> {
    /// Constructor for softmax exp multiplication prover.
    pub fn initialize(exp_hi: Vec<i32>, exp_lo: Vec<i32>, params: MultParams<F>) -> Self {
        let eq = GruenSplitEqPolynomial::new(&params.r, jolt_poly::BindingOrder::LowToHigh);
        let exp_hi = MultilinearPolynomial::from(exp_hi);
        let exp_lo = MultilinearPolynomial::from(exp_lo);
        Self {
            params,
            eq,
            exp_hi,
            exp_lo,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for MultProver<F> {
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
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.node)
            .into_provider(transcript, opening_point);
        provider.append_advice(VirtualPoly::SoftmaxExpHi, self.exp_hi.final_claim());
        provider.append_advice(VirtualPoly::SoftmaxExpLo, self.exp_lo.final_claim());
    }
}

impl<F: JoltField> MultProver<F> {
    fn bind(&mut self, r: F) {
        let r_j = F::challenge_from_field(r);
        self.eq.bind(r);
        self.exp_hi.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.exp_lo.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

impl<F: JoltField> ProveRounds<F> for MultProver<F> {
    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        let Self {
            eq, exp_hi, exp_lo, ..
        } = self;
        eq.gruen_round_deg_3(
            &|g| {
                let exp_hi_0 = exp_hi.get_bound_coeff(2 * g);
                let exp_hi_inf = exp_hi.get_bound_coeff(2 * g + 1) - exp_hi_0;
                let exp_lo_0 = exp_lo.get_bound_coeff(2 * g);
                let exp_lo_inf = exp_lo.get_bound_coeff(2 * g + 1) - exp_lo_0;
                [exp_lo_0 * exp_hi_0, exp_hi_inf * exp_lo_inf]
            },
            &|g| exp_lo.get_bound_coeff(2 * g + 1) * exp_hi.get_bound_coeff(2 * g + 1),
            previous_claim,
        )
        .map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}
