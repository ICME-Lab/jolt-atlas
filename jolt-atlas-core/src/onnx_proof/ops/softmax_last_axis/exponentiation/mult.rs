use crate::onnx_proof::relation::{RelationSetup, RelationWitness};
use atlas_algebra::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        split_eq_poly::SplitEqUnreducedFold,
    },
};
use atlas_sumcheck::upstream_sumcheck::round_check_failed;
pub use jolt_atlas_verifier::onnx_proof::ops::softmax_last_axis::exponentiation::mult::*;
use jolt_claims::SymbolicSumcheck;
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};

/// Prover state for softmax exp multiplication.
pub struct MultProver<F: JoltField> {
    params: MultParams<F>,
    setup: RelationSetup<F>,
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
            setup: params.relation_setup(),
            params,
            eq,
            exp_hi,
            exp_lo,
        }
    }
}

impl<F: JoltField> RelationWitness<F> for MultProver<F> {
    type Relation = MultParams<F>;

    fn relation(&self) -> &MultParams<F> {
        &self.params
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        vec![self.exp_hi.final_claim(), self.exp_lo.final_claim()]
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
        self.params.rounds()
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
