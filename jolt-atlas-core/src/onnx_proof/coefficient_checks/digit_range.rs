//! Base-four digit zero check from Atlas's original digit range sumcheck.

use super::*;
use allocative::Allocative;
use atlas_algebra::field::JoltField;
use atlas_algebra::poly::{
    multilinear::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
    split_eq::{GruenSplitEqPolynomial, SplitEqUnreducedFold},
    univariate::UnivariatePoly,
};
use atlas_claims::SymbolicSumcheck;
use atlas_sumcheck::{
    engine::{prover::round_check_failed, prover::ProveRounds, SumcheckError},
    relation::{prover::RelationWitness, RelationChallenge, RelationSetup},
};
use common::parallel::par_enabled;
use rayon::prelude::*;

#[derive(Allocative)]
pub struct DigitRangeProver<F: JoltField> {
    #[allocative(skip)]
    check: CoefficientCheck,
    #[allocative(skip)]
    setup: RelationSetup<F>,
    #[allocative(skip)]
    eq: GruenSplitEqPolynomial<F>,
    digits: Vec<MultilinearPolynomial<F>>,
    weights: Vec<F>,
}

impl<F: JoltField> DigitRangeProver<F> {
    /// Build the base-four check over the original dense digit columns.
    pub fn new(
        check: CoefficientCheck,
        setup: RelationSetup<F>,
        digits: Vec<MultilinearPolynomial<F>>,
    ) -> Result<Self, ProofVerifyError> {
        if check.one_hot_address_vars().is_some()
            || setup.incoming_point.len() != check.rounds()
            || digits.len() != check.polynomials().len()
            || digits.iter().any(|poly| {
                matches!(poly, MultilinearPolynomial::OneHot(_))
                    || poly.get_num_vars() != check.rounds()
                    || poly.is_bound()
            })
        {
            return Err(invalid("wrong digit witness shape for range check"));
        }
        let eq = GruenSplitEqPolynomial::new(&setup.incoming_point, BindingOrder::LowToHigh);
        let weights = (0..digits.len())
            .map(|i| setup.challenge(&RelationChallenge::Batching(i)))
            .collect();
        Ok(Self {
            check,
            setup,
            eq,
            digits,
            weights,
        })
    }

    fn bind(&mut self, challenge: F) {
        let challenge_field = F::challenge_from_field(challenge);
        self.digits
            .par_iter_mut()
            .with_min_len(par_enabled())
            .for_each(|poly| {
                poly.bind_parallel(challenge_field, BindingOrder::LowToHigh);
            });
        self.eq.bind(challenge);
    }

    fn residual(value: F) -> F {
        (1..4).fold(value, |product, root| product * (value - F::from_u64(root)))
    }
}

impl<F: JoltField> ProveRounds<F> for DigitRangeProver<F> {
    fn num_rounds(&self) -> usize {
        self.check.rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        self.eq
            .gruen_round_from_evals::<4>(
                &|index| {
                    let mut samples = [F::zero(); 4];
                    for (digit, &gamma) in self.digits.iter().zip(&self.weights) {
                        let low = digit.get_bound_coeff(2 * index);
                        let mut value = digit.get_bound_coeff(2 * index + 1);
                        let delta = value - low;
                        for sample in &mut samples[..3] {
                            *sample += gamma * Self::residual(value);
                            value += delta;
                        }
                        samples[3] += gamma * delta.square().square();
                    }
                    samples
                },
                &|index| {
                    self.digits
                        .iter()
                        .zip(&self.weights)
                        .map(|(digit, &gamma)| {
                            gamma * Self::residual(digit.get_bound_coeff(2 * index))
                        })
                        .sum()
                },
                previous_claim,
            )
            .map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> RelationWitness<F> for DigitRangeProver<F> {
    type Relation = CoefficientCheck;

    fn relation(&self) -> &Self::Relation {
        &self.check
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        self.digits
            .iter()
            .map(PolynomialBinding::final_claim)
            .collect()
    }
}
