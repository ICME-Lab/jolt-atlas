//! The prover side of [`super`].

use super::*;
use crate::relation::prover::RelationWitness;
use crate::upstream_sumcheck::round_check_failed;
use allocative::Allocative;
use atlas_algebra::poly::{
    multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
    split_eq_poly::SplitEqUnreducedFold,
};
use common::parallel::par_enabled;
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rayon::prelude::*;

/// Prover of a [`DigitRangeRelation`] over Gruen's split eq: per round, the summand's degree-`b`
/// factor `q(X) = Σ_j γ^j·∏_{a<b}(d_j(X) − a) + Σ_j β·b^(n−1−j)·d_j(X)` is sampled at
/// `X = 1, …, b − 1` with its leading coefficient `Σ_j γ^j·(d_j(1) − d_j(0))^b`.
#[derive(Allocative)]
pub struct DigitRangeProver<F: JoltField> {
    digits: Vec<MultilinearPolynomial<F>>,
    #[allocative(skip)]
    eq_r: GruenSplitEqPolynomial<F>,
    gamma_powers: Vec<F>,
    /// `β·b^(n−1−j)` per digit column.
    weights: Vec<F>,
    #[allocative(skip)]
    relation: DigitRangeRelation,
    #[allocative(skip)]
    setup: RelationSetup<F>,
}

impl<F: JoltField> DigitRangeProver<F> {
    /// The prover of `params` over its digit columns, in `params.digits` order.
    pub fn new(params: DigitRangeParams<F>, digits: Vec<MultilinearPolynomial<F>>) -> Self {
        assert_eq!(
            digits.len(),
            params.digits.len(),
            "one witness per digit column"
        );
        assert!(
            matches!(params.base, 2 | 4),
            "digit range prover: base {} (only 2 and 4 are implemented)",
            params.base
        );
        let eq_r = GruenSplitEqPolynomial::new(&params.r, jolt_poly::BindingOrder::LowToHigh);
        let gamma_powers = params.gamma_powers.clone();
        let weights = params
            .place_values()
            .into_iter()
            .map(|place| params.beta * F::from_u64(place))
            .collect();
        let (relation, setup) = params.into_parts();
        Self {
            digits,
            eq_r,
            gamma_powers,
            weights,
            relation,
            setup,
        }
    }

    fn bind(&mut self, r: F) {
        let r_j = F::challenge_from_field(r);
        self.digits
            .par_iter_mut()
            .with_min_len(par_enabled())
            .for_each(|d| d.bind_parallel(r_j, BindingOrder::LowToHigh));
        self.eq_r.bind(r);
    }

    /// `∏_{a<b}(x − a)`.
    fn vanishing<const B: usize>(x: F) -> F {
        (1..B).fold(x, |p, a| p * (x - F::from_u64(a as u64)))
    }

    /// The round message for base `B`: `q`'s samples at `1, …, B − 1` and its leading
    /// coefficient, per eq group.
    fn round<const B: usize>(&self, previous_claim: F) -> Result<UnivariatePoly<F>, F> {
        let Self {
            digits,
            eq_r,
            gamma_powers,
            weights,
            ..
        } = self;
        eq_r.gruen_round_from_evals::<B>(
            &|g| {
                let mut out = [F::zero(); B];
                for ((d, &gamma), &weight) in digits.iter().zip(gamma_powers).zip(weights) {
                    let d0 = d.get_bound_coeff(2 * g);
                    let d1 = d.get_bound_coeff(2 * g + 1);
                    let delta = d1 - d0;
                    let mut x = d1;
                    for sample in out.iter_mut().take(B - 1) {
                        *sample += gamma * Self::vanishing::<B>(x) + weight * x;
                        x += delta;
                    }
                    let lead = (1..B).fold(delta, |p, _| p * delta);
                    out[B - 1] += gamma * lead;
                }
                out
            },
            &|g| {
                digits
                    .iter()
                    .zip(gamma_powers)
                    .zip(weights)
                    .map(|((d, &gamma), &weight)| {
                        let d0 = d.get_bound_coeff(2 * g);
                        gamma * Self::vanishing::<B>(d0) + weight * d0
                    })
                    .sum()
            },
            previous_claim,
        )
    }
}

impl<F: JoltField> RelationWitness<F> for DigitRangeProver<F> {
    type Relation = DigitRangeRelation;

    fn relation(&self) -> &DigitRangeRelation {
        &self.relation
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        self.digits.iter().map(|d| d.final_claim()).collect()
    }
}

impl<F: JoltField> ProveRounds<F> for DigitRangeProver<F> {
    fn num_rounds(&self) -> usize {
        self.relation.rounds()
    }

    #[tracing::instrument(skip_all, name = "DigitRangeProver::prove_round")]
    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        match self.relation.base() {
            2 => self.round::<2>(previous_claim),
            4 => self.round::<4>(previous_claim),
            other => unreachable!("digit range prover: base {other} rejected at construction"),
        }
        .map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}
