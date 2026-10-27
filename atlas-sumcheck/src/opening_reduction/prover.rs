//! The prover side of [`super`].

use super::*;
use crate::relation::prover::RelationWitness;
use crate::{
    opening_proof::{prover::ProverOpeningAccumulator, OpeningId, SumcheckId},
    upstream_sumcheck::round_check_failed,
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_std::Zero;
use atlas_algebra::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        one_hot_polynomial::OneHotPolynomial,
        ra_poly::RaPolynomial,
        split_eq_poly::check_gruen_endpoint,
    },
    utils::{
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
};
use atlas_transcript::Transcript;
use common::parallel::par_enabled;
use common::CommittedPoly;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use std::{
    collections::{BTreeMap, HashMap},
    mem,
    sync::{Arc, RwLock},
};

/// Prover state for one *group* of openings in the batch opening reduction
/// sumcheck: the polynomials appended together at one point (one dense
/// polynomial, or the one-hot chunks of a lookup). The group is reduced as a
/// single instance on the polynomial `Σ_i ρ^i·P_i` (with `ρ` drawn when the
/// batch starts), so the reduction costs one instance per group rather than
/// one per polynomial.
#[derive(Clone, Allocative)]
pub struct OpeningProofReductionSumcheckProver<F>
where
    F: JoltField,
{
    pub prover_state: ProverOpening<F>,
    /// The polynomials of the group, in append order.
    pub polynomials: Vec<CommittedPoly>,
    /// The ID of the sumcheck these openings originated from
    pub sumcheck_id: SumcheckId,
    /// The group's opening point (big-endian).
    pub point: Vec<F>,
    /// Per-polynomial claims at the point.
    pub claims: Vec<F>,
    /// Per-polynomial coefficients `ρ^i` (see [`Self::set_coefficients`]).
    pub coefficients: Vec<F>,
    pub sumcheck_claim: Option<F>,
    #[allocative(skip)]
    relation: OpeningReductionRelation,
    #[allocative(skip)]
    setup: RelationSetup<F>,
}

impl<F: JoltField> RelationWitness<F> for OpeningProofReductionSumcheckProver<F> {
    type Relation = OpeningReductionRelation;

    fn relation(&self) -> &OpeningReductionRelation {
        &self.relation
    }

    /// The point and the coefficients: the input claim's setup. The reduced claim the output
    /// expression reads is the prover's own final claim, sent in the proof.
    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        Vec::new()
    }

    /// The group produces no openings; its final claim is cached for the joint opening proof.
    fn append_claims<T: Transcript>(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        let claim = match &self.prover_state {
            ProverOpening::Dense(opening) => opening.final_claim(),
            ProverOpening::OneHot(opening) => opening.final_claim(),
        };
        accumulator.cache_opening_reduction_claim(self.key(), claim);
    }
}

crate::relation_witness_instance!([F: JoltField] OpeningProofReductionSumcheckProver<F>);

impl<F: JoltField> OpeningProofReductionSumcheckProver<F> {
    fn bind(&mut self, r: F, round: usize) {
        let r_j = F::challenge_from_field(r);
        match &mut self.prover_state {
            ProverOpening::Dense(opening) => opening.bind(r_j, round),
            ProverOpening::OneHot(opening) => opening.bind(r_j, round),
        }
    }
}

impl<F: JoltField> ProveRounds<F> for OpeningProofReductionSumcheckProver<F> {
    fn num_rounds(&self) -> usize {
        self.relation.rounds()
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
        match &mut self.prover_state {
            ProverOpening::Dense(opening) => opening.compute_message(round, previous_claim),
            ProverOpening::OneHot(opening) => opening.compute_message(round, previous_claim),
        }
        .map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind, self.relation.rounds() - 1);
        Ok(())
    }
}

/// Prover opening state - either dense or one-hot polynomial.
#[derive(derive_more::From, Clone, Allocative)]
pub enum ProverOpening<F: JoltField> {
    Dense(DensePolynomialProverOpening<F>),
    OneHot(OneHotPolynomialProverOpening<F>),
}

/// An opening (of a dense polynomial) computed by the prover.
///
/// May be a batched opening, where multiple dense polynomials opened
/// at the *same* point are reduced to a single polynomial opened
/// at the (same) point.
/// Multiple openings can be accumulated and further
/// batched/reduced using a `ProverOpeningAccumulator`.
#[derive(Clone, Allocative)]
pub struct DensePolynomialProverOpening<F: JoltField> {
    /// The polynomial being opened. May be a random linear combination
    /// of multiple polynomials all being opened at the same point.
    pub polynomial: Option<Arc<RwLock<SharedDensePolynomial<F>>>>,
    /// The multilinear extension EQ(x, opening_point). This is typically
    /// an intermediate value used to compute `claim`, but is also used in
    /// the `ProverOpeningAccumulator::prove_batch_opening_reduction` sumcheck.
    pub eq_poly: Arc<RwLock<EqCycleState<F>>>,
}

impl<F: JoltField> DensePolynomialProverOpening<F> {
    #[tracing::instrument(skip_all, name = "DensePolynomialProverOpening::compute_message")]
    pub fn compute_message(
        &mut self,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, F> {
        let shared_eq = self.eq_poly.read().unwrap();
        let polynomial_ref = self.polynomial.as_ref().unwrap();
        let polynomial = &polynomial_ref.read().unwrap().poly;
        let gruen_eq = &shared_eq.D;

        // q(c) = Σ_j polynomial(c, j)·eq(r, j); the current variable is the top one.
        let mle_half = polynomial.len() / 2;
        let q_1 = fold_high_to_low(gruen_eq, |j| polynomial.get_bound_coeff(j + mle_half));
        let q_0 = || fold_high_to_low(gruen_eq, |j| polynomial.get_bound_coeff(j));
        let message = gruen_eq.gruen_poly_deg_2(q_1, previous_claim, q_0);
        check_gruen_endpoint(gruen_eq, &message, false, q_0);
        message
    }

    #[tracing::instrument(skip_all, name = "DensePolynomialProverOpening::bind")]
    pub fn bind(&mut self, r_j: F::Challenge, round: usize) {
        let mut shared_eq = self.eq_poly.write().unwrap();
        if shared_eq.num_variables_bound <= round {
            shared_eq.D.bind(r_j.into());
            shared_eq.num_variables_bound += 1;
        }

        let shared_poly_ref = self.polynomial.as_mut().unwrap();
        let mut shared_poly = shared_poly_ref.write().unwrap();
        if shared_poly.num_variables_bound <= round {
            shared_poly.poly.bind_parallel(r_j, BindingOrder::HighToLow);
            shared_poly.num_variables_bound += 1;
        }
    }

    pub fn final_claim(&self) -> F {
        let poly_ref = self.polynomial.as_ref().unwrap();
        poly_ref.read().unwrap().poly.final_claim()
    }
}

#[derive(Clone, Allocative)]
pub struct OneHotPolynomialProverOpening<F: JoltField> {
    pub log_T: usize,
    /// Address-space size shared by the group.
    pub K: usize,
    /// The group's `(nonzero indices, coefficient)` per polynomial.
    #[allocative(skip)]
    polynomials: Vec<(Arc<Vec<Option<u16>>>, F)>,
    /// `G[k] = Σ_i ρ^i·Σ_t D(t)·[idx_i(t) = k]` (Section 6.3 of Twist/Shout).
    G: Vec<F>,
    H: GroupH<F>,
    pub eq_address_state: Arc<RwLock<EqAddressState<F>>>,
    pub eq_cycle_state: Arc<RwLock<EqCycleState<F>>>,
}

impl<F: JoltField> OneHotPolynomialProverOpening<F> {
    #[tracing::instrument(skip_all, name = "OneHotPolynomialProverOpening::new")]
    pub fn new(
        eq_address_state: Arc<RwLock<EqAddressState<F>>>,
        eq_cycle_state: Arc<RwLock<EqCycleState<F>>>,
    ) -> Self {
        Self {
            log_T: 0,
            K: 1,
            polynomials: Vec::new(),
            G: Vec::new(),
            H: GroupH::None,
            eq_address_state,
            eq_cycle_state,
        }
    }

    /// Initialize with the group's polynomials and their coefficients.
    #[tracing::instrument(skip_all, name = "OneHotPolynomialProverOpening::initialize")]
    pub fn initialize(&mut self, group: Vec<(OneHotPolynomial<F>, F)>) {
        let K = group[0].0.K;
        let T = group[0].0.nonzero_indices.len();
        for (p, _) in &group {
            assert_eq!(p.K, K, "one-hot group: mismatched K");
            assert_eq!(p.nonzero_indices.len(), T, "one-hot group: mismatched T");
        }
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = (T / num_chunks).max(1);

        let eq = self.eq_cycle_state.read().unwrap();
        let D_coeffs_for_G = eq.D.merge();
        let D_coeffs_for_G = D_coeffs_for_G.evals();

        // Compute G as described in Section 6.3, summed over the group.
        let G = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_index| {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(T);
                let mut result: Vec<F> = unsafe_allocate_zero_vec(K);
                for (p, c) in &group {
                    let single = group.len() == 1;
                    for (j, k) in p.nonzero_indices[start..end].iter().enumerate() {
                        if let Some(k) = k {
                            let d = D_coeffs_for_G[start + j];
                            result[*k as usize] += if single { d } else { *c * d };
                        }
                    }
                }
                result
            })
            .reduce(
                || unsafe_allocate_zero_vec(K),
                |mut running, new| {
                    running
                        .par_iter_mut()
                        .zip(new.into_par_iter())
                        .with_min_len(par_enabled())
                        .for_each(|(x, y)| *x += y);
                    running
                },
            );

        self.G = G;
        self.K = K;
        self.log_T = T.log_2();
        self.polynomials = group
            .into_iter()
            .map(|(p, c)| (p.nonzero_indices.clone(), c))
            .collect();
    }

    #[tracing::instrument(skip_all, name = "OneHotPolynomialProverOpening::compute_message")]
    pub fn compute_message(
        &mut self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, F> {
        let shared_eq_address = self.eq_address_state.read().unwrap();
        let shared_eq_cycle = self.eq_cycle_state.read().unwrap();

        if round < self.K.log_2() {
            let num_unbound_address_variables = self.K.log_2() - round;
            let B = &shared_eq_address.B;
            let F = &shared_eq_address.F;
            let G = &self.G;

            let unreduced_univariate_poly_evals = (0..B.len() / 2)
                .into_par_iter()
                .with_min_len(par_enabled())
                .map(|k_prime| {
                    let B_evals = B.sumcheck_evals_array::<2>(k_prime, BindingOrder::HighToLow);
                    let inner_sum = G
                        .par_iter()
                        .with_min_len(par_enabled())
                        .enumerate()
                        .skip(k_prime)
                        .step_by(B.len() / 2)
                        .map(|(k, &G_k)| {
                            let k_m = (k >> (num_unbound_address_variables - 1)) & 1;
                            let F_k = F[k >> num_unbound_address_variables];
                            let G_times_F = G_k * F_k;

                            let eval_c0 = if k_m == 0 { G_times_F } else { F::zero() };
                            let eval_c2 = if k_m == 0 {
                                -G_times_F
                            } else {
                                G_times_F + G_times_F
                            };
                            [eval_c0, eval_c2]
                        })
                        .reduce(
                            || [F::zero(); 2],
                            |running, new| [running[0] + new[0], running[1] + new[1]],
                        );

                    [
                        B_evals[0].mul_to_product_accum(inner_sum[0]),
                        B_evals[1].mul_to_product_accum(inner_sum[1]),
                    ]
                })
                .reduce(
                    || [F::UnreducedProductAccum::zero(); 2],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                );

            let univariate_poly_evals = unreduced_univariate_poly_evals
                .into_iter()
                .map(|evals| F::reduce_product_accum(evals))
                .collect::<Vec<_>>();

            Ok(UnivariatePoly::from_evals_and_hint(
                previous_claim,
                &univariate_poly_evals,
            ))
        } else {
            // T-variable rounds
            let B = &shared_eq_address.B;
            let d_gruen = &shared_eq_cycle.D;
            let eq_r_address_claim = B.final_claim();
            let H = &self.H;

            // s(X) = eq(r_address, ρ)·D(X)·q(X). A vanishing address factor makes the whole
            // message zero, so it is never divided by.
            let Some(eq_r_address_inverse) = eq_r_address_claim.inverse() else {
                return if previous_claim.is_zero() {
                    Ok(UnivariatePoly::new(vec![F::zero(); 3]))
                } else {
                    Err(F::zero())
                };
            };
            let h_half = H.len() / 2;
            let q_1 = fold_high_to_low(d_gruen, |j| H.get_bound_coeff(j + h_half));
            let q_0 = || fold_high_to_low(d_gruen, |j| H.get_bound_coeff(j));
            let message = d_gruen.gruen_poly_deg_2(q_1, previous_claim * eq_r_address_inverse, q_0);
            check_gruen_endpoint(d_gruen, &message, false, q_0);
            Ok(message? * eq_r_address_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPolynomialProverOpening::bind")]
    pub fn bind(&mut self, r: F::Challenge, round: usize) {
        let mut shared_eq_address = self.eq_address_state.write().unwrap();
        let mut shared_eq_cycle = self.eq_cycle_state.write().unwrap();
        let log_K = self.K.log_2();
        let num_variables_bound =
            shared_eq_address.num_variables_bound + shared_eq_cycle.num_variables_bound;

        // Bind shared state if not already bound
        if num_variables_bound <= round {
            if round < log_K {
                shared_eq_address
                    .B
                    .bind_parallel(r, BindingOrder::HighToLow);

                shared_eq_address.F.update(r);
                shared_eq_address.num_variables_bound += 1;
            } else {
                shared_eq_cycle.D.bind(r.into());
                shared_eq_cycle.num_variables_bound += 1;
            }
        }

        // For the first two log T rounds we want to use F still
        if round == log_K - 1 {
            if matches!(self.H, GroupH::None) {
                let f = shared_eq_address.F.clone_values();
                self.H = if self.polynomials.len() == 1 {
                    GroupH::Single(RaPolynomial::new(self.polynomials[0].0.clone(), f))
                } else {
                    let polys = &self.polynomials;
                    let T = polys[0].0.len();
                    let h: Vec<F> = (0..T)
                        .into_par_iter()
                        .with_min_len(par_enabled())
                        .map(|t| {
                            polys.iter().fold(F::zero(), |acc, (idx, c)| match idx[t] {
                                Some(k) => acc + *c * f[k as usize],
                                None => acc,
                            })
                        })
                        .collect();
                    GroupH::Dense(MultilinearPolynomial::from(h))
                };
            }

            let g = mem::take(&mut self.G);
            drop_in_background_thread(g);
        } else if round >= log_K {
            // Bind H for subsequent T rounds
            if self.H.len().log_2() == self.log_T + log_K - round {
                self.H.bind_parallel(r);
            }
        }
    }

    pub fn final_claim(&self) -> F {
        self.H.final_claim()
    }
}

impl<F> OpeningProofReductionSumcheckProver<F>
where
    F: JoltField,
{
    pub fn new_dense<U>(
        polynomial: CommittedPoly,
        sumcheck_id: SumcheckId,
        eq_poly: Arc<RwLock<EqCycleState<F>>>,
        opening_point: Vec<U>,
        claim: F,
    ) -> Self
    where
        U: Copy + Send + Sync + Into<F>,
    {
        let opening = DensePolynomialProverOpening {
            polynomial: None, // Defer initialization until opening proof reduction sumcheck
            eq_poly,
        };
        Self::group(
            vec![polynomial],
            sumcheck_id,
            opening_point,
            vec![claim],
            opening.into(),
        )
    }

    pub fn new_one_hot<U>(
        polynomials: Vec<CommittedPoly>,
        sumcheck_id: SumcheckId,
        eq_address: Arc<RwLock<EqAddressState<F>>>,
        eq_cycle: Arc<RwLock<EqCycleState<F>>>,
        opening_point: Vec<U>,
        claims: Vec<F>,
    ) -> Self
    where
        U: Copy + Send + Sync + Into<F>,
    {
        assert_eq!(polynomials.len(), claims.len());
        let opening = OneHotPolynomialProverOpening::new(eq_address, eq_cycle);
        Self::group(
            polynomials,
            sumcheck_id,
            opening_point,
            claims,
            opening.into(),
        )
    }

    fn group<U>(
        polynomials: Vec<CommittedPoly>,
        sumcheck_id: SumcheckId,
        opening_point: Vec<U>,
        claims: Vec<F>,
        prover_state: ProverOpening<F>,
    ) -> Self
    where
        U: Copy + Send + Sync + Into<F>,
    {
        let point: Vec<F> = opening_point.into_iter().map(Into::into).collect();
        let coefficients = vec![F::one(); polynomials.len()];
        let relation =
            OpeningReductionRelation::group(polynomials.clone(), sumcheck_id, point.len());
        let setup = RelationSetup {
            incoming_point: point.clone(),
            challenges: coefficient_challenges(&coefficients),
        };
        Self {
            prover_state,
            polynomials,
            sumcheck_id,
            point,
            claims,
            coefficients,
            sumcheck_claim: None,
            relation,
            setup,
        }
    }

    /// The instance's key: the first polynomial's opening id.
    pub fn key(&self) -> OpeningId {
        OpeningId::new(self.polynomials[0], self.sumcheck_id)
    }

    /// Fix the in-group coefficients to `ρ^i`.
    pub fn set_coefficients(&mut self, rho: F) {
        let mut c = F::one();
        self.coefficients = (0..self.polynomials.len())
            .map(|_| {
                let x = c;
                c *= rho;
                x
            })
            .collect();
        self.setup.challenges = coefficient_challenges(&self.coefficients);
    }

    #[tracing::instrument(skip_all, name = "OpeningProofReductionSumcheck::prepare_sumcheck")]
    pub fn prepare_sumcheck(
        &mut self,
        polynomials_map: &BTreeMap<CommittedPoly, MultilinearPolynomial<F>>,
        shared_dense_polynomials: &HashMap<CommittedPoly, Arc<RwLock<SharedDensePolynomial<F>>>>,
    ) {
        #[cfg(test)]
        {
            use atlas_algebra::poly::multilinear_polynomial::PolynomialEvaluation;
            for (poly_id, claim) in self.polynomials.iter().zip(&self.claims) {
                let poly = polynomials_map.get(poly_id).unwrap();
                debug_assert_eq!(
                    poly.evaluate(&self.point),
                    *claim,
                    "Evaluation mismatch for {:?} {:?}",
                    self.sumcheck_id,
                    poly_id,
                );
                let num_vars = poly.get_num_vars();
                let opening_point_len = self.point.len();
                debug_assert_eq!(
                    num_vars,
                    opening_point_len,
                    "{:?} has {num_vars} variables but opening point from {:?} has length {opening_point_len}",
                    poly_id,
                    self.sumcheck_id,
                );
            }
        }

        match &mut self.prover_state {
            ProverOpening::Dense(opening) => {
                let poly = shared_dense_polynomials.get(&self.polynomials[0]).unwrap();
                opening.polynomial = Some(poly.clone());
            }
            ProverOpening::OneHot(opening) => {
                let group: Vec<(OneHotPolynomial<F>, F)> = self
                    .polynomials
                    .iter()
                    .zip(&self.coefficients)
                    .map(|(id, c)| match polynomials_map.get(id).unwrap() {
                        MultilinearPolynomial::OneHot(one_hot) => (one_hot.clone(), *c),
                        _ => panic!("Unexpected non-one-hot polynomial"),
                    })
                    .collect();
                opening.initialize(group);
            }
        };
    }

    pub fn cache_sumcheck_claim(&mut self) {
        debug_assert!(self.sumcheck_claim.is_none());
        let claim = match &mut self.prover_state {
            ProverOpening::Dense(opening) => opening.final_claim(),
            ProverOpening::OneHot(opening) => opening.final_claim(),
        };
        self.sumcheck_claim = Some(claim);
    }
}
