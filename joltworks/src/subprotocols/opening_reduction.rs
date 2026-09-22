//! Opening reduction sumcheck prover and verifier.
//!
//! This module contains the sumcheck-specific logic for the batch opening reduction protocol.
//! The higher-level orchestration remains in `poly/opening_proof.rs`.

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        one_hot_polynomial::OneHotPolynomial,
        opening_proof::{
            Opening, OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator,
            SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        ra_poly::RaPolynomial,
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{
        expanding_table::ExpandingTable,
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_std::Zero;
use common::parallel::par_enabled;
use common::CommittedPoly;
use rayon::prelude::*;
use std::{
    collections::{BTreeMap, HashMap},
    mem,
    ops::Mul,
    sync::{Arc, RwLock},
};

/// Degree of the sumcheck round polynomials in opening reduction.
pub const OPENING_SUMCHECK_DEGREE: usize = 2;

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
    /// (point, `Σ_i coefficients[i]·claims[i]`)
    pub opening: Opening<F>,
    /// Per-polynomial claims at the point.
    pub claims: Vec<F>,
    /// Per-polynomial coefficients `ρ^i` (see [`Self::set_coefficients`]).
    pub coefficients: Vec<F>,
    pub sumcheck_claim: Option<F>,
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
        Self {
            polynomials: vec![polynomial],
            sumcheck_id,
            opening: (opening_point.into(), claim),
            claims: vec![claim],
            coefficients: vec![F::one()],
            prover_state: opening.into(),
            sumcheck_claim: None,
        }
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
        let n = polynomials.len();
        Self {
            polynomials,
            sumcheck_id,
            opening: (opening_point.into(), claims.iter().copied().sum()),
            claims,
            coefficients: vec![F::one(); n],
            prover_state: opening.into(),
            sumcheck_claim: None,
        }
    }

    /// The instance's key: the first polynomial's opening id.
    pub fn key(&self) -> OpeningId {
        OpeningId::new(self.polynomials[0], self.sumcheck_id)
    }

    /// Fix the in-group coefficients to `ρ^i` and the batched input claim.
    pub fn set_coefficients(&mut self, rho: F) {
        let mut c = F::one();
        self.coefficients = (0..self.polynomials.len())
            .map(|_| {
                let x = c;
                c *= rho;
                x
            })
            .collect();
        self.opening.1 = self
            .coefficients
            .iter()
            .zip(&self.claims)
            .map(|(c, v)| *c * *v)
            .sum();
    }

    #[tracing::instrument(skip_all, name = "OpeningProofReductionSumcheck::prepare_sumcheck")]
    pub fn prepare_sumcheck(
        &mut self,
        polynomials_map: &BTreeMap<CommittedPoly, MultilinearPolynomial<F>>,
        shared_dense_polynomials: &HashMap<CommittedPoly, Arc<RwLock<SharedDensePolynomial<F>>>>,
    ) {
        #[cfg(test)]
        {
            use crate::poly::multilinear_polynomial::PolynomialEvaluation;
            for (poly_id, claim) in self.polynomials.iter().zip(&self.claims) {
                let poly = polynomials_map.get(poly_id).unwrap();
                debug_assert_eq!(
                    poly.evaluate(&self.opening.0.r),
                    *claim,
                    "Evaluation mismatch for {:?} {:?}",
                    self.sumcheck_id,
                    poly_id,
                );
                let num_vars = poly.get_num_vars();
                let opening_point_len = self.opening.0.len();
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

impl<F: JoltField> SumcheckInstanceParams<F> for Opening<F> {
    fn degree(&self) -> usize {
        OPENING_SUMCHECK_DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.0.len()
    }

    fn input_claim(&self, _: &dyn OpeningAccumulator<F>) -> F {
        self.1
    }

    fn normalize_opening_point(&self, _: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        unimplemented!("Unused")
    }

    // ZK methods: minimal starting impls so `BatchedSumcheck::prove_zk` doesn't
    // hit the trait's `todo!()` defaults. These are placeholders -- input claim
    // is left as a free witness variable and the output claim is not yet bound
    // to `eq * y_P`. Closing the loop requires the params to know the
    // polynomial and prior sumcheck id (see
    // wiki/jolt-atlas/book/src/underway/batched-opening-sound-verifier.md).
    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> crate::subprotocols::blindfold::InputClaimConstraint {
        crate::subprotocols::blindfold::InputClaimConstraint::default()
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        Vec::new()
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(
        &self,
    ) -> Option<crate::subprotocols::blindfold::OutputClaimConstraint> {
        None
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, _: &[F::Challenge]) -> Vec<F> {
        Vec::new()
    }
}

impl<F, T: Transcript> SumcheckInstanceProver<F, T> for OpeningProofReductionSumcheckProver<F>
where
    F: JoltField,
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.opening
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        match &mut self.prover_state {
            ProverOpening::Dense(opening) => opening.compute_message(round, previous_claim),
            ProverOpening::OneHot(opening) => opening.compute_message(round, previous_claim),
        }
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        match &mut self.prover_state {
            ProverOpening::Dense(opening) => opening.bind(r_j, round),
            ProverOpening::OneHot(opening) => opening.bind(r_j, round),
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // Cache the final sumcheck claim in the accumulator
        let claim = match &self.prover_state {
            ProverOpening::Dense(opening) => opening.final_claim(),
            ProverOpening::OneHot(opening) => opening.final_claim(),
        };
        accumulator.cache_opening_reduction_claim(self.key(), claim);

        // Also register the reduced evaluation as a standard opening so the
        // BlindFold R1CS can reference it via `OpeningId`. The opening point
        // is the batched-opening sumcheck's challenge vector r_sumcheck; the
        // claim is the per-poly value P(r_sumcheck). Stored unconditionally
        // (cheap insert); only the `--features zk` path actually consumes it
        // via the extra constraint linking `joint_claim = sum gamma_i * y_P_i`
        // to `y_com`.
        use crate::field::IntoOpening;
        let opening_point = OpeningPoint::new(sumcheck_challenges.into_opening());
        let opening_id = crate::poly::opening_proof::OpeningId::new(
            self.polynomials[0],
            SumcheckId::BlindFoldBatchOpening,
        );
        accumulator
            .openings
            .insert(opening_id, (opening_point, claim));
    }
}

/// Verifier state for one group of openings in the batch opening reduction sumcheck.
pub struct OpeningProofReductionSumcheckVerifier<F>
where
    F: JoltField,
{
    /// The polynomials of the group, in append order.
    pub polynomials: Vec<CommittedPoly>,
    /// The ID of the sumcheck these openings originated from
    pub sumcheck_id: SumcheckId,
    opening: Opening<F>,
    /// Per-polynomial claims at the point.
    pub claims: Vec<F>,
    /// Per-polynomial coefficients `ρ^i`.
    pub coefficients: Vec<F>,
    pub sumcheck_claim: Option<F>,
}

impl<F: JoltField> OpeningProofReductionSumcheckVerifier<F> {
    pub fn new<U>(
        polynomials: Vec<CommittedPoly>,
        sumcheck_id: SumcheckId,
        opening_point: Vec<U>,
        claims: Vec<F>,
    ) -> Self
    where
        U: Copy + Send + Sync + Into<F>,
    {
        assert_eq!(polynomials.len(), claims.len());
        let n = polynomials.len();
        Self {
            polynomials,
            sumcheck_id,
            opening: (opening_point.into(), claims.iter().copied().sum()),
            claims,
            coefficients: vec![F::one(); n],
            sumcheck_claim: None,
        }
    }

    /// The instance's key: the first polynomial's opening id.
    pub fn key(&self) -> OpeningId {
        OpeningId::new(self.polynomials[0], self.sumcheck_id)
    }

    /// Fix the in-group coefficients to `ρ^i` and the batched input claim.
    pub fn set_coefficients(&mut self, rho: F) {
        let mut c = F::one();
        self.coefficients = (0..self.polynomials.len())
            .map(|_| {
                let x = c;
                c *= rho;
                x
            })
            .collect();
        self.opening.1 = self
            .coefficients
            .iter()
            .zip(&self.claims)
            .map(|(c, v)| *c * *v)
            .sum();
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for OpeningProofReductionSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.opening
    }

    fn expected_output_claim(
        &self,
        _accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let eq_eval = EqPolynomial::<F>::mle(&self.opening.0.r, sumcheck_challenges);
        eq_eval * self.sumcheck_claim.unwrap()
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // Mirror the prover-side insert: register an opening at
        // `(self.polynomial, SumcheckId::BlindFoldBatchOpening)` so the
        // BlindFold R1CS can reference it. In ZK mode the actual claim is a
        // placeholder; the value the prover assigned is verified by BlindFold.
        use crate::field::IntoOpening;
        let opening_point = OpeningPoint::new(sumcheck_challenges.into_opening());
        let opening_id = crate::poly::opening_proof::OpeningId::new(
            self.polynomials[0],
            SumcheckId::BlindFoldBatchOpening,
        );
        accumulator
            .openings
            .insert(opening_id, (opening_point, F::zero()));
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
    pub fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let shared_eq = self.eq_poly.read().unwrap();
        let polynomial_ref = self.polynomial.as_ref().unwrap();
        let polynomial = &polynomial_ref.read().unwrap().poly;
        let gruen_eq = &shared_eq.D;

        // Compute q(0) = sum of polynomial(i) * eq(r, i) for i in [0, mle_half)
        let mle_half = polynomial.len() / 2;
        let q_0 = if gruen_eq.E_in_current_len() <= 1 {
            // E_in is fully bound
            let unreduced_q_0 = (0..mle_half)
                .into_par_iter()
                .with_min_len(par_enabled())
                .map(|j| {
                    let eq_eval = gruen_eq.E_out_current()[j];
                    // TODO(quang): special case depending on the polynomial type?
                    let poly_eval = polynomial.get_bound_coeff(j);
                    eq_eval.mul_unreduced::<9>(poly_eval)
                })
                .reduce(F::Unreduced::<9>::zero, |running, new| running + new);
            F::from_montgomery_reduce(unreduced_q_0)
        } else {
            let num_x_out = gruen_eq.E_out_current_len();
            let num_x_in = gruen_eq.E_in_current_len();
            let num_x_out_bits = num_x_out.log_2();
            let d_e_in = gruen_eq.E_in_current();
            let d_e_out = gruen_eq.E_out_current();

            (0..num_x_in)
                .into_par_iter()
                .with_min_len(par_enabled())
                .map(|x_in| {
                    let unreduced_inner_sum = (0..num_x_out)
                        .into_par_iter()
                        .with_min_len(par_enabled())
                        .map(|x_out| {
                            let j = (x_in << num_x_out_bits) | x_out;
                            let poly_eval = polynomial.get_bound_coeff(j);
                            d_e_out[x_out].mul_unreduced::<9>(poly_eval)
                        })
                        .reduce(F::Unreduced::<9>::zero, |running, new| running + new);
                    let inner_sum = F::from_montgomery_reduce(unreduced_inner_sum);
                    d_e_in[x_in] * inner_sum
                })
                .sum()
        };

        gruen_eq.gruen_poly_deg_2(q_0, previous_claim)
    }

    #[tracing::instrument(skip_all, name = "DensePolynomialProverOpening::bind")]
    pub fn bind(&mut self, r_j: F::Challenge, round: usize) {
        let mut shared_eq = self.eq_poly.write().unwrap();
        if shared_eq.num_variables_bound <= round {
            shared_eq.D.bind(r_j);
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

/// Shared state for a dense polynomial during sumcheck binding.
#[derive(Clone, Debug, Allocative)]
pub struct SharedDensePolynomial<F: JoltField> {
    pub poly: MultilinearPolynomial<F>,
    /// The number of variables that have been bound during sumcheck so far
    pub num_variables_bound: usize,
}

impl<F: JoltField> SharedDensePolynomial<F> {
    pub fn new(poly: MultilinearPolynomial<F>) -> Self {
        Self {
            poly,
            num_variables_bound: 0,
        }
    }
}

/// State related to the address variable (i.e. k) terms appearing in the opening
/// proof reduction sumcheck.
#[derive(Clone, Debug, Allocative)]
pub struct EqAddressState<F: JoltField> {
    /// B stores eq(r, k), see Equation (53)
    pub B: MultilinearPolynomial<F>,
    /// F will maintain an array that, at the end of sumcheck round m, has size 2^m
    /// and stores all 2^m values eq((k_1, ..., k_m), (r_1, ..., r_m))
    pub F: ExpandingTable<F>,
    /// The number of variables that have been bound during sumcheck so far
    pub num_variables_bound: usize,
}

/// State related to the cycle variable (i.e. j) terms appearing in the opening
/// proof reduction sumcheck.
#[derive(Clone, Debug, Allocative)]
pub struct EqCycleState<F: JoltField> {
    /// D stores eq(r', j), see Equation (54) but with Gruen X Dao-Thaler optimizations
    pub D: GruenSplitEqPolynomial<F>,
    /// The number of variables that have been bound during sumcheck so far
    pub num_variables_bound: usize,
}

impl<F: JoltField> EqAddressState<F> {
    #[tracing::instrument(skip_all, name = "EqAddressState::new")]
    pub fn new<U>(r_address: &[U]) -> Self
    where
        U: Copy + Send + Sync + Into<F>,
        F: Mul<U, Output = F>,
    {
        let K = 1 << r_address.len();
        // F will maintain an array that, at the end of sumcheck round m, has size 2^m
        // and stores all 2^m values eq((k_1, ..., k_m), (r_1, ..., r_m))
        let mut F = ExpandingTable::new(K, BindingOrder::HighToLow);
        F.reset(F::one());

        Self {
            B: MultilinearPolynomial::from(EqPolynomial::<F>::evals(r_address)),
            F,
            num_variables_bound: 0,
        }
    }
}

impl<F: JoltField> EqCycleState<F> {
    #[tracing::instrument(skip_all, name = "EqCycleState::new")]
    pub fn new<U>(r_cycle: &[U]) -> Self
    where
        U: Copy + Send + Sync + Into<F>,
        F: Mul<U, Output = F>,
    {
        let D = GruenSplitEqPolynomial::new(r_cycle, BindingOrder::HighToLow);
        Self {
            D,
            num_variables_bound: 0,
        }
    }
}

/// The opening proof reduction sumcheck is a batched sumcheck where
/// each sumcheck instance in the batch corresponds to one opening.
/// The sumcheck instance for a one-hot polynomial opening has the form
///   \sum eq(k, r_address) * eq(j, r_cycle) * ra(k, j)
/// so we use a simplified version of the prover algorithm for the
/// Booleanity sumcheck described in Section 6.3 of the Twist/Shout paper.
/// The `H` polynomial of a one-hot opening group during the cycle rounds.
#[derive(Clone, Allocative)]
enum GroupH<F: JoltField> {
    None,
    /// A single polynomial: the lazily materialized `ra(r_address, ·)`.
    Single(RaPolynomial<u16, F>),
    /// A group: the dense `Σ_i ρ^i·ra_i(r_address, ·)`.
    Dense(MultilinearPolynomial<F>),
}

impl<F: JoltField> GroupH<F> {
    fn len(&self) -> usize {
        match self {
            Self::None => panic!("H not initialized"),
            Self::Single(h) => h.len(),
            Self::Dense(h) => h.len(),
        }
    }

    fn get_bound_coeff(&self, j: usize) -> F {
        match self {
            Self::None => panic!("H not initialized"),
            Self::Single(h) => h.get_bound_coeff(j),
            Self::Dense(h) => h.get_bound_coeff(j),
        }
    }

    fn bind_parallel(&mut self, r: F::Challenge) {
        match self {
            Self::None => panic!("H not initialized"),
            Self::Single(h) => h.bind_parallel(r, BindingOrder::HighToLow),
            Self::Dense(h) => h.bind_parallel(r, BindingOrder::HighToLow),
        }
    }

    fn final_claim(&self) -> F {
        match self {
            Self::None => panic!("H not initialized"),
            Self::Single(h) => h.final_claim(),
            Self::Dense(h) => h.final_claim(),
        }
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
        let D_coeffs_for_G = &eq.D.merge();

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
    pub fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
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
                        B_evals[0].mul_unreduced::<9>(inner_sum[0]),
                        B_evals[1].mul_unreduced::<9>(inner_sum[1]),
                    ]
                })
                .reduce(
                    || [F::Unreduced::<9>::zero(); 2],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                );

            let univariate_poly_evals = unreduced_univariate_poly_evals
                .into_iter()
                .map(|evals| F::from_montgomery_reduce(evals))
                .collect::<Vec<_>>();

            UniPoly::from_evals_and_hint(previous_claim, &univariate_poly_evals)
        } else {
            // T-variable rounds
            let B = &shared_eq_address.B;
            let d_gruen = &shared_eq_cycle.D;
            let eq_r_address_claim = B.final_claim();
            let H = &self.H;

            let gruen_eval_0 = if d_gruen.E_in_current_len() == 1 {
                let unreduced_gruen_eval_0 = (0..d_gruen.len() / 2)
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|j| d_gruen.E_out_current()[j].mul_unreduced::<9>(H.get_bound_coeff(j)))
                    .reduce(F::Unreduced::<9>::zero, |running, new| running + new);
                F::from_montgomery_reduce(unreduced_gruen_eval_0)
            } else {
                let d_e_in = d_gruen.E_in_current();
                let d_e_out = d_gruen.E_out_current();
                let num_x_in = d_gruen.E_in_current_len();
                let num_x_out = d_gruen.E_out_current_len();
                let num_x_out_bits = num_x_out.log_2();

                (0..num_x_in)
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|x_in| {
                        let unreduced_inner_sum = (0..num_x_out)
                            .into_par_iter()
                            .with_min_len(par_enabled())
                            .map(|x_out| {
                                let j = (x_in << num_x_out_bits) | x_out;
                                d_e_out[x_out].mul_unreduced::<9>(H.get_bound_coeff(j))
                            })
                            .reduce(F::Unreduced::<9>::zero, |running, new| running + new);
                        let inner_sum = F::from_montgomery_reduce(unreduced_inner_sum);
                        d_e_in[x_in] * inner_sum
                    })
                    .sum()
            };

            let gruen_univariate_evals =
                d_gruen.gruen_poly_deg_2(gruen_eval_0, previous_claim / eq_r_address_claim);

            gruen_univariate_evals * eq_r_address_claim
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
                shared_eq_cycle.D.bind(r);
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

#[cfg(test)]
mod tests {
    use crate::{
        field::JoltField,
        poly::{
            commitment::{
                commitment_scheme::CommitmentScheme,
                hyperkzg::{
                    HyperKZG, HyperKZGCommitment, HyperKZGProof, HyperKZGProverKey, HyperKZGSRS,
                    HyperKZGVerifierKey,
                },
            },
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            one_hot_polynomial::OneHotPolynomial,
            opening_proof::{
                OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
                VerifierOpeningAccumulator, BIG_ENDIAN,
            },
            rlc_polynomial::build_materialized_rlc,
        },
        transcripts::{Blake2bTranscript, Transcript},
    };
    use ark_bn254::Bn254;
    use ark_ec::pairing::Pairing;
    use ark_std::UniformRand;
    use common::CommittedPoly;
    use itertools::Itertools;
    use rand::{Rng, SeedableRng};
    use std::collections::BTreeMap;

    type Fr = <Bn254 as Pairing>::ScalarField;
    type Challenge = <Fr as JoltField>::Challenge;

    #[test]
    fn test_3_dense() {
        let log_T = 6;
        struct PolyData {
            poly: MultilinearPolynomial<Fr>,
            commitment: HyperKZGCommitment<Bn254>,
            point: Vec<Challenge>,
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0x76543);

        // Define dense polynomial sizes (log_N values)
        let dense_configs = [log_T, log_T, log_T];

        // Calculate max size needed
        let max_dense_size = 1 << dense_configs.iter().max().unwrap();

        // Setup
        let srs = HyperKZGSRS::setup(&mut rng, max_dense_size);
        let (pk, vk): (HyperKZGProverKey<Bn254>, HyperKZGVerifierKey<Bn254>) =
            srs.trim(max_dense_size);
        let mut prover_tr = Blake2bTranscript::new(b"TestEval");
        let mut prover_opening_accumulator: ProverOpeningAccumulator<Fr> =
            ProverOpeningAccumulator::new();

        // Create dense polynomials
        let dense_polys: Vec<PolyData> = dense_configs
            .iter()
            .enumerate()
            .map(|(i, &log_n)| {
                let n = 1 << log_n;
                // Generate evaluation point based on largest polynomial
                let point: Vec<Challenge> = (0..log_n)
                    .map(|_| Challenge::from(rng.gen::<u128>()))
                    .collect();
                let raw: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
                let poly = MultilinearPolynomial::from(raw);
                let eval = poly.evaluate(&point);
                let commitment = HyperKZG::commit(&poly, &pk).0;
                let id = OpeningId::new(
                    CommittedPoly::DivNodeQuotient(i),
                    SumcheckId::NodeExecution(0),
                );
                prover_opening_accumulator.append_dense(&mut prover_tr, id, point.clone(), eval);
                PolyData {
                    poly,
                    commitment,
                    point,
                }
            })
            .collect();

        // Combine all polynomials for RLC
        let all_polys: Vec<(CommittedPoly, MultilinearPolynomial<Fr>)> = dense_polys
            .iter()
            .enumerate()
            .map(|(i, data)| (CommittedPoly::DivNodeQuotient(i), data.poly.clone()))
            .collect();

        // Prepare sumcheck
        let polynomial_map = BTreeMap::from_iter(all_polys);

        prover_opening_accumulator.prepare_for_sumcheck(&polynomial_map, &mut prover_tr);

        // Run sumcheck
        let (accumulator_sumcheck_proof, r_sumcheck_acc) =
            prover_opening_accumulator.prove_batch_opening_sumcheck(&mut prover_tr);

        // Finalize sumcheck (uses claims cached via cache_openings, derives gamma, cleans up)
        let state = prover_opening_accumulator
            .finalize_batch_opening_sumcheck(r_sumcheck_acc.clone(), &mut prover_tr);
        let sumcheck_claims: Vec<Fr> = state.sumcheck_claims.clone();

        // Build RLC
        let rlc = build_materialized_rlc(&state.poly_coeffs, &polynomial_map);

        // Prove
        let eval_proof: HyperKZGProof<Bn254> =
            HyperKZG::open(&pk, &rlc, &state.r_sumcheck, &mut prover_tr).unwrap();

        // Verify
        let mut verifier_tr = Blake2bTranscript::new(b"TestEval");
        verifier_tr.compare_to(prover_tr);
        let mut verifier_opening_accumulator: VerifierOpeningAccumulator<Fr> =
            VerifierOpeningAccumulator::new();
        // Take claims
        for (key, (_, value)) in &prover_opening_accumulator.openings {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::default();
            verifier_opening_accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }

        dense_polys.iter().enumerate().for_each(|(i, data)| {
            let id = OpeningId::new(
                CommittedPoly::DivNodeQuotient(i),
                SumcheckId::NodeExecution(0),
            );
            verifier_opening_accumulator.append_dense(&mut verifier_tr, id, data.point.clone());
        });

        // Prepare - populate sumcheck claims
        verifier_opening_accumulator
            .prepare_for_sumcheck(&sumcheck_claims, &mut verifier_tr)
            .unwrap();

        // Verify sumcheck
        let r_sumcheck = verifier_opening_accumulator
            .verify_batch_opening_sumcheck(&accumulator_sumcheck_proof, &mut verifier_tr)
            .unwrap();

        // Finalize and store state in accumulator for Stage 8
        let verifier_state = verifier_opening_accumulator.finalize_batch_opening_sumcheck(
            r_sumcheck,
            &sumcheck_claims,
            &mut verifier_tr,
        );

        let mut commitments_map: BTreeMap<CommittedPoly, HyperKZGCommitment<Bn254>> =
            BTreeMap::new();
        dense_polys.iter().enumerate().for_each(|(i, data)| {
            commitments_map.insert(CommittedPoly::DivNodeQuotient(i), data.commitment.clone());
        });

        // Compute joint commitment
        let joint_commitment = VerifierOpeningAccumulator::compute_joint_commitment::<
            HyperKZG<Bn254>,
        >(&mut commitments_map, &verifier_state);

        // Verify joint opening
        verifier_opening_accumulator
            .verify_joint_opening::<_, HyperKZG<Bn254>>(
                &vk,
                &eval_proof,
                &joint_commitment,
                &verifier_state,
                &mut verifier_tr,
            )
            .unwrap();
    }

    #[test]
    fn test_3_dense_oh() {
        let log_T = 6;
        struct PolyData {
            poly: MultilinearPolynomial<Fr>,
            commitment: HyperKZGCommitment<Bn254>,
            point: Vec<Challenge>,
        }

        struct OneHotPolyData {
            poly: OneHotPolynomial<Fr>,
            commitment: HyperKZGCommitment<Bn254>,
            point: Vec<Challenge>,
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0x76543);

        // Define dense polynomial sizes (log_N values)
        let dense_configs = [log_T, log_T, log_T];

        // Define OneHot polynomial parameters (log_K, log_T)
        let oh_configs = [(5, log_T), (5, log_T), (5, log_T)]; // (K, T) pairs

        // Calculate max size needed
        let max_dense_size = 1 << dense_configs.iter().max().unwrap();
        let max_oh_size = oh_configs
            .iter()
            .map(|(log_k, log_t)| (1 << log_k) * (1 << log_t))
            .max()
            .unwrap();
        let max_size = max_dense_size.max(max_oh_size);

        // Setup
        let srs = HyperKZGSRS::setup(&mut rng, max_size);
        let (pk, vk): (HyperKZGProverKey<Bn254>, HyperKZGVerifierKey<Bn254>) = srs.trim(max_size);
        let mut prover_tr = Blake2bTranscript::new(b"TestEval");
        let mut prover_opening_accumulator: ProverOpeningAccumulator<Fr> =
            ProverOpeningAccumulator::new();

        // Create dense polynomials
        let dense_polys: Vec<PolyData> = dense_configs
            .iter()
            .enumerate()
            .map(|(i, &log_n)| {
                let n = 1 << log_n;
                // Generate evaluation point based on largest polynomial
                let point: Vec<Challenge> = (0..log_n)
                    .map(|_| Challenge::from(rng.gen::<u128>()))
                    .collect();
                let raw: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
                let poly = MultilinearPolynomial::from(raw);
                let eval = poly.evaluate(&point);
                let commitment = HyperKZG::commit(&poly, &pk).0;
                let id = OpeningId::new(
                    CommittedPoly::DivNodeQuotient(i),
                    SumcheckId::NodeExecution(0),
                );
                prover_opening_accumulator.append_dense(&mut prover_tr, id, point.clone(), eval);
                PolyData {
                    poly,
                    commitment,
                    point,
                }
            })
            .collect();

        // Create OneHot polynomials
        let oh_polys: Vec<OneHotPolyData> = oh_configs
            .iter()
            .enumerate()
            .map(|(i, &(log_k, log_t))| {
                let k = 1 << log_k;
                let t = 1 << log_t;
                let num_vars = log_k + log_t;
                let point = (0..num_vars)
                    .map(|_| Challenge::from(rng.gen::<u128>()))
                    .collect_vec();
                let nonzero_indices: Vec<Option<u16>> = (0..t)
                    .map(|_| Some((rng.gen::<u64>() % k as u64) as u16))
                    .collect();
                let one_hot = OneHotPolynomial::<Fr>::from_indices(nonzero_indices, k);
                let eval = one_hot.evaluate(&point);
                let poly_wrapped = MultilinearPolynomial::OneHot(one_hot.clone());
                let commitment = HyperKZG::commit(&poly_wrapped, &pk).0;

                let (r_address, r_cycle) = point.split_at(log_k);
                prover_opening_accumulator.append_sparse(
                    &mut prover_tr,
                    vec![CommittedPoly::NodeOutputRaD(i, 0)],
                    SumcheckId::NodeExecution(0),
                    r_address.to_vec(),
                    r_cycle.to_vec(),
                    vec![eval],
                );
                OneHotPolyData {
                    poly: one_hot,
                    commitment,
                    point,
                }
            })
            .collect();

        // Combine all polynomials for RLC
        let mut all_polys: Vec<(CommittedPoly, MultilinearPolynomial<Fr>)> = dense_polys
            .iter()
            .enumerate()
            .map(|(i, data)| (CommittedPoly::DivNodeQuotient(i), data.poly.clone()))
            .collect();
        all_polys.extend(oh_polys.iter().enumerate().map(|(i, data)| {
            (
                CommittedPoly::NodeOutputRaD(i, 0),
                MultilinearPolynomial::OneHot(data.poly.clone()),
            )
        }));

        // Prepare sumcheck
        let polynomial_map = BTreeMap::from_iter(all_polys);

        prover_opening_accumulator.prepare_for_sumcheck(&polynomial_map, &mut prover_tr);

        // Run sumcheck
        let (accumulator_sumcheck_proof, r_sumcheck_acc) =
            prover_opening_accumulator.prove_batch_opening_sumcheck(&mut prover_tr);

        // Finalize sumcheck (uses claims cached via cache_openings, derives gamma, cleans up)
        let state = prover_opening_accumulator
            .finalize_batch_opening_sumcheck(r_sumcheck_acc.clone(), &mut prover_tr);
        let sumcheck_claims: Vec<Fr> = state.sumcheck_claims.clone();

        // Build RLC
        let rlc = build_materialized_rlc(&state.poly_coeffs, &polynomial_map);

        // Prove
        let eval_proof: HyperKZGProof<Bn254> =
            HyperKZG::open(&pk, &rlc, &state.r_sumcheck, &mut prover_tr).unwrap();

        // Verify
        let mut verifier_tr = Blake2bTranscript::new(b"TestEval");
        verifier_tr.compare_to(prover_tr);
        let mut verifier_opening_accumulator: VerifierOpeningAccumulator<Fr> =
            VerifierOpeningAccumulator::new();
        // Take claims
        for (key, (_, value)) in &prover_opening_accumulator.openings {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::default();
            verifier_opening_accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }

        dense_polys.iter().enumerate().for_each(|(i, data)| {
            let id = OpeningId::new(
                CommittedPoly::DivNodeQuotient(i),
                SumcheckId::NodeExecution(0),
            );
            verifier_opening_accumulator.append_dense(&mut verifier_tr, id, data.point.clone());
        });

        oh_polys.iter().enumerate().for_each(|(i, data)| {
            verifier_opening_accumulator.append_sparse(
                &mut verifier_tr,
                vec![CommittedPoly::NodeOutputRaD(i, 0)],
                SumcheckId::NodeExecution(0),
                data.point.clone(),
            );
        });

        // Prepare - populate sumcheck claims
        verifier_opening_accumulator
            .prepare_for_sumcheck(&sumcheck_claims, &mut verifier_tr)
            .unwrap();

        // Verify sumcheck
        let r_sumcheck = verifier_opening_accumulator
            .verify_batch_opening_sumcheck(&accumulator_sumcheck_proof, &mut verifier_tr)
            .unwrap();

        // Finalize and store state in accumulator for Stage 8
        let verifier_state = verifier_opening_accumulator.finalize_batch_opening_sumcheck(
            r_sumcheck,
            &sumcheck_claims,
            &mut verifier_tr,
        );

        let mut commitments_map: BTreeMap<CommittedPoly, HyperKZGCommitment<Bn254>> =
            BTreeMap::new();
        dense_polys.iter().enumerate().for_each(|(i, data)| {
            commitments_map.insert(CommittedPoly::DivNodeQuotient(i), data.commitment.clone());
        });
        oh_polys.iter().enumerate().for_each(|(i, data)| {
            commitments_map.insert(CommittedPoly::NodeOutputRaD(i, 0), data.commitment.clone());
        });
        // Compute joint commitment
        let joint_commitment = VerifierOpeningAccumulator::compute_joint_commitment::<
            HyperKZG<Bn254>,
        >(&mut commitments_map, &verifier_state);

        // Verify joint opening
        verifier_opening_accumulator
            .verify_joint_opening::<_, HyperKZG<Bn254>>(
                &vk,
                &eval_proof,
                &joint_commitment,
                &verifier_state,
                &mut verifier_tr,
            )
            .unwrap();
    }

    #[test]
    fn test_mix() {
        struct PolyData {
            poly: MultilinearPolynomial<Fr>,
            commitment: HyperKZGCommitment<Bn254>,
            point: Vec<Challenge>,
        }

        struct OneHotPolyData {
            poly: OneHotPolynomial<Fr>,
            commitment: HyperKZGCommitment<Bn254>,
            point: Vec<Challenge>,
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0x76543);

        // Define dense polynomial sizes (log_N values)
        let dense_configs = [5, 6, 7];

        // Define OneHot polynomial parameters (log_K, log_T)
        let oh_configs = [(8, 4), (5, 6), (4, 7)]; // (K, T) pairs

        // Calculate max size needed
        let max_dense_size = 1 << dense_configs.iter().max().unwrap();
        let max_oh_size = oh_configs
            .iter()
            .map(|(log_k, log_t)| (1 << log_k) * (1 << log_t))
            .max()
            .unwrap();
        let max_size = max_dense_size.max(max_oh_size);

        // Setup
        let srs = HyperKZGSRS::setup(&mut rng, max_size);
        let (pk, vk): (HyperKZGProverKey<Bn254>, HyperKZGVerifierKey<Bn254>) = srs.trim(max_size);
        let mut prover_tr = Blake2bTranscript::new(b"TestEval");
        let mut prover_opening_accumulator: ProverOpeningAccumulator<Fr> =
            ProverOpeningAccumulator::new();

        // Create dense polynomials
        let dense_polys: Vec<PolyData> = dense_configs
            .iter()
            .enumerate()
            .map(|(i, &log_n)| {
                let n = 1 << log_n;
                // Generate evaluation point based on largest polynomial
                let point: Vec<Challenge> = (0..log_n)
                    .map(|_| Challenge::from(rng.gen::<u128>()))
                    .collect();
                let raw: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
                let poly = MultilinearPolynomial::from(raw);
                let eval = poly.evaluate(&point);
                let commitment = HyperKZG::commit(&poly, &pk).0;
                let id = OpeningId::new(
                    CommittedPoly::DivNodeQuotient(i),
                    SumcheckId::NodeExecution(0),
                );
                prover_opening_accumulator.append_dense(&mut prover_tr, id, point.clone(), eval);
                PolyData {
                    poly,
                    commitment,
                    point,
                }
            })
            .collect();

        // Create OneHot polynomials
        let oh_polys: Vec<OneHotPolyData> = oh_configs
            .iter()
            .enumerate()
            .map(|(i, &(log_k, log_t))| {
                let k = 1 << log_k;
                let t = 1 << log_t;
                let num_vars = log_k + log_t;
                let point = (0..num_vars)
                    .map(|_| Challenge::from(rng.gen::<u128>()))
                    .collect_vec();
                let nonzero_indices: Vec<Option<u16>> = (0..t)
                    .map(|_| Some((rng.gen::<u64>() % k as u64) as u16))
                    .collect();
                let one_hot = OneHotPolynomial::<Fr>::from_indices(nonzero_indices, k);
                let eval = one_hot.evaluate(&point);
                let poly_wrapped = MultilinearPolynomial::OneHot(one_hot.clone());
                let commitment = HyperKZG::commit(&poly_wrapped, &pk).0;

                let (r_address, r_cycle) = point.split_at(log_k);
                prover_opening_accumulator.append_sparse(
                    &mut prover_tr,
                    vec![CommittedPoly::NodeOutputRaD(i, 0)],
                    SumcheckId::NodeExecution(0),
                    r_address.to_vec(),
                    r_cycle.to_vec(),
                    vec![eval],
                );
                OneHotPolyData {
                    poly: one_hot,
                    commitment,
                    point,
                }
            })
            .collect();

        // Combine all polynomials for RLC
        let mut all_polys: Vec<(CommittedPoly, MultilinearPolynomial<Fr>)> = dense_polys
            .iter()
            .enumerate()
            .map(|(i, data)| (CommittedPoly::DivNodeQuotient(i), data.poly.clone()))
            .collect();
        all_polys.extend(oh_polys.iter().enumerate().map(|(i, data)| {
            (
                CommittedPoly::NodeOutputRaD(i, 0),
                MultilinearPolynomial::OneHot(data.poly.clone()),
            )
        }));

        // Prepare sumcheck
        let polynomial_map = BTreeMap::from_iter(all_polys);

        prover_opening_accumulator.prepare_for_sumcheck(&polynomial_map, &mut prover_tr);

        // Run sumcheck
        let (accumulator_sumcheck_proof, r_sumcheck_acc) =
            prover_opening_accumulator.prove_batch_opening_sumcheck(&mut prover_tr);

        // Finalize sumcheck (uses claims cached via cache_openings, derives gamma, cleans up)
        let state = prover_opening_accumulator
            .finalize_batch_opening_sumcheck(r_sumcheck_acc.clone(), &mut prover_tr);
        let sumcheck_claims: Vec<Fr> = state.sumcheck_claims.clone();

        // Build RLC
        let rlc = build_materialized_rlc(&state.poly_coeffs, &polynomial_map);

        // Prove
        let eval_proof: HyperKZGProof<Bn254> =
            HyperKZG::open(&pk, &rlc, &state.r_sumcheck, &mut prover_tr).unwrap();

        // Verify
        let mut verifier_tr = Blake2bTranscript::new(b"TestEval");
        verifier_tr.compare_to(prover_tr);
        let mut verifier_opening_accumulator: VerifierOpeningAccumulator<Fr> =
            VerifierOpeningAccumulator::new();
        // Take claims
        for (key, (_, value)) in &prover_opening_accumulator.openings {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::default();
            verifier_opening_accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }

        dense_polys.iter().enumerate().for_each(|(i, data)| {
            let id = OpeningId::new(
                CommittedPoly::DivNodeQuotient(i),
                SumcheckId::NodeExecution(0),
            );
            verifier_opening_accumulator.append_dense(&mut verifier_tr, id, data.point.clone());
        });

        oh_polys.iter().enumerate().for_each(|(i, data)| {
            verifier_opening_accumulator.append_sparse(
                &mut verifier_tr,
                vec![CommittedPoly::NodeOutputRaD(i, 0)],
                SumcheckId::NodeExecution(0),
                data.point.clone(),
            );
        });

        // Prepare - populate sumcheck claims
        verifier_opening_accumulator
            .prepare_for_sumcheck(&sumcheck_claims, &mut verifier_tr)
            .unwrap();

        // Verify sumcheck
        let r_sumcheck = verifier_opening_accumulator
            .verify_batch_opening_sumcheck(&accumulator_sumcheck_proof, &mut verifier_tr)
            .unwrap();

        // Finalize and store state in accumulator for Stage 8
        let verifier_state = verifier_opening_accumulator.finalize_batch_opening_sumcheck(
            r_sumcheck,
            &sumcheck_claims,
            &mut verifier_tr,
        );

        let mut commitments_map: BTreeMap<CommittedPoly, HyperKZGCommitment<Bn254>> =
            BTreeMap::new();
        dense_polys.iter().enumerate().for_each(|(i, data)| {
            commitments_map.insert(CommittedPoly::DivNodeQuotient(i), data.commitment.clone());
        });
        oh_polys.iter().enumerate().for_each(|(i, data)| {
            commitments_map.insert(CommittedPoly::NodeOutputRaD(i, 0), data.commitment.clone());
        });
        // Compute joint commitment
        let joint_commitment = VerifierOpeningAccumulator::compute_joint_commitment::<
            HyperKZG<Bn254>,
        >(&mut commitments_map, &verifier_state);

        // Verify joint opening
        verifier_opening_accumulator
            .verify_joint_opening::<_, HyperKZG<Bn254>>(
                &vk,
                &eval_proof,
                &joint_commitment,
                &verifier_state,
                &mut verifier_tr,
            )
            .unwrap();
    }
}
