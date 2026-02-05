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
            Opening, OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
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
use common::CommittedPolynomial;
use rayon::prelude::*;
use std::{
    collections::{BTreeMap, HashMap},
    mem,
    sync::{Arc, RwLock},
};

/// Degree of the sumcheck round polynomials in opening reduction.
pub const OPENING_SUMCHECK_DEGREE: usize = 2;

/// Prover state for a single opening in the batch opening reduction sumcheck.
#[derive(Clone, Allocative)]
pub struct OpeningProofReductionSumcheckProver<F>
where
    F: JoltField,
{
    pub prover_state: ProverOpening<F>,
    /// Represents the polynomial opened.
    pub polynomial: CommittedPolynomial,
    /// The ID of the sumcheck these openings originated from
    pub sumcheck_id: SumcheckId,
    pub opening: Opening<F>,
    pub sumcheck_claim: Option<F>,
}

impl<F> OpeningProofReductionSumcheckProver<F>
where
    F: JoltField,
{
    pub fn new_dense(
        polynomial: CommittedPolynomial,
        sumcheck_id: SumcheckId,
        eq_poly: Arc<RwLock<EqCycleState<F>>>,
        opening_point: Vec<F::Challenge>,
        claim: F,
    ) -> Self {
        let opening = DensePolynomialProverOpening {
            polynomial: None, // Defer initialization until opening proof reduction sumcheck
            eq_poly,
        };
        Self {
            polynomial,
            sumcheck_id,
            opening: (opening_point.into(), claim),
            prover_state: opening.into(),
            sumcheck_claim: None,
        }
    }

    pub fn new_one_hot(
        polynomial: CommittedPolynomial,
        sumcheck_id: SumcheckId,
        eq_address: Arc<RwLock<EqAddressState<F>>>,
        eq_cycle: Arc<RwLock<EqCycleState<F>>>,
        opening_point: Vec<F::Challenge>,
        claim: F,
    ) -> Self {
        let opening = OneHotPolynomialProverOpening::new(eq_address, eq_cycle);
        Self {
            polynomial,
            sumcheck_id,
            opening: (opening_point.into(), claim),
            prover_state: opening.into(),
            sumcheck_claim: None,
        }
    }

    #[tracing::instrument(skip_all, name = "OpeningProofReductionSumcheck::prepare_sumcheck")]
    pub fn prepare_sumcheck(
        &mut self,
        polynomials_map: &BTreeMap<CommittedPolynomial, MultilinearPolynomial<F>>,
        shared_dense_polynomials: &HashMap<
            CommittedPolynomial,
            Arc<RwLock<SharedDensePolynomial<F>>>,
        >,
    ) {
        #[cfg(test)]
        {
            use crate::poly::multilinear_polynomial::PolynomialEvaluation;
            let poly = polynomials_map.get(&self.polynomial).unwrap();
            debug_assert_eq!(
                poly.evaluate(&self.opening.0.r),
                self.opening.1,
                "Evaluation mismatch for {:?} {:?}",
                self.sumcheck_id,
                self.polynomial,
            );
            let num_vars = poly.get_num_vars();
            let opening_point_len = self.opening.0.len();
            debug_assert_eq!(
                num_vars,
                opening_point_len,
                "{:?} has {num_vars} variables but opening point from {:?} has length {opening_point_len}",
                self.polynomial,
                self.sumcheck_id,
            );
        }

        match &mut self.prover_state {
            ProverOpening::Dense(opening) => {
                let poly = shared_dense_polynomials.get(&self.polynomial).unwrap();
                opening.polynomial = Some(poly.clone());
            }
            ProverOpening::OneHot(opening) => {
                let poly = polynomials_map.get(&self.polynomial).unwrap();
                if let MultilinearPolynomial::OneHot(one_hot) = poly {
                    opening.initialize(one_hot.clone());
                } else {
                    panic!("Unexpected non-one-hot polynomial")
                }
            }
        };
    }

    pub fn cache_sumcheck_claim(&mut self) {
        debug_assert!(self.sumcheck_claim.is_none());
        let claim = match &mut self.prover_state {
            ProverOpening::Dense(opening) => opening.final_sumcheck_claim(),
            ProverOpening::OneHot(opening) => opening.final_sumcheck_claim(),
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

    fn normalize_opening_point(
        &self,
        _: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        unimplemented!("Unused")
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
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // Cache the final sumcheck claim in the accumulator
        let claim = match &self.prover_state {
            ProverOpening::Dense(opening) => opening.final_sumcheck_claim(),
            ProverOpening::OneHot(opening) => opening.final_sumcheck_claim(),
        };
        accumulator.cache_opening_reduction_claim(self.polynomial, claim);
    }
}

/// Verifier state for a single opening in the batch opening reduction sumcheck.
pub struct OpeningProofReductionSumcheckVerifier<F>
where
    F: JoltField,
{
    /// Represents the polynomial opened.
    pub polynomial: CommittedPolynomial,
    opening: Opening<F>,
    pub sumcheck_claim: Option<F>,
}

impl<F: JoltField> OpeningProofReductionSumcheckVerifier<F> {
    pub fn new(
        polynomial: CommittedPolynomial,
        opening_point: Vec<F::Challenge>,
        input_claim: F,
    ) -> Self {
        Self {
            polynomial,
            opening: (opening_point.into(), input_claim),
            sumcheck_claim: None,
        }
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
        _accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // Nothing to cache.
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
                .map(|x_in| {
                    let unreduced_inner_sum = (0..num_x_out)
                        .into_par_iter()
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

    pub fn final_sumcheck_claim(&self) -> F {
        let poly_ref = self.polynomial.as_ref().unwrap();
        poly_ref.read().unwrap().poly.final_sumcheck_claim()
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
    pub fn new(r_address: &[F::Challenge]) -> Self {
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
    pub fn new(r_cycle: &[F::Challenge]) -> Self {
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
#[derive(Clone, Allocative)]
pub struct OneHotPolynomialProverOpening<F: JoltField> {
    pub log_T: usize,
    pub polynomial: OneHotPolynomial<F>,
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
            polynomial: OneHotPolynomial::default(),
            eq_address_state,
            eq_cycle_state,
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPolynomialProverOpening::initialize")]
    pub fn initialize(&mut self, mut polynomial: OneHotPolynomial<F>) {
        let nonzero_indices = &polynomial.nonzero_indices;
        let T = nonzero_indices.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = (T / num_chunks).max(1);

        let eq = self.eq_cycle_state.read().unwrap();
        let D_coeffs_for_G = &eq.D.merge();

        // Compute G as described in Section 6.3
        let G = nonzero_indices
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, chunk)| {
                let mut result = unsafe_allocate_zero_vec(polynomial.K);
                let mut j = chunk_index * chunk_size;
                for k in chunk {
                    if let Some(k) = k {
                        result[*k as usize] += D_coeffs_for_G[j];
                    }
                    j += 1;
                }
                result
            })
            .reduce(
                || unsafe_allocate_zero_vec(polynomial.K),
                |mut running, new| {
                    running
                        .par_iter_mut()
                        .zip(new.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running
                },
            );

        polynomial.G = G;
        self.polynomial = polynomial;
        self.log_T = T.log_2();
    }

    #[tracing::instrument(skip_all, name = "OneHotPolynomialProverOpening::compute_message")]
    pub fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let shared_eq_address = self.eq_address_state.read().unwrap();
        let shared_eq_cycle = self.eq_cycle_state.read().unwrap();
        let polynomial = &self.polynomial;

        if round < polynomial.K.log_2() {
            let num_unbound_address_variables = polynomial.K.log_2() - round;
            let B = &shared_eq_address.B;
            let F = &shared_eq_address.F;
            let G = &polynomial.G;

            let unreduced_univariate_poly_evals = (0..B.len() / 2)
                .into_par_iter()
                .map(|k_prime| {
                    let B_evals = B.sumcheck_evals_array::<2>(k_prime, BindingOrder::HighToLow);
                    let inner_sum = G
                        .par_iter()
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
            let eq_r_address_claim = B.final_sumcheck_claim();
            let H = &polynomial.H.read().unwrap();

            let gruen_eval_0 = if d_gruen.E_in_current_len() == 1 {
                let unreduced_gruen_eval_0 = (0..d_gruen.len() / 2)
                    .into_par_iter()
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
                    .map(|x_in| {
                        let unreduced_inner_sum = (0..num_x_out)
                            .into_par_iter()
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
        let polynomial = &mut self.polynomial;
        let num_variables_bound =
            shared_eq_address.num_variables_bound + shared_eq_cycle.num_variables_bound;

        // Bind shared state if not already bound
        if num_variables_bound <= round {
            if round < polynomial.K.log_2() {
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
        if round == polynomial.K.log_2() - 1 {
            let nonzero_indices = &polynomial.nonzero_indices;

            let mut lock = polynomial.H.write().unwrap();
            if matches!(*lock, RaPolynomial::None) {
                *lock =
                    RaPolynomial::new(nonzero_indices.clone(), shared_eq_address.F.clone_values());
            }

            let g = mem::take(&mut polynomial.G);
            drop_in_background_thread(g);
        } else if round >= polynomial.K.log_2() {
            // Bind H for subsequent T rounds
            let mut H = polynomial.H.write().unwrap();
            if H.len().log_2() == self.log_T + polynomial.K.log_2() - round {
                H.bind_parallel(r, BindingOrder::HighToLow);
            }
        }
    }

    pub fn final_sumcheck_claim(&self) -> F {
        self.polynomial.H.read().unwrap().final_sumcheck_claim()
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
                OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
                BIG_ENDIAN,
            },
            rlc_polynomial::build_materialized_rlc,
        },
        transcripts::{Blake2bTranscript, Transcript},
    };
    use ark_bn254::Bn254;
    use ark_ec::pairing::Pairing;
    use ark_std::UniformRand;
    use common::CommittedPolynomial;
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
                prover_opening_accumulator.append_dense(
                    &mut prover_tr,
                    CommittedPolynomial::DivNodeQuotient(i),
                    SumcheckId::Execution,
                    point.clone(),
                    eval,
                );
                PolyData {
                    poly,
                    commitment,
                    point,
                }
            })
            .collect();

        // Combine all polynomials for RLC
        let all_polys: Vec<(CommittedPolynomial, MultilinearPolynomial<Fr>)> = dense_polys
            .iter()
            .enumerate()
            .map(|(i, data)| (CommittedPolynomial::DivNodeQuotient(i), data.poly.clone()))
            .collect();

        // Prepare sumcheck
        let polynomial_map = BTreeMap::from_iter(all_polys);

        prover_opening_accumulator.prepare_for_sumcheck(&polynomial_map);

        // Run sumcheck
        let (accumulator_sumcheck_proof, r_sumcheck_acc) =
            prover_opening_accumulator.prove_batch_opening_sumcheck(&mut prover_tr);

        // Finalize sumcheck (uses claims cached via cache_openings, derives gamma, cleans up)
        let state = prover_opening_accumulator
            .finalize_batch_opening_sumcheck(r_sumcheck_acc.clone(), &mut prover_tr);
        let sumcheck_claims: Vec<Fr> = state.sumcheck_claims.clone();

        // Build RLC
        let rlc = build_materialized_rlc(&state.gamma_powers, &polynomial_map);

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
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier_opening_accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }

        dense_polys.iter().enumerate().for_each(|(i, data)| {
            verifier_opening_accumulator.append_dense(
                &mut verifier_tr,
                CommittedPolynomial::DivNodeQuotient(i), // ID doesn't matter for verification
                SumcheckId::Execution,
                data.point.clone(),
            );
        });

        // Prepare - populate sumcheck claims
        verifier_opening_accumulator.prepare_for_sumcheck(&sumcheck_claims);

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

        let mut commitments_map: BTreeMap<CommittedPolynomial, HyperKZGCommitment<Bn254>> =
            BTreeMap::new();
        dense_polys.iter().enumerate().for_each(|(i, data)| {
            commitments_map.insert(
                CommittedPolynomial::DivNodeQuotient(i),
                data.commitment.clone(),
            );
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
                prover_opening_accumulator.append_dense(
                    &mut prover_tr,
                    CommittedPolynomial::DivNodeQuotient(i),
                    SumcheckId::Execution,
                    point.clone(),
                    eval,
                );
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
                    vec![CommittedPolynomial::NodeOutputRaD(i, 0)],
                    SumcheckId::Execution,
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
        let mut all_polys: Vec<(CommittedPolynomial, MultilinearPolynomial<Fr>)> = dense_polys
            .iter()
            .enumerate()
            .map(|(i, data)| (CommittedPolynomial::DivNodeQuotient(i), data.poly.clone()))
            .collect();
        all_polys.extend(oh_polys.iter().enumerate().map(|(i, data)| {
            (
                CommittedPolynomial::NodeOutputRaD(i, 0),
                MultilinearPolynomial::OneHot(data.poly.clone()),
            )
        }));

        // Prepare sumcheck
        let polynomial_map = BTreeMap::from_iter(all_polys);

        prover_opening_accumulator.prepare_for_sumcheck(&polynomial_map);

        // Run sumcheck
        let (accumulator_sumcheck_proof, r_sumcheck_acc) =
            prover_opening_accumulator.prove_batch_opening_sumcheck(&mut prover_tr);

        // Finalize sumcheck (uses claims cached via cache_openings, derives gamma, cleans up)
        let state = prover_opening_accumulator
            .finalize_batch_opening_sumcheck(r_sumcheck_acc.clone(), &mut prover_tr);
        let sumcheck_claims: Vec<Fr> = state.sumcheck_claims.clone();

        // Build RLC
        let rlc = build_materialized_rlc(&state.gamma_powers, &polynomial_map);

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
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier_opening_accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }

        dense_polys.iter().enumerate().for_each(|(i, data)| {
            verifier_opening_accumulator.append_dense(
                &mut verifier_tr,
                CommittedPolynomial::DivNodeQuotient(i), // ID doesn't matter for verification
                SumcheckId::Execution,
                data.point.clone(),
            );
        });

        oh_polys.iter().enumerate().for_each(|(i, data)| {
            verifier_opening_accumulator.append_sparse(
                &mut verifier_tr,
                vec![CommittedPolynomial::NodeOutputRaD(i, 0)],
                SumcheckId::Execution,
                data.point.clone(),
            );
        });

        // Prepare - populate sumcheck claims
        verifier_opening_accumulator.prepare_for_sumcheck(&sumcheck_claims);

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

        let mut commitments_map: BTreeMap<CommittedPolynomial, HyperKZGCommitment<Bn254>> =
            BTreeMap::new();
        dense_polys.iter().enumerate().for_each(|(i, data)| {
            commitments_map.insert(
                CommittedPolynomial::DivNodeQuotient(i),
                data.commitment.clone(),
            );
        });
        oh_polys.iter().enumerate().for_each(|(i, data)| {
            commitments_map.insert(
                CommittedPolynomial::NodeOutputRaD(i, 0),
                data.commitment.clone(),
            );
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
                prover_opening_accumulator.append_dense(
                    &mut prover_tr,
                    CommittedPolynomial::DivNodeQuotient(i),
                    SumcheckId::Execution,
                    point.clone(),
                    eval,
                );
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
                    vec![CommittedPolynomial::NodeOutputRaD(i, 0)],
                    SumcheckId::Execution,
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
        let mut all_polys: Vec<(CommittedPolynomial, MultilinearPolynomial<Fr>)> = dense_polys
            .iter()
            .enumerate()
            .map(|(i, data)| (CommittedPolynomial::DivNodeQuotient(i), data.poly.clone()))
            .collect();
        all_polys.extend(oh_polys.iter().enumerate().map(|(i, data)| {
            (
                CommittedPolynomial::NodeOutputRaD(i, 0),
                MultilinearPolynomial::OneHot(data.poly.clone()),
            )
        }));

        // Prepare sumcheck
        let polynomial_map = BTreeMap::from_iter(all_polys);

        prover_opening_accumulator.prepare_for_sumcheck(&polynomial_map);

        // Run sumcheck
        let (accumulator_sumcheck_proof, r_sumcheck_acc) =
            prover_opening_accumulator.prove_batch_opening_sumcheck(&mut prover_tr);

        // Finalize sumcheck (uses claims cached via cache_openings, derives gamma, cleans up)
        let state = prover_opening_accumulator
            .finalize_batch_opening_sumcheck(r_sumcheck_acc.clone(), &mut prover_tr);
        let sumcheck_claims: Vec<Fr> = state.sumcheck_claims.clone();

        // Build RLC
        let rlc = build_materialized_rlc(&state.gamma_powers, &polynomial_map);

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
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier_opening_accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }

        dense_polys.iter().enumerate().for_each(|(i, data)| {
            verifier_opening_accumulator.append_dense(
                &mut verifier_tr,
                CommittedPolynomial::DivNodeQuotient(i), // ID doesn't matter for verification
                SumcheckId::Execution,
                data.point.clone(),
            );
        });

        oh_polys.iter().enumerate().for_each(|(i, data)| {
            verifier_opening_accumulator.append_sparse(
                &mut verifier_tr,
                vec![CommittedPolynomial::NodeOutputRaD(i, 0)],
                SumcheckId::Execution,
                data.point.clone(),
            );
        });

        // Prepare - populate sumcheck claims
        verifier_opening_accumulator.prepare_for_sumcheck(&sumcheck_claims);

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

        let mut commitments_map: BTreeMap<CommittedPolynomial, HyperKZGCommitment<Bn254>> =
            BTreeMap::new();
        dense_polys.iter().enumerate().for_each(|(i, data)| {
            commitments_map.insert(
                CommittedPolynomial::DivNodeQuotient(i),
                data.commitment.clone(),
            );
        });
        oh_polys.iter().enumerate().for_each(|(i, data)| {
            commitments_map.insert(
                CommittedPolynomial::NodeOutputRaD(i, 0),
                data.commitment.clone(),
            );
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
