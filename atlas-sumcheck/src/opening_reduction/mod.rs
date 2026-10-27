//! Opening reduction sumcheck prover and verifier.
//!
//! This module contains the sumcheck-specific logic for the batch opening reduction protocol.
//! The higher-level orchestration remains in `poly/opening_proof.rs`.

use crate::{
    opening_proof::{
        Opening, OpeningAccumulator, OpeningId, SumcheckId, VerifierOpeningAccumulator,
    },
    sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_std::Zero;
use atlas_algebra::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_point::{OpeningPoint, BIG_ENDIAN},
        ra_poly::RaPolynomial,
    },
    utils::{expanding_table::ExpandingTable, math::Math},
};
use atlas_transcript::Transcript;
use common::parallel::par_enabled;
use common::CommittedPoly;
use jolt_poly::GruenSplitEqPolynomial;
use rayon::prelude::*;
use std::ops::Mul;

/// Degree of the sumcheck round polynomials in opening reduction.
pub const OPENING_SUMCHECK_DEGREE: usize = 2;

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
}

/// `Σ_j eq(j)·value(j)` over the unbound half of a high-to-low split eq: `j` runs over the
/// first half of the remaining hypercube (the current variable set to 0), weighted by the
/// unbound `E_out`/`E_in` tables.
fn fold_high_to_low<F: JoltField>(
    gruen_eq: &GruenSplitEqPolynomial<F>,
    value: impl Fn(usize) -> F + Sync + Send,
) -> F {
    let d_e_out = gruen_eq.e_out_current();
    if gruen_eq.e_in_current_len() <= 1 {
        // E_in is fully bound
        let unreduced = (0..d_e_out.len())
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|j| d_e_out[j].mul_to_product_accum(value(j)))
            .reduce(F::UnreducedProductAccum::zero, |running, new| running + new);
        F::reduce_product_accum(unreduced)
    } else {
        let d_e_in = gruen_eq.e_in_current();
        let num_x_out_bits = d_e_out.len().log_2();
        (0..d_e_in.len())
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|x_in| {
                let unreduced_inner_sum = (0..d_e_out.len())
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|x_out| {
                        let j = (x_in << num_x_out_bits) | x_out;
                        d_e_out[x_out].mul_to_product_accum(value(j))
                    })
                    .reduce(F::UnreducedProductAccum::zero, |running, new| running + new);
                d_e_in[x_in] * F::reduce_product_accum(unreduced_inner_sum)
            })
            .sum()
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
        _accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
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
    #[allocative(skip)]
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
        let r_cycle: Vec<F> = r_cycle.iter().map(|&r| r.into()).collect();
        let D = GruenSplitEqPolynomial::new(&r_cycle, jolt_poly::BindingOrder::HighToLow);
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

#[cfg(test)]
mod tests {
    use crate::opening_proof::{
        prover::ProverOpeningAccumulator, OpeningId, SumcheckId, VerifierOpeningAccumulator,
    };
    use ark_std::UniformRand;
    use atlas_algebra::field::fp128::Fp128 as Fr;
    use atlas_algebra::{
        field::JoltField,
        poly::{
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            one_hot_polynomial::OneHotPolynomial,
            opening_point::{OpeningPoint, BIG_ENDIAN},
            rlc_polynomial::build_materialized_rlc,
        },
    };
    use atlas_transcript::{
        Blake2bProver, Blake2bVerifier, ProvingTranscript, Transcript, VerifyingTranscript,
    };
    use common::CommittedPoly;
    use itertools::Itertools;
    use rand::{Rng, SeedableRng};
    use std::collections::BTreeMap;

    type Challenge = <Fr as JoltField>::Challenge;

    #[test]
    fn test_3_dense() {
        let log_T = 6;
        struct PolyData {
            poly: MultilinearPolynomial<Fr>,
            point: Vec<Challenge>,
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0x76543);

        // Define dense polynomial sizes (log_N values)
        let dense_configs = [log_T, log_T, log_T];

        let mut prover_tr = Blake2bProver::new(b"TestEval");
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
                let id = OpeningId::new(
                    CommittedPoly::DivNodeQuotient(i),
                    SumcheckId::NodeExecution(0),
                );
                prover_opening_accumulator.append_dense(&mut prover_tr, id, point.clone(), eval);
                PolyData { poly, point }
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

        // Verify
        let mut verifier_tr = Blake2bVerifier::new(b"TestEval", prover_tr.narg().to_vec());
        verifier_tr.compare_to(prover_tr.state_history().to_vec());
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

        // The reduced claim is the RLC of the committed polynomials at the sumcheck point.
        assert_eq!(
            rlc.evaluate(&verifier_state.r_sumcheck),
            verifier_opening_accumulator.compute_joint_claim::<Blake2bVerifier>(&verifier_state)
        );
    }

    #[test]
    fn test_3_dense_oh() {
        let log_T = 6;
        struct PolyData {
            poly: MultilinearPolynomial<Fr>,
            point: Vec<Challenge>,
        }

        struct OneHotPolyData {
            poly: OneHotPolynomial<Fr>,
            point: Vec<Challenge>,
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0x76543);

        // Define dense polynomial sizes (log_N values)
        let dense_configs = [log_T, log_T, log_T];

        // Define OneHot polynomial parameters (log_K, log_T)
        let oh_configs = [(5, log_T), (5, log_T), (5, log_T)]; // (K, T) pairs

        let mut prover_tr = Blake2bProver::new(b"TestEval");
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
                let id = OpeningId::new(
                    CommittedPoly::DivNodeQuotient(i),
                    SumcheckId::NodeExecution(0),
                );
                prover_opening_accumulator.append_dense(&mut prover_tr, id, point.clone(), eval);
                PolyData { poly, point }
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

        // Verify
        let mut verifier_tr = Blake2bVerifier::new(b"TestEval", prover_tr.narg().to_vec());
        verifier_tr.compare_to(prover_tr.state_history().to_vec());
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

        // The reduced claim is the RLC of the committed polynomials at the sumcheck point.
        assert_eq!(
            rlc.evaluate(&verifier_state.r_sumcheck),
            verifier_opening_accumulator.compute_joint_claim::<Blake2bVerifier>(&verifier_state)
        );
    }

    #[test]
    fn test_mix() {
        struct PolyData {
            poly: MultilinearPolynomial<Fr>,
            point: Vec<Challenge>,
        }

        struct OneHotPolyData {
            poly: OneHotPolynomial<Fr>,
            point: Vec<Challenge>,
        }

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0x76543);

        // Define dense polynomial sizes (log_N values)
        let dense_configs = [5, 6, 7];

        // Define OneHot polynomial parameters (log_K, log_T)
        let oh_configs = [(8, 4), (5, 6), (4, 7)]; // (K, T) pairs

        let mut prover_tr = Blake2bProver::new(b"TestEval");
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
                let id = OpeningId::new(
                    CommittedPoly::DivNodeQuotient(i),
                    SumcheckId::NodeExecution(0),
                );
                prover_opening_accumulator.append_dense(&mut prover_tr, id, point.clone(), eval);
                PolyData { poly, point }
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

        // Verify
        let mut verifier_tr = Blake2bVerifier::new(b"TestEval", prover_tr.narg().to_vec());
        verifier_tr.compare_to(prover_tr.state_history().to_vec());
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

        // The reduced claim is the RLC of the committed polynomials at the sumcheck point.
        assert_eq!(
            rlc.evaluate(&verifier_state.r_sumcheck),
            verifier_opening_accumulator.compute_joint_claim::<Blake2bVerifier>(&verifier_state)
        );
    }
}

pub mod prover;
