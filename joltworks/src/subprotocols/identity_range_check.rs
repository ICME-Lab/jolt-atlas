use crate::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        prefix_suffix::{Prefix, PrefixRegistry, PrefixSuffixDecomposition},
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{
        expanding_table::ExpandingTable, lookup_bits::LookupBits, math::Math,
        thread::drop_in_background_thread,
    },
};
use ark_std::Zero;
use common::parallel::par_enabled;
use common::VirtualPoly;
use itertools::Itertools;
use rayon::prelude::*;
use std::array;

const DEGREE_BOUND: usize = 2;

#[derive(Debug, Clone)]
/// Parameters for the Prefix suffix Read-raf checking sum-check protocol.
///
/// This protocol proves correct lookups from instruction tables by verifying that
/// read values match the table entries at computed addresses. The protocol uses
/// gamma batching to efficiently combine the lookup output with left/right operand
/// contributions and RAF values that depend on operand interleaving.
///
/// The sumcheck proceeds in two phases:
/// - Address phase (log K rounds): binds address variables using prefix-suffix decomposition
/// - Cycle phase (log T rounds): binds cycle variables and evaluates equality polynomials
pub struct IdentityRCParams<F>
where
    F: JoltField,
{
    /// log2(T): number of variables in the node output polynomial (cycle variables).
    pub log_T: usize,
    /// log2(K): number of variables in the address polynomial (address variables).
    pub log_K: usize,
    /// Opening point for the node output polynomial (r_reduction).
    pub r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    /// Polynomial types for opening accumulator (to cache ra claim)
    pub polynomial_type: VirtualPoly,
    /// Sumcheck ID for opening accumulator (to cache ra claim)
    pub sumcheck_id: SumcheckId,
    /// Input claim for the read-checking sum-check
    pub input_claim: F,
    /// Number of phases in the address rounds (first log K rounds).
    pub phases: usize,
}

impl<F> IdentityRCParams<F>
where
    F: JoltField,
{
    pub fn new(
        provider: &impl IdentityRCProvider<F>,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let r_node_output = provider.r_cycle(accumulator);
        let log_T = r_node_output.len();
        let (polynomial_type, sumcheck_id) = provider.ra_poly();
        Self {
            log_T,
            r_node_output,
            polynomial_type,
            sumcheck_id,
            log_K: provider.log_K(),
            phases: provider.phases(),
            input_claim: provider.input_claim(accumulator),
        }
    }
}

impl<F> SumcheckInstanceParams<F> for IdentityRCParams<F>
where
    F: JoltField,
{
    fn num_rounds(&self) -> usize {
        self.log_K + self.log_T
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.input_claim
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let (r_address_prime, r_node_output_prime) = challenges.split_at(self.log_K);
        let r_node_output_prime = r_node_output_prime
            .iter()
            .copied()
            .rev()
            .collect::<Vec<_>>();
        OpeningPoint::new([r_address_prime.to_vec(), r_node_output_prime].concat())
    }
}

/// Prover for the Prefix suffix Read-raf checking sum-check protocol.
///
/// Proves correct instruction lookups by evaluating table values using prefix-suffix
/// decomposition and combining them with RAF values that account for operand interleaving.
/// The protocol combines the lookup output with gamma-batched operand contributions.
///
/// The sumcheck has two phases:
/// - Address phase: binds address variables (log K rounds), accumulating ra(k,j)·Val and ra(k,j)·RafVal
/// - Cycle phase: binds cycle variables (log T rounds), evaluating with equality polynomials
pub struct IdentityRCProver<F>
where
    F: JoltField,
{
    /// Shared params between prover and verifier, including table info and opening claims.
    params: IdentityRCParams<F>,
    /// Materialized `ra(k, j)` MLE over (address, cycle) after the first log(K) rounds.
    /// Present only in the last log(T) rounds.
    ra: Option<MultilinearPolynomial<F>>,
    /// u_evals for read-checking and RAF: eq(r_node_output, j).
    u_evals: Vec<F>,
    /// Precomputed lookup keys k (bit-packed) per cycle j.
    lookup_indices: Vec<LookupBits>,
    /// Val(r_address)
    raf_val: Option<F>,
    /// number of phases in the first log K rounds
    phases: usize,
    /// Expanding tables accumulating address-prefix products per phase.
    v: Vec<ExpandingTable<F>>,
    /// Gruen-split equality polynomial over cycle vars.
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    /// Registry holding prefix checkpoint values for `PrefixSuffixDecomposition` instances.
    prefix_registry: PrefixRegistry<F>,
    /// Prefix-suffix decomposition for the instruction-identity path (RAF flag path).
    identity_ps: PrefixSuffixDecomposition<F, 2>,
}

impl<F> IdentityRCProver<F>
where
    F: JoltField,
{
    #[tracing::instrument(name = "IdentityRCProver::gen", skip_all)]
    /// Initializes the Prefix suffix Read-raf checking prover with trace data.
    ///
    /// Computes lookup indices from operand tensors, initializes prefix-suffix
    /// decomposition structures for table values and RAF values, and prepares
    /// expanding tables for accumulating results during the address phase.
    pub fn gen(params: IdentityRCParams<F>, lookup_indices: Vec<LookupBits>) -> Self {
        let log_m = params.log_K / params.phases;
        let u_evals = EqPolynomial::evals(&params.r_node_output.r);
        let identity_poly = IdentityPolynomial::new(params.log_K);
        let span = tracing::span!(tracing::Level::INFO, "Init PrefixSuffixDecomposition");
        let _guard = span.enter();
        let identity_ps =
            PrefixSuffixDecomposition::new(Box::new(identity_poly), log_m, params.log_K);
        drop(_guard);
        drop(span);
        let eq_r_node_output =
            GruenSplitEqPolynomial::<F>::new(&params.r_node_output.r, BindingOrder::LowToHigh);
        let mut res = Self {
            phases: params.phases,
            lookup_indices,
            u_evals,
            v: (0..params.phases)
                .map(|_| ExpandingTable::new(1 << log_m, BindingOrder::HighToLow))
                .collect(),
            identity_ps,
            eq_r_node_output,
            prefix_registry: PrefixRegistry::new(),
            ra: None,
            raf_val: None,
            params,
        };
        res.init_phase(0);
        res
    }

    #[tracing::instrument(name = "IdentityRCProver::init_phase", skip_all)]
    fn init_phase(&mut self, phase: usize) {
        let log_m = self.params.log_K / self.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;

        // Condensation: update u evals
        if phase != 0 {
            self.lookup_indices
                .par_iter()
                .zip(&mut self.u_evals)
                .for_each(|(k, u_eval)| {
                    let (prefix, _) = k.split((self.phases - phase) * log_m);
                    let k_bound = prefix & m_mask;
                    *u_eval *= self.v[phase - 1][k_bound];
                });
        }
        self.identity_ps.init_Q(
            &self.u_evals,
            &(0..self.params.log_T.pow2()).collect_vec(),
            &self.lookup_indices,
        );
        self.identity_ps.init_P(&mut self.prefix_registry);
        self.v[phase].reset(F::one());
    }

    #[tracing::instrument(name = "IdentityRCProver::input_claim", skip_all)]
    /// Address-round prover message: sum of read-checking and RAF components.
    ///
    /// Each component is a degree-2 univariate evaluated at X∈{0,2} using
    /// prefix–suffix decomposition, then added to form the batched message.
    fn compute_prefix_suffix_prover_message(&self, _round: usize, previous_claim: F) -> UniPoly<F> {
        UniPoly::from_evals_and_hint(previous_claim, &self.prover_msg())
    }

    #[tracing::instrument(name = "IdentityRCProver::prover_msg", skip_all)]
    /// RAF part for address rounds.
    ///
    /// Builds two evaluations at X∈{0,2} for the batched
    /// (Left + γ·Right) vs Identity path, folding γ-weights into the result.
    fn prover_msg(&self) -> [F; 2] {
        let len = self.identity_ps.Q_len();
        (0..len / 2)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|b| {
                let (i0, i2) = self.identity_ps.sumcheck_evals(b);
                [i0, i2]
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |running, new| array::from_fn(|i| running[i] + new[i]),
            )
    }

    /// To be called before the last log(T) rounds
    /// Handoff between address and cycle rounds:
    /// - Materializes ra(k,j) from expanding tables across all phases
    /// - Commits prefix checkpoints into a fixed `PrefixEval` vector
    /// - Materializes Val_j(k) from table prefixes/suffixes
    /// - Materializes RafVal_j(k) from (Left,Right,Identity) prefixes with γ-weights
    /// - Converts ra/Val/RafVal into MultilinearPolynomial over (addr,cycle)
    #[tracing::instrument(skip_all, name = "InstructionReadRafProver::init_log_t_rounds")]
    fn init_log_t_rounds(&mut self) {
        let log_m = self.params.log_K / self.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;
        // Drop stuff that's no longer needed
        drop_in_background_thread((std::mem::take(&mut self.u_evals),));

        // Materialize ra polynomial
        let ra = {
            let span = tracing::span!(tracing::Level::INFO, "Materialize ra polynomial");
            let _guard = span.enter();
            self.lookup_indices
                .par_iter()
                .with_min_len(par_enabled())
                .map(|k| {
                    (0..self.phases)
                        .map(|phase| {
                            let (prefix, _) = k.split((self.phases - 1 - phase) * log_m);
                            let k_bound = prefix & m_mask;
                            self.v[phase][k_bound]
                        })
                        .product::<F>()
                })
                .collect::<Vec<_>>()
        };

        drop_in_background_thread(std::mem::take(&mut self.v));
        self.ra = Some(ra.into());
    }
}

impl<F: JoltField, FS: Transcript> SumcheckInstanceProver<F, FS> for IdentityRCProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "InstructionIdentityRCProver::compute_message")]
    /// Produces the prover's degree-≤3 univariate for the current round.
    ///
    /// - For the first LOG_K rounds: returns two evaluations combining
    ///   read-checking and RAF prefix–suffix messages (at X∈{0,2}).
    /// - For the last log(T) rounds: uses Gruen-split EQ.
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if round < self.params.log_K {
            // Phase 1: First log(K) rounds
            self.compute_prefix_suffix_prover_message(round, previous_claim)
        } else {
            let ra = self.ra.as_ref().unwrap();

            let raf_val = self.raf_val.unwrap();
            let [eval_at_0] = self
                .eq_r_node_output
                .par_fold_out_in(
                    || [F::Unreduced::<9>::zero(); 1],
                    |inner, j, _x_in, e_in| {
                        let ra_at_0_j = ra.get_bound_coeff(2 * j);
                        inner[0] += e_in.mul_unreduced::<9>(ra_at_0_j);
                    },
                    |_x_out, e_out, inner| {
                        array::from_fn(|i| {
                            let reduced = F::from_montgomery_reduce(inner[i]);
                            e_out.mul_unreduced::<9>(reduced)
                        })
                    },
                    |a, b| array::from_fn(|i| a[i] + b[i]),
                )
                .map(F::from_montgomery_reduce);
            self.eq_r_node_output
                .gruen_poly_deg_2(eval_at_0 * raf_val, previous_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "InstructionIdentityRCProver::ingest_challenge")]
    /// Binds the next variable (address or cycle) and advances state.
    ///
    /// Address rounds: bind all active prefix–suffix polynomials and the
    /// expanding-table accumulator; update checkpoints every two rounds;
    /// initialize next phase/handoff when needed. Cycle rounds: bind the ra/Val
    /// polynomials and Gruen EQ.
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        let log_m = self.params.log_K / self.phases;
        if round < self.params.log_K {
            let phase = round / log_m;

            // Bind suffix polynomials & update v
            rayon::scope(|s| {
                s.spawn(|_| self.identity_ps.bind(r_j));
                s.spawn(|_| self.v[phase].update(r_j));
            });

            // check if this is the last round in the phase
            if (round + 1).is_multiple_of(log_m) {
                self.prefix_registry.update_checkpoints();
                // if not last phase, init next phase
                if phase != self.phases - 1 {
                    self.init_phase(phase + 1);
                };
            }

            if (round + 1) == self.params.log_K {
                let raf_val = self.prefix_registry.checkpoints[Prefix::Identity].unwrap();
                self.raf_val = Some(raf_val);
                self.init_log_t_rounds();
            }
        } else {
            self.ra
                .as_mut()
                .unwrap()
                .bind_parallel(r_j, BindingOrder::LowToHigh);
            self.eq_r_node_output.bind(r_j);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut FS,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        accumulator.append_virtual(
            transcript,
            OpeningId::new(self.params.polynomial_type, self.params.sumcheck_id),
            opening_point,
            self.ra.as_ref().unwrap().final_claim(),
        );
    }
}

/// Verifier for the Prefix suffix Read-raf checking sum-check protocol.
pub struct IdentityRCVerifier<F>
where
    F: JoltField,
{
    params: IdentityRCParams<F>,
}

impl<F> IdentityRCVerifier<F>
where
    F: JoltField,
{
    /// Creates a new Prefix suffix Read-raf checking verifier.
    pub fn new(params: IdentityRCParams<F>) -> Self {
        Self { params }
    }
}

impl<F: JoltField, FS: Transcript> SumcheckInstanceVerifier<F, FS> for IdentityRCVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut FS,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        accumulator.append_virtual(
            transcript,
            OpeningId::new(self.params.polynomial_type, self.params.sumcheck_id),
            opening_point,
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let (r_address_prime, r_node_output_prime) = opening_point.split_at(self.params.log_K);
        let (_, ra_claim) = accumulator.get_virtual_polynomial_opening(OpeningId::new(
            self.params.polynomial_type,
            self.params.sumcheck_id,
        ));
        let eq_eval = EqPolynomial::mle(&self.params.r_node_output.r, &r_node_output_prime.r);
        let identity_poly_eval =
            IdentityPolynomial::<F>::new(self.params.log_K).evaluate(&r_address_prime.r);
        let raf_claim = identity_poly_eval;
        eq_eval * ra_claim * raf_claim
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReadRafClaims<F>
where
    F: JoltField,
{
    pub rv_claim: F,
    pub left_operand_claim: F,
    pub right_operand_claim: F,
}

/// Provider trait for identity-based range-check sumcheck instances.
pub trait IdentityRCProvider<F>
where
    F: JoltField,
{
    /// Returns the opening claim for the polynomial being range-checked.
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F;
    /// Returns the virtual polynomial and sumcheck id for the one-hot encoded read-address polynomial.
    fn ra_poly(&self) -> (VirtualPoly, SumcheckId);
    /// Returns challenge used in the prefix-suffix sum-check protocol
    fn r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F>;
    /// Returns log₂ of the range upper bound K.
    fn log_K(&self) -> usize;
    /// Number of phases for the range-check sumcheck.
    ///
    /// The prefix-suffix decomposition requires `log_m = log_K / phases` to be
    /// **even** (for `IdentityPolynomial` suffix/prefix MLE halving) and to
    /// divide `log_K` exactly.  The default picks `log_m = 4` when possible,
    /// falling back to `log_m = 2` for even `log_K` values that are not
    /// multiples of 4.
    fn phases(&self) -> usize {
        let log_K = self.log_K();
        if log_K <= 2 {
            1
        } else if log_K % 4 == 0 {
            log_K / 4
        } else if log_K % 2 == 0 {
            log_K / 2
        } else {
            panic!(
                "IdentityRCProvider: odd log_K={log_K} is not supported \
                 (PrefixSuffix decomposition requires even chunk sizes)"
            )
        }
    }
}

// TODO: Auto do ra_one_hot_checks_aswell
pub fn identity_rangecheck_prover<F: JoltField>(
    provider: &impl IdentityRCProvider<F>,
    lookup_indices: Vec<LookupBits>,
    accumulator: &mut ProverOpeningAccumulator<F>,
) -> IdentityRCProver<F> {
    let params = IdentityRCParams::new(provider, accumulator);
    IdentityRCProver::gen(params, lookup_indices)
}

pub fn identity_rangecheck_verifier<F: JoltField>(
    provider: &impl IdentityRCProvider<F>,
    accumulator: &mut VerifierOpeningAccumulator<F>,
) -> IdentityRCVerifier<F> {
    let params = IdentityRCParams::new(provider, accumulator);
    IdentityRCVerifier::new(params)
}
