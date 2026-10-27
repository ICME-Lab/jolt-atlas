//! The prover side of [`super`].

use super::*;
use crate::{
    opening_proof::{prover::ProverOpeningAccumulator, OpeningId, SumcheckId},
    ps_shout::{CyclePoly, CycleWeight},
    sumcheck::prover::SumcheckInstanceProver,
    sumcheck_verifier::SumcheckInstanceParams,
    upstream_sumcheck::round_check_failed,
};
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::{
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        prefix_suffix::{Prefix, PrefixRegistry, PrefixSuffixDecomposition},
        split_eq_poly::SplitEqUnreducedFold,
    },
    utils::{
        expanding_table::ExpandingTable, lookup_bits::LookupBits, thread::drop_in_background_thread,
    },
};
use atlas_transcript::Transcript;
use common::{parallel::par_enabled, VirtualPoly};
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use std::array;

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
    /// Cycle-round weight polynomial (Gruen split-eq, or a dense packed weight).
    cycle_poly: CyclePoly<F>,
    /// Registry holding prefix checkpoint values for `PrefixSuffixDecomposition` instances.
    prefix_registry: PrefixRegistry<F>,
    /// Prefix-suffix decomposition for the instruction-identity path (RAF flag path).
    identity_ps: PrefixSuffixDecomposition<F, 2, false>,
}

impl<F> IdentityRCProver<F>
where
    F: JoltField,
{
    #[tracing::instrument(name = "IdentityRCProver::input_claim", skip_all)]
    /// Address-round prover message: sum of read-checking and RAF components.
    ///
    /// Each component is a degree-2 univariate evaluated at X∈{0,2} using
    /// prefix–suffix decomposition, then added to form the batched message.
    fn compute_prefix_suffix_prover_message(
        &self,
        _round: usize,
        previous_claim: F,
    ) -> UnivariatePoly<F> {
        UnivariatePoly::from_evals_and_hint(previous_claim, &self.prover_msg())
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
}

impl<F: JoltField, FS: Transcript> SumcheckInstanceProver<F, FS> for IdentityRCProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
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

impl<F: JoltField> IdentityRCProver<F> {
    #[tracing::instrument(skip_all, name = "InstructionIdentityRCProver::bind")]
    /// Binds the next variable (address or cycle) and advances state.
    ///
    /// Address rounds: bind all active prefix–suffix polynomials and the
    /// expanding-table accumulator; update checkpoints every two rounds;
    /// initialize next phase/handoff when needed. Cycle rounds: bind the ra/Val
    /// polynomials and Gruen EQ.
    fn bind(&mut self, r: F, round: usize) {
        let r_j = F::challenge_from_field(r);
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
            match &mut self.cycle_poly {
                CyclePoly::Gruen(eq) => eq.bind(r),
                CyclePoly::Dense(w) => w.bind_parallel(r_j, BindingOrder::LowToHigh),
            }
        }
    }
}

impl<F: JoltField> ProveRounds<F> for IdentityRCProver<F> {
    fn num_rounds(&self) -> usize {
        SumcheckInstanceParams::num_rounds(&self.params)
    }

    #[tracing::instrument(skip_all, name = "InstructionIdentityRCProver::prove_round")]
    /// Produces the prover's degree-≤3 univariate for the current round.
    ///
    /// - For the first LOG_K rounds: returns two evaluations combining
    ///   read-checking and RAF prefix–suffix messages (at X∈{0,2}).
    /// - For the last log(T) rounds: uses Gruen-split EQ.
    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r, round - 1);
        }
        {
            if round < self.params.log_K {
                // Phase 1: First log(K) rounds
                Ok(self.compute_prefix_suffix_prover_message(round, previous_claim))
            } else {
                let ra = self.ra.as_ref().unwrap();

                let raf_val = self.raf_val.unwrap();
                match &self.cycle_poly {
                    CyclePoly::Gruen(eq_r_node_output) => eq_r_node_output
                        .gruen_round_deg_2_scaled(
                            &|j| ra.get_bound_coeff(2 * j + 1),
                            &|j| ra.get_bound_coeff(2 * j),
                            raf_val,
                            previous_claim,
                        ),
                    CyclePoly::Dense(w) => {
                        let [e0, e2] = (0..ra.len() / 2)
                            .into_par_iter()
                            .with_min_len(par_enabled())
                            .map(|j| {
                                let r = ra.sumcheck_evals_array::<2>(j, BindingOrder::LowToHigh);
                                let ww = w.sumcheck_evals_array::<2>(j, BindingOrder::LowToHigh);
                                [r[0] * ww[0], r[1] * ww[1]]
                            })
                            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);
                        Ok(UnivariatePoly::from_evals_and_hint(
                            previous_claim,
                            &[e0 * raf_val, e2 * raf_val],
                        ))
                    }
                }
            }
        }
        .map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind, SumcheckInstanceParams::num_rounds(&self.params) - 1);
        Ok(())
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

/// Prover for an explicit [`CycleWeight`] and input claim (packed instances).
pub fn identity_rangecheck_prover_with_cycle<F: JoltField>(
    cycle: CycleWeight<F>,
    log_K: usize,
    ra_poly: (VirtualPoly, SumcheckId),
    input_claim: F,
    lookup_indices: Vec<LookupBits>,
) -> IdentityRCProver<F> {
    let params = IdentityRCParams::from_parts_with_cycle(cycle, log_K, ra_poly, input_claim);
    IdentityRCProver::gen(params, lookup_indices)
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
        let u_evals = params.cycle.evals();
        let identity_poly = IdentityPolynomial::new(params.log_K);
        let span = tracing::span!(tracing::Level::INFO, "Init PrefixSuffixDecomposition");
        let _guard = span.enter();
        let identity_ps =
            PrefixSuffixDecomposition::new(Box::new(identity_poly), log_m, params.log_K);
        drop(_guard);
        drop(span);
        let cycle_poly = match &params.cycle {
            CycleWeight::Eq(r) => CyclePoly::Gruen(GruenSplitEqPolynomial::<F>::new(
                &r.r,
                jolt_poly::BindingOrder::LowToHigh,
            )),
            CycleWeight::Packed { .. } => {
                CyclePoly::Dense(MultilinearPolynomial::from(u_evals.clone()))
            }
        };
        let mut res = Self {
            phases: params.phases,
            lookup_indices,
            u_evals,
            v: (0..params.phases)
                .map(|_| ExpandingTable::new(1 << log_m, BindingOrder::HighToLow))
                .collect(),
            identity_ps,
            cycle_poly,
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
        self.identity_ps.init_Q(&self.u_evals, &self.lookup_indices);
        self.identity_ps.init_P(&mut self.prefix_registry);
        self.v[phase].reset(F::one());
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
