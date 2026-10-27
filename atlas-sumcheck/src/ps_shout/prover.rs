//! The prover side of [`super`].

use super::*;
use crate::relation::prover::RelationWitness;
use crate::upstream_sumcheck::round_check_failed;
use atlas_algebra::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        prefix_suffix::PrefixRegistry,
        split_eq_poly::SplitEqUnreducedFold,
    },
    utils::{
        expanding_table::ExpandingTable,
        lookup_bits::LookupBits,
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
};
use atlas_lookup_tables::{
    prefixes::{PrefixCheckpoints, PrefixEval, Prefixes},
    JoltLookupTable, PrefixSuffixDecompositionTrait,
};
use common::parallel::par_enabled;
use itertools::Itertools;
use jolt_claims::SymbolicSumcheck;
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use std::array;

/// Generic prover for the Prefix-Suffix Read-RAF sumcheck.
///
/// `LOG_K`: address phase rounds.  `N`: table MLE bit-width.
/// `VD`: verifier RAF data.  `PS`: prover RAF state (PS decompositions).
pub struct ReadRafSumcheckProver<F, LUT, const LOG_K: usize, const N: usize, VD, PS>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
    PS: RafProverState<F>,
{
    params: ReadRafSumcheckParams<F, LUT, LOG_K, N, VD>,
    setup: RelationSetup<F>,
    ra: Option<MultilinearPolynomial<F>>,
    r: Vec<F::Challenge>,
    u_evals: Vec<F>,
    prefix_checkpoints: PrefixCheckpoints<F>,
    suffix_polys: Vec<DensePolynomial<F>>,
    lookup_indices: Vec<LookupBits>,
    val: Option<F>,
    raf_val: Option<F>,
    phases: usize,
    v: Vec<ExpandingTable<F>>,
    cycle_poly: CyclePoly<F>,
    prefix_registry: PrefixRegistry<F>,
    raf_state: PS,
}

impl<F, LUT, const LOG_K: usize, const N: usize, VD, PS>
    ReadRafSumcheckProver<F, LUT, LOG_K, N, VD, PS>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
    PS: RafProverState<F>,
{
    fn compute_prefix_suffix_prover_message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> UnivariatePoly<F> {
        let mut read_checking = [F::zero(), F::zero()];
        let mut raf = [F::zero(), F::zero()];

        rayon::join(
            || read_checking = self.prover_msg_read_checking(round),
            || raf = self.raf_state.prover_msg(self.params.gamma),
        );

        UnivariatePoly::from_evals_and_hint(
            previous_claim,
            &[read_checking[0] + raf[0], read_checking[1] + raf[1]],
        )
    }

    fn prover_msg_read_checking(&self, j: usize) -> [F; 2] {
        let r_x = if j % 2 == 1 {
            self.r.last().copied()
        } else {
            None
        };
        let half_poly_len = self.suffix_polys[0].len() / 2;
        let [eval_0, eval_2_low, eval_2_high] = (0..half_poly_len)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|i| {
                let b = LookupBits::new(i as u64, half_poly_len.log_2());
                let prefix_evals_0 = self
                    .params
                    .table
                    .prefixes()
                    .iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<N, F, F::Challenge>(
                            &self.prefix_checkpoints,
                            r_x,
                            0,
                            b,
                            j,
                        )
                    })
                    .collect_vec();
                let prefix_evals_2 = self
                    .params
                    .table
                    .prefixes()
                    .iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<N, F, F::Challenge>(
                            &self.prefix_checkpoints,
                            r_x,
                            2,
                            b,
                            j,
                        )
                    })
                    .collect_vec();
                let suffix_evals_low = self.suffix_polys.iter().map(|s| s[i]).collect_vec();
                let suffix_evals_high = self
                    .suffix_polys
                    .iter()
                    .map(|s| s[i + half_poly_len])
                    .collect_vec();
                [
                    self.params
                        .table
                        .combine(&prefix_evals_0, &suffix_evals_low),
                    self.params
                        .table
                        .combine(&prefix_evals_2, &suffix_evals_low),
                    self.params
                        .table
                        .combine(&prefix_evals_2, &suffix_evals_high),
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |running, new| array::from_fn(|i| running[i] + new[i]),
            );
        [eval_0, eval_2_high + eval_2_high - eval_2_low]
    }
}

impl<F, LUT, const LOG_K: usize, const N: usize, VD, PS> RelationWitness<F>
    for ReadRafSumcheckProver<F, LUT, LOG_K, N, VD, PS>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
    PS: RafProverState<F>,
{
    type Relation = ReadRafSumcheckParams<F, LUT, LOG_K, N, VD>;

    fn relation(&self) -> &Self::Relation {
        &self.params
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        vec![self.ra.as_ref().unwrap().final_claim()]
    }
}

impl<F: JoltField, LUT, const LOG_K: usize, const N: usize, VD, PS> ProveRounds<F>
    for ReadRafSumcheckProver<F, LUT, LOG_K, N, VD, PS>
where
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
    PS: RafProverState<F>,
{
    fn num_rounds(&self) -> usize {
        self.params.rounds()
    }

    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::prove_round")]
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
            if round < LOG_K {
                Ok(self.compute_prefix_suffix_prover_message(round, previous_claim))
            } else {
                let ra = self.ra.as_ref().unwrap();
                let val = self.val.unwrap();
                let raf_val = self.raf_val.unwrap();
                match &self.cycle_poly {
                    CyclePoly::Gruen(eq_r_node_output) => eq_r_node_output
                        .gruen_round_deg_2_scaled(
                            &|j| ra.get_bound_coeff(2 * j + 1),
                            &|j| ra.get_bound_coeff(2 * j),
                            val + raf_val,
                            previous_claim,
                        ),
                    CyclePoly::Dense(w) => {
                        // Degree-2 message Σ_j w(j)·ra(j), evaluated at X ∈ {0, 2}.
                        let [e0, e2] = (0..ra.len() / 2)
                            .into_par_iter()
                            .with_min_len(par_enabled())
                            .map(|j| {
                                let r = ra.sumcheck_evals_array::<2>(j, BindingOrder::LowToHigh);
                                let ww = w.sumcheck_evals_array::<2>(j, BindingOrder::LowToHigh);
                                [r[0] * ww[0], r[1] * ww[1]]
                            })
                            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);
                        let scale = val + raf_val;
                        Ok(UnivariatePoly::from_evals_and_hint(
                            previous_claim,
                            &[e0 * scale, e2 * scale],
                        ))
                    }
                }
            }
        }
        .map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind, self.params.rounds() - 1);
        Ok(())
    }
}

impl<F, LUT, const LOG_K: usize, const N: usize, VD, PS>
    ReadRafSumcheckProver<F, LUT, LOG_K, N, VD, PS>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
    PS: RafProverState<F>,
{
    pub fn new_inner(
        params: ReadRafSumcheckParams<F, LUT, LOG_K, N, VD>,
        lookup_indices: Vec<LookupBits>,
        raf_state: PS,
    ) -> Self {
        let (phases, log_m) = phase_schedule(LOG_K);
        let u_evals = params.cycle.evals();
        let prefix_checkpoints = PrefixCheckpoints::new();
        let suffix_polys: Vec<DensePolynomial<F>> = params
            .table
            .suffixes()
            .iter()
            .collect::<Vec<_>>()
            .par_iter()
            .with_min_len(par_enabled())
            .map(|_| DensePolynomial::default())
            .collect();
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
            setup: params.relation_setup(),
            r: Vec::with_capacity(params.log_T + LOG_K),
            params,
            phases,
            lookup_indices,
            prefix_checkpoints,
            suffix_polys,
            u_evals,
            v: (0..phases)
                .map(|_| ExpandingTable::new(1 << log_m, BindingOrder::HighToLow))
                .collect(),
            raf_state,
            cycle_poly,
            prefix_registry: PrefixRegistry::new(),
            val: None,
            ra: None,
            raf_val: None,
        };
        res.init_phase(0);
        res
    }

    fn init_phase(&mut self, phase: usize) {
        let log_m = LOG_K / self.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;

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

        {
            let raf_state = &mut self.raf_state;
            let u_evals = &self.u_evals;
            let lookup_indices = &self.lookup_indices;
            raf_state.init_Q(u_evals, lookup_indices);
        }

        self.init_suffix_polys(phase);

        {
            let raf_state = &mut self.raf_state;
            let prefix_registry = &mut self.prefix_registry;
            raf_state.init_P(prefix_registry);
        }

        self.v[phase].reset(F::one());
    }

    #[tracing::instrument(skip_all, name = "InstructionReadRafProver::init_suffix_polys")]
    fn init_suffix_polys(&mut self, phase: usize) {
        let log_m = LOG_K / self.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;
        let suffix_len = (self.phases - 1 - phase) * log_m;
        let new_suffix_polys = self
            .params
            .table
            .suffixes()
            .par_iter()
            .with_min_len(par_enabled())
            .map(|s| {
                let mut Q = unsafe_allocate_zero_vec(m);
                self.lookup_indices.iter().enumerate().for_each(|(j, &k)| {
                    let (prefix_bits, suffix_bits) = k.split(suffix_len);
                    let y = prefix_bits & m_mask;
                    let t = s.suffix_mle::<N>(suffix_bits);
                    if t != 0 {
                        Q[y] += self.u_evals[j] * F::from_u32(t)
                    };
                });
                Q
            })
            .collect::<Vec<_>>();

        self.suffix_polys
            .iter_mut()
            .zip(new_suffix_polys.into_iter())
            .for_each(|(poly, mut coeffs)| {
                *poly = DensePolynomial::new(std::mem::take(&mut coeffs))
            });
    }

    #[tracing::instrument(skip_all, name = "InstructionReadRafProver::init_log_t_rounds")]
    fn init_log_t_rounds(&mut self) {
        let log_m = LOG_K / self.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;
        drop_in_background_thread((std::mem::take(&mut self.u_evals),));

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

impl<F: JoltField, LUT, const LOG_K: usize, const N: usize, VD, PS>
    ReadRafSumcheckProver<F, LUT, LOG_K, N, VD, PS>
where
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
    PS: RafProverState<F>,
{
    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::bind")]
    fn bind(&mut self, r: F, round: usize) {
        let r_j = F::challenge_from_field(r);
        let log_m = LOG_K / self.phases;
        self.r.push(r_j);
        if round < LOG_K {
            let phase = round / log_m;

            rayon::scope(|s| {
                s.spawn(|_| {
                    self.suffix_polys
                        .par_iter_mut()
                        .with_min_len(par_enabled())
                        .for_each(|s| s.bind(r_j, BindingOrder::HighToLow));
                });
                s.spawn(|_| self.raf_state.bind(r_j));
                s.spawn(|_| self.v[phase].update(r_j));
            });

            if self.r.len().is_multiple_of(2) {
                let suffix_len = LOG_K - (round / log_m + 1) * log_m;
                Prefixes::update_checkpoints::<N, F, F::Challenge>(
                    &mut self.prefix_checkpoints,
                    self.r[self.r.len() - 2],
                    self.r[self.r.len() - 1],
                    round,
                    suffix_len,
                );
            }

            if (round + 1).is_multiple_of(log_m) {
                self.prefix_registry.update_checkpoints();
                if phase != self.phases - 1 {
                    self.init_phase(phase + 1);
                }
            }

            if (round + 1) == LOG_K {
                let prefixes: Vec<PrefixEval<F>> = self
                    .params
                    .table
                    .prefixes()
                    .into_iter()
                    .map(|p| {
                        PrefixEval::from(
                            self.prefix_checkpoints[p]
                                .unwrap_or_else(|| panic!("{p:?} should have a bounded value")),
                        )
                    })
                    .collect();
                let suffixes: Vec<_> = self
                    .params
                    .table
                    .suffixes()
                    .iter()
                    .map(|suffix| F::from_u32(suffix.suffix_mle::<N>(LookupBits::new(0, 0))))
                    .collect();
                self.val = Some(self.params.table.combine(&prefixes, &suffixes));
                let raf_val = self
                    .raf_state
                    .raf_val(&self.prefix_registry, self.params.gamma);
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
