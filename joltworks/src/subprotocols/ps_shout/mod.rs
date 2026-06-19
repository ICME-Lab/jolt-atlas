use crate::{
    field::{IntoOpening, JoltField},
    lookup_tables::{
        prefixes::{PrefixCheckpoints, PrefixEval, Prefixes},
        JoltLookupTable, PrefixSuffixDecompositionTrait,
    },
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        prefix_suffix::PrefixRegistry,
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
        lookup_bits::LookupBits,
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
};
use ark_std::Zero;
use common::{parallel::par_enabled, VirtualPoly};
use itertools::Itertools;
use rayon::prelude::*;
use std::array;

#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{
    InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource,
};

pub mod binary;
pub mod unary;

pub(crate) const DEGREE_BOUND: usize = 2;

/// Common interface shared by both binary and unary ps_shout providers.
///
/// Contains the three methods that are structurally identical across
/// `binary::PrefixSuffixShoutProvider` and `unary::PrefixSuffixShoutProvider`,
/// so implementations only write them once regardless of which variant is used.
pub trait RafShoutProvider<F: JoltField> {
    fn ra_poly(&self) -> (VirtualPoly, SumcheckId);
    fn r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F>;
}
pub(crate) const NUM_PHASES: usize = 8;

/// Verifier-side RAF: computing the RAF's contribution to the input claim
/// and the RAF claim at a given address evaluation point.
///
/// Binary: `raf_input_claim(γ) = left_claim + γ * right_claim`
/// Unary:  `raf_input_claim(γ) = operand_claim`
///
/// Then `input_claim = rv_claim + γ * raf.raf_input_claim(γ)`.
pub trait RafVerifierData<F: JoltField>: Send + Sync {
    fn raf_input_claim(&self, gamma: F) -> F;
    fn raf_claim_at(&self, r_address: &[F], gamma: F) -> F;
}

/// Prover-side RAF state: holds the PS decompositions and produces
/// per-round univariate contributions and final checkpoint values.
pub trait RafProverState<F: JoltField>: Send + Sync {
    fn init_Q(&mut self, u_evals: &[F], lookup_indices: &[LookupBits]);
    fn init_P(&mut self, registry: &mut PrefixRegistry<F>);
    fn prover_msg(&self, gamma: F) -> [F; 2];
    fn bind(&mut self, r_j: F::Challenge);
    fn raf_val(&self, registry: &PrefixRegistry<F>, gamma: F) -> F;
}

/// Parameters shared by the prover and verifier.
///
/// `LOG_K`: total address bits (= number of address-phase rounds).
/// `N`: bit-width used for table prefix/suffix MLE computations.
/// `VD`: verifier-side RAF data (claims + evaluation).
pub struct ReadRafSumcheckParams<F, LUT, const LOG_K: usize, const N: usize, VD>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
{
    pub gamma: F,
    pub log_T: usize,
    pub r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    pub table: LUT,
    pub polynomial_type: VirtualPoly,
    pub sumcheck_id: SumcheckId,
    pub rv_claim: F,
    pub raf: VD,
}

impl<F, LUT, const LOG_K: usize, const N: usize, VD> ReadRafSumcheckParams<F, LUT, LOG_K, N, VD>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        gamma: F,
        log_T: usize,
        r_node_output: OpeningPoint<BIG_ENDIAN, F>,
        table: LUT,
        polynomial_type: VirtualPoly,
        sumcheck_id: SumcheckId,
        rv_claim: F,
        raf: VD,
    ) -> Self {
        Self {
            gamma,
            log_T,
            r_node_output,
            table,
            polynomial_type,
            sumcheck_id,
            rv_claim,
            raf,
        }
    }
}

impl<F, LUT, const LOG_K: usize, const N: usize, VD> SumcheckInstanceParams<F>
    for ReadRafSumcheckParams<F, LUT, LOG_K, N, VD>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
{
    fn num_rounds(&self) -> usize {
        LOG_K + self.log_T
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.rv_claim + self.gamma * self.raf.raf_input_claim(self.gamma)
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let (r_address_prime, r_node_output_prime) = challenges.split_at(LOG_K);
        let r_node_output_prime = r_node_output_prime
            .iter()
            .copied()
            .rev()
            .collect::<Vec<_>>();
        OpeningPoint::new([r_address_prime.to_vec(), r_node_output_prime].concat())
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::default()
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(
        &self,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        Vec::new()
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        let ra_id = OpeningId::new(self.polynomial_type, self.sumcheck_id);
        Some(OutputClaimConstraint::sum_of_products(vec![
            ProductTerm::scaled(ValueSource::Challenge(0), vec![ValueSource::Opening(ra_id)]),
        ]))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        let opening_point = self.normalize_opening_point(&sumcheck_challenges.into_opening());
        let (r_address_prime, r_node_output_prime) = opening_point.split_at(LOG_K);
        let eq_eval = EqPolynomial::mle(&self.r_node_output.r, &r_node_output_prime.r);
        let val_claim = self.table.evaluate_mle(&r_address_prime.r);
        let raf_claim = self.raf.raf_claim_at(&r_address_prime.r, self.gamma);
        vec![eq_eval * (val_claim + self.gamma * raf_claim)]
    }
}

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
    eq_r_node_output: GruenSplitEqPolynomial<F>,
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
    pub fn new_inner(
        params: ReadRafSumcheckParams<F, LUT, LOG_K, N, VD>,
        lookup_indices: Vec<LookupBits>,
        raf_state: PS,
    ) -> Self {
        let phases = NUM_PHASES;
        let log_m = LOG_K / phases;
        let u_evals = EqPolynomial::evals(&params.r_node_output.r);
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
        let eq_r_node_output =
            GruenSplitEqPolynomial::<F>::new(&params.r_node_output.r, BindingOrder::LowToHigh);
        let mut res = Self {
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
            eq_r_node_output,
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

    fn compute_prefix_suffix_prover_message(&self, round: usize, previous_claim: F) -> UniPoly<F> {
        let mut read_checking = [F::zero(), F::zero()];
        let mut raf = [F::zero(), F::zero()];

        rayon::join(
            || read_checking = self.prover_msg_read_checking(round),
            || raf = self.raf_state.prover_msg(self.params.gamma),
        );

        UniPoly::from_evals_and_hint(
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

impl<F: JoltField, FS: Transcript, LUT, const LOG_K: usize, const N: usize, VD, PS>
    SumcheckInstanceProver<F, FS> for ReadRafSumcheckProver<F, LUT, LOG_K, N, VD, PS>
where
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
    PS: RafProverState<F>,
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if round < LOG_K {
            self.compute_prefix_suffix_prover_message(round, previous_claim)
        } else {
            let ra = self.ra.as_ref().unwrap();
            let val = self.val.unwrap();
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
                .gruen_poly_deg_2(eval_at_0 * (val + raf_val), previous_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
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
        let id = OpeningId::new(self.params.polynomial_type, self.params.sumcheck_id);
        accumulator.append_virtual(
            transcript,
            id,
            opening_point,
            self.ra.as_ref().unwrap().final_claim(),
        );
    }
}

/// Generic verifier for the Prefix-Suffix Read-RAF sumcheck.
pub struct ReadRafSumcheckVerifier<F, LUT, const LOG_K: usize, const N: usize, VD>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
{
    params: ReadRafSumcheckParams<F, LUT, LOG_K, N, VD>,
}

impl<F, LUT, const LOG_K: usize, const N: usize, VD> ReadRafSumcheckVerifier<F, LUT, LOG_K, N, VD>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
{
    pub fn new(params: ReadRafSumcheckParams<F, LUT, LOG_K, N, VD>) -> Self {
        Self { params }
    }
}

impl<F: JoltField, FS: Transcript, LUT, const LOG_K: usize, const N: usize, VD>
    SumcheckInstanceVerifier<F, FS> for ReadRafSumcheckVerifier<F, LUT, LOG_K, N, VD>
where
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let (r_address_prime, r_node_output_prime) = opening_point.split_at(LOG_K);
        let id = OpeningId::new(self.params.polynomial_type, self.params.sumcheck_id);
        let (_, ra_claim) = accumulator.get_virtual_polynomial_opening(id);
        let val_claim = self.params.table.evaluate_mle(&r_address_prime.r);
        let eq_eval = EqPolynomial::mle(&self.params.r_node_output.r, &r_node_output_prime.r);
        let raf_claim = self
            .params
            .raf
            .raf_claim_at(&r_address_prime.r, self.params.gamma);
        eq_eval * ra_claim * (val_claim + self.params.gamma * raf_claim)
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
        let id = OpeningId::new(self.params.polynomial_type, self.params.sumcheck_id);
        accumulator.append_virtual(transcript, id, opening_point);
    }
}
