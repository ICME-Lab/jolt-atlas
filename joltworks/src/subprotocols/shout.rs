use crate::{
    config::OneHotParams,
    field::{FieldChallengeOps, IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        opening_proof::{OpeningAccumulator, OpeningPoint, SumcheckId, BIG_ENDIAN},
    },
    subprotocols::{
        booleanity::{
            BooleanitySumcheckParams, BooleanitySumcheckProver, BooleanitySumcheckVerifier,
        },
        hamming_weight::{
            HammingWeightSumcheckParams, HammingWeightSumcheckProver, HammingWeightSumcheckVerifier,
        },
        ra_virtual::{RaSumcheckParams, RaSumcheckProver, RaSumcheckVerifier},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::thread::unsafe_allocate_zero_vec,
};
use common::{CommittedPolynomial, VirtualPolynomial};
use rayon::prelude::*;
use std::array;

use crate::{
    poly::{
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
        unipoly::UniPoly,
    },
    subprotocols::sumcheck_verifier::SumcheckInstanceParams,
};

const READ_RAF_DEGREE_BOUND: usize = 2;

pub fn read_raf_prover<F: JoltField, T: Transcript>(
    provider: &impl ReadRafProvider<F>,
    lookup_indices: &[usize],
    table: &[i32],
    accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut T,
) -> Box<dyn SumcheckInstanceProver<F, T>> {
    let params = ReadRafParams::new(provider, accumulator, transcript);
    Box::new(ReadRafProver::initialize(lookup_indices, table, params))
}

pub fn read_raf_verifier<F: JoltField, T: Transcript>(
    provider: &impl ReadRafProvider<F>,
    table: Vec<i32>,
    accumulator: &VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> Box<dyn SumcheckInstanceVerifier<F, T>> {
    Box::new(ReadRafVerifier::new(
        provider,
        table,
        accumulator,
        transcript,
    ))
}

// TODO: Have one-unified trait for read-raf and ra-one-hot checks.

pub trait ReadRafProvider<F: JoltField>: Clone + Send + Sync + Sized {
    /// Returns the opening claim for the rv claim
    fn rv_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F;

    /// Returns the opening claim for the raf claim
    fn raf_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F;

    /// Returns challenge used in the read-raf sum-check protocol
    fn r(&self, accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F>;

    /// Returns the virtual polynomial and sumcheck id for the one-hot encoded read-address polynomial.
    fn ra_poly(&self) -> (VirtualPolynomial, SumcheckId);

    /// Log(table_size) for the shout lookup protocol.
    ///
    /// Also known as `num_rounds` for the read-raf sumcheck.
    fn log_K(&self) -> usize;
}

/// Shared prover/verifier parameters for Shout.
#[derive(Clone)]
pub struct ReadRafParams<F: JoltField> {
    r: Vec<F>,
    gamma: F,
    rv_claim: F,
    raf_claim: F,
    ra_vp: VirtualPolynomial,
    ra_sid: SumcheckId,
    log_K: usize,
}

impl<F: JoltField> ReadRafParams<F> {
    /// Create new parameters for exponentiation lookups.
    pub fn new(
        provider: &impl ReadRafProvider<F>,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma = transcript.challenge_scalar();
        Self {
            r: provider.r(accumulator).r,
            gamma,
            rv_claim: provider.rv_claim(accumulator),
            raf_claim: provider.raf_claim(accumulator),
            ra_vp: provider.ra_poly().0,
            ra_sid: provider.ra_poly().1,
            log_K: provider.log_K(),
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ReadRafParams<F> {
    fn degree(&self) -> usize {
        READ_RAF_DEGREE_BOUND
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.rv_claim + self.gamma * self.raf_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<BIG_ENDIAN, F>::new(challenges.to_vec())
    }

    fn num_rounds(&self) -> usize {
        self.log_K
    }
}

/// Prover state for lookups into small tables.
pub struct ReadRafProver<F: JoltField> {
    params: ReadRafParams<F>,
    val: MultilinearPolynomial<F>,
    F: MultilinearPolynomial<F>,
    int: IdentityPolynomial<F>,
}

impl<F: JoltField> ReadRafProver<F> {
    #[tracing::instrument(name = "ReadRafProver::initialize", skip_all)]
    /// Initialize the prover for lookups.
    pub fn initialize(lookup_indices: &[usize], table: &[i32], params: ReadRafParams<F>) -> Self {
        let table_size = 1 << params.log_K;
        let E = EqPolynomial::evals(&params.r);
        let F = lookup_indices
            .par_iter()
            .enumerate()
            .fold(
                || unsafe_allocate_zero_vec::<F>(table_size),
                |mut local_F, (j, &lookup_index)| {
                    local_F[lookup_index] += E[j];
                    local_F
                },
            )
            .reduce(
                || unsafe_allocate_zero_vec::<F>(table_size),
                |mut acc, local_F| {
                    for (i, &val) in local_F.iter().enumerate() {
                        acc[i] += val;
                    }
                    acc
                },
            );
        let val = MultilinearPolynomial::from(table.to_vec());
        Self {
            int: IdentityPolynomial::new(params.log_K),
            params,
            val,
            F: MultilinearPolynomial::from(F),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ReadRafProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(name = "ReadRafProver::compute_message", skip_all)]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self { F, val, int, .. } = self;
        let half_poly_len = val.len() / 2;
        let uni_poly_evals: [F; 2] = (0..half_poly_len)
            .into_par_iter()
            .map(|i| {
                let val_evals =
                    val.sumcheck_evals(i, READ_RAF_DEGREE_BOUND, BindingOrder::HighToLow);
                let f_evals = F.sumcheck_evals(i, READ_RAF_DEGREE_BOUND, BindingOrder::HighToLow);
                let int_evals =
                    int.sumcheck_evals(i, READ_RAF_DEGREE_BOUND, BindingOrder::HighToLow);
                [
                    f_evals[0] * (val_evals[0] + self.params.gamma * int_evals[0]),
                    f_evals[1] * (val_evals[1] + self.params.gamma * int_evals[1]),
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| array::from_fn(|i| running[i] + new[i]),
            );
        UniPoly::from_evals_and_hint(previous_claim, &uni_poly_evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.val.bind_parallel(r_j, BindingOrder::HighToLow);
        self.F.bind_parallel(r_j, BindingOrder::HighToLow);
        self.int.bind_parallel(r_j, BindingOrder::HighToLow);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let sc_challenges: Vec<F> = sumcheck_challenges.to_vec().into_opening();
        let opening_point = [&sc_challenges, self.params.r.as_slice()].concat();
        accumulator.append_virtual(
            transcript,
            self.params.ra_vp,
            self.params.ra_sid,
            opening_point.into(),
            self.F.final_sumcheck_claim(),
        );
    }
}

/// Verifier for lookups into small tables
pub struct ReadRafVerifier<F: JoltField> {
    params: ReadRafParams<F>,
    table: Vec<i32>,
}

impl<F: JoltField> ReadRafVerifier<F> {
    /// Create new verifier for lookups into small tables.
    pub fn new(
        provider: &impl ReadRafProvider<F>,
        table: Vec<i32>,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = ReadRafParams::new(provider, accumulator, transcript);
        Self { params, table }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ReadRafVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let ra_claim = accumulator
            .get_virtual_polynomial_opening(self.params.ra_vp, self.params.ra_sid)
            .1;

        let span = tracing::span!(tracing::Level::INFO, "ReadRafVerifier::val_evaluation");
        let _guard = span.enter();
        let val_claim =
            MultilinearPolynomial::from(self.table.clone()).evaluate(sumcheck_challenges);
        drop(_guard);
        drop(span);
        let int_claim = IdentityPolynomial::new(self.params.log_K).evaluate(sumcheck_challenges);
        ra_claim * (val_claim + self.params.gamma * int_claim)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let sc_challenges: Vec<F> = sumcheck_challenges.to_vec().into_opening();
        let opening_point = [&sc_challenges, self.params.r.as_slice()].concat();
        accumulator.append_virtual(
            transcript,
            self.params.ra_vp,
            self.params.ra_sid,
            opening_point.into(),
        );
    }
}

// ---------------------------------------------------------------------------
// RaOneHotEncoding trait
// ---------------------------------------------------------------------------

/// Look up a virtual-polynomial opening, routing `NodeOutput` through
/// `get_node_output_opening` (which scans all consumer entries for the
/// given producer) instead of requiring an exact `(VP, SumcheckId)` key.
fn resolve_vp_opening<F: JoltField>(
    accumulator: &dyn OpeningAccumulator<F>,
    vp: VirtualPolynomial,
    sid: SumcheckId,
) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
    if let VirtualPolynomial::NodeOutput(producer_idx) = vp {
        accumulator.get_node_output_opening(producer_idx)
    } else {
        accumulator.get_virtual_polynomial_opening(vp, sid)
    }
}

/// Describes the polynomial configuration for the three RA one-hot sumchecks
/// (RaVirtual, HammingWeight, Booleanity). Implementors specify which
/// `CommittedPolynomial` / `VirtualPolynomial` variants to use, the split
/// point for R_a, and the `OneHotParams`.
pub trait RaOneHotEncoding {
    /// The committed polynomial for the `d`-th one-hot decomposition chunk.
    fn committed_poly(&self, d: usize) -> CommittedPolynomial;

    /// The `(VirtualPolynomial, SumcheckId)` whose opening point supplies
    /// `r_cycle` for HammingWeight and Booleanity.
    fn r_cycle_source(&self) -> (VirtualPolynomial, SumcheckId);

    /// The `(VirtualPolynomial, SumcheckId)` whose opening gives `(r, ra_claim)`
    /// for RaVirtual. The point is split at `self.log_k()`.
    fn ra_source(&self) -> (VirtualPolynomial, SumcheckId);

    /// Split point: the first `log_k()` coordinates of the RA opening point
    /// become `r_address`, the rest become `r_cycle` for RaVirtual.
    fn log_k(&self) -> usize;

    /// One-hot encoding parameters (chunk size, instruction_d, etc.).
    fn one_hot_params(&self) -> OneHotParams;
}

// ---------------------------------------------------------------------------
// Generic RA one-hot prover / verifier construction
// ---------------------------------------------------------------------------

/// Build the three RA one-hot sumcheck **provers** (RaVirtual, HammingWeight,
/// Booleanity) from an [`RaOneHotEncoding`] and pre-computed `lookup_indices`.
///
/// Transcript challenge draw order:
/// 1. HammingWeight — `gamma_powers`
/// 2. Booleanity — `gammas`, then `r_address`
/// 3. RaVirtual — no challenges
///
/// Returns `[RaVirtual, HammingWeight, Booleanity]`.
pub fn ra_onehot_provers<F: JoltField, T: Transcript>(
    encoding: &impl RaOneHotEncoding,
    lookup_indices: &[usize],
    accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut T,
) -> [Box<dyn SumcheckInstanceProver<F, T>>; 3] {
    let one_hot_params = encoding.one_hot_params();
    let d = one_hot_params.instruction_d;

    let polynomial_types: Vec<CommittedPolynomial> =
        (0..d).map(|i| encoding.committed_poly(i)).collect();

    // --- HammingWeight params (draws gamma_powers) ---
    let (r_cycle_vp, r_cycle_sid) = encoding.r_cycle_source();
    let r_cycle_hw = resolve_vp_opening(accumulator, r_cycle_vp, r_cycle_sid).0.r;
    let gamma_powers = transcript.challenge_scalar_powers(d);
    let hamming_weight_params = HammingWeightSumcheckParams {
        d,
        num_rounds: one_hot_params.log_k_chunk,
        gamma_powers,
        polynomial_types: polynomial_types.clone(),
        sumcheck_id: SumcheckId::HammingWeight,
        r_cycle: r_cycle_hw,
    };

    // --- Booleanity params (draws gammas, r_address) ---
    let (r_cycle_vp, r_cycle_sid) = encoding.r_cycle_source();
    let r_cycle_bool = resolve_vp_opening(accumulator, r_cycle_vp, r_cycle_sid).0;
    let gammas = transcript.challenge_vector_optimized::<F>(d);
    let r_address = transcript.challenge_vector_optimized::<F>(one_hot_params.log_k_chunk);
    let booleanity_params = BooleanitySumcheckParams {
        d,
        log_k_chunk: one_hot_params.log_k_chunk,
        log_t: r_cycle_bool.r.len(),
        r_cycle: r_cycle_bool.r,
        r_address: r_address.into_opening(),
        gammas,
        polynomial_types: polynomial_types.clone(),
        sumcheck_id: SumcheckId::Booleanity,
    };

    // --- RaVirtual params (no transcript challenges) ---
    let (ra_vp, ra_sid) = encoding.ra_source();
    let (r, ra_claim) = accumulator.get_virtual_polynomial_opening(ra_vp, ra_sid);
    let (r_address_ra, r_cycle_ra) = r.split_at(encoding.log_k());
    let ra_params = RaSumcheckParams {
        r_address: r_address_ra,
        r_cycle: r_cycle_ra,
        one_hot_params: one_hot_params.clone(),
        ra_claim,
        polynomial_types,
    };

    // --- Compute G and H_indices once (shared across all 3 provers) ---
    let G = compute_ra_evals(lookup_indices, &one_hot_params, &booleanity_params.r_cycle);
    let H_indices = compute_instruction_h_indices(lookup_indices, &one_hot_params);

    [
        Box::new(RaSumcheckProver::gen(ra_params, H_indices.clone())),
        Box::new(HammingWeightSumcheckProver::gen(
            hamming_weight_params,
            G.clone(),
        )),
        Box::new(BooleanitySumcheckProver::gen(
            booleanity_params,
            G,
            H_indices,
        )),
    ]
}

/// Build the three RA one-hot sumcheck **verifiers** (RaVirtual,
/// HammingWeight, Booleanity) from an [`RaOneHotEncoding`].
///
/// Transcript challenge draw order matches [`ra_onehot_provers`].
///
/// Returns `[RaVirtual, HammingWeight, Booleanity]`.
pub fn ra_onehot_verifiers<F: JoltField, T: Transcript>(
    encoding: &impl RaOneHotEncoding,
    accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut T,
) -> [Box<dyn SumcheckInstanceVerifier<F, T>>; 3] {
    let one_hot_params = encoding.one_hot_params();
    let d = one_hot_params.instruction_d;

    let polynomial_types: Vec<CommittedPolynomial> =
        (0..d).map(|i| encoding.committed_poly(i)).collect();

    // --- HammingWeight params ---
    let (r_cycle_vp, r_cycle_sid) = encoding.r_cycle_source();
    let r_cycle_hw = resolve_vp_opening(accumulator, r_cycle_vp, r_cycle_sid).0.r;
    let gamma_powers = transcript.challenge_scalar_powers(d);
    let hamming_weight_params = HammingWeightSumcheckParams {
        d,
        num_rounds: one_hot_params.log_k_chunk,
        gamma_powers,
        polynomial_types: polynomial_types.clone(),
        sumcheck_id: SumcheckId::HammingWeight,
        r_cycle: r_cycle_hw,
    };

    // --- Booleanity params ---
    let (r_cycle_vp, r_cycle_sid) = encoding.r_cycle_source();
    let r_cycle_bool = resolve_vp_opening(accumulator, r_cycle_vp, r_cycle_sid).0;
    let gammas = transcript.challenge_vector_optimized::<F>(d);
    let r_address = transcript.challenge_vector_optimized::<F>(one_hot_params.log_k_chunk);
    let booleanity_params = BooleanitySumcheckParams {
        d,
        log_k_chunk: one_hot_params.log_k_chunk,
        log_t: r_cycle_bool.r.len(),
        r_cycle: r_cycle_bool.r,
        r_address: r_address.into_opening(),
        gammas,
        polynomial_types: polynomial_types.clone(),
        sumcheck_id: SumcheckId::Booleanity,
    };

    // --- RaVirtual params ---
    let (ra_vp, ra_sid) = encoding.ra_source();
    let (r, ra_claim) = accumulator.get_virtual_polynomial_opening(ra_vp, ra_sid);
    let (r_address_ra, r_cycle_ra) = r.split_at(encoding.log_k());
    let ra_params = RaSumcheckParams {
        r_address: r_address_ra,
        r_cycle: r_cycle_ra,
        one_hot_params: one_hot_params.clone(),
        ra_claim,
        polynomial_types,
    };

    [
        Box::new(RaSumcheckVerifier::new(ra_params)),
        Box::new(HammingWeightSumcheckVerifier::new(hamming_weight_params)),
        Box::new(BooleanitySumcheckVerifier::new(booleanity_params)),
    ]
}

#[tracing::instrument(skip_all, name = "compute_instruction_h_indices")]
pub fn compute_instruction_h_indices(
    trace: &[usize],
    one_hot_params: &OneHotParams,
) -> Vec<Vec<Option<u8>>> {
    (0..one_hot_params.instruction_d)
        .map(|i| {
            trace
                .par_iter()
                .map(|lookup_index| {
                    Some(one_hot_params.lookup_index_chunk(*lookup_index as u64, i))
                })
                .collect()
        })
        .collect()
}

#[tracing::instrument(skip_all, name = "compute_ra_evals")]
pub fn compute_ra_evals<F, U>(
    trace: &[usize],
    one_hot_params: &OneHotParams,
    r_cycle: &[U],
) -> Vec<Vec<F>>
where
    U: Copy + Send + Sync + Into<F>,
    F: JoltField + FieldChallengeOps<U>,
{
    let eq_r_cycle = EqPolynomial::evals(r_cycle);

    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);

    trace
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, trace_chunk)| {
            let mut result: Vec<Vec<F>> = (0..one_hot_params.instruction_d)
                .map(|_| unsafe_allocate_zero_vec(one_hot_params.k_chunk))
                .collect();
            let mut j = chunk_index * chunk_size;
            for lookup_index in trace_chunk {
                for i in 0..one_hot_params.instruction_d {
                    let k = one_hot_params.lookup_index_chunk(*lookup_index as u64, i);
                    result[i][k as usize] += eq_r_cycle[j];
                }
                j += 1;
            }
            result
        })
        .reduce(
            || {
                (0..one_hot_params.instruction_d)
                    .map(|_| unsafe_allocate_zero_vec(one_hot_params.k_chunk))
                    .collect()
            },
            |mut running, new| {
                running.iter_mut().zip(new.into_iter()).for_each(|(x, y)| {
                    x.par_iter_mut()
                        .zip(y.into_par_iter())
                        .for_each(|(x, y)| *x += y)
                });
                running
            },
        )
}
