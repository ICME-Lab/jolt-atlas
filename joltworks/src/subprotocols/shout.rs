use crate::{
    config::OneHotParams,
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        opening_proof::{OpeningAccumulator, SumcheckId},
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

// ---------------------------------------------------------------------------
// RaOneHotEncoding trait
// ---------------------------------------------------------------------------

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
    let r_cycle_hw = accumulator
        .get_virtual_polynomial_opening(r_cycle_vp, r_cycle_sid)
        .0
        .r;
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
    let r_cycle_bool = accumulator
        .get_virtual_polynomial_opening(r_cycle_vp, r_cycle_sid)
        .0;
    let gammas = transcript.challenge_vector_optimized::<F>(d);
    let r_address = transcript.challenge_vector_optimized::<F>(one_hot_params.log_k_chunk);
    let booleanity_params = BooleanitySumcheckParams {
        d,
        log_k_chunk: one_hot_params.log_k_chunk,
        log_t: r_cycle_bool.r.len(),
        r_cycle: r_cycle_bool.r,
        r_address,
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
    let r_cycle_hw = accumulator
        .get_virtual_polynomial_opening(r_cycle_vp, r_cycle_sid)
        .0
        .r;
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
    let r_cycle_bool = accumulator
        .get_virtual_polynomial_opening(r_cycle_vp, r_cycle_sid)
        .0;
    let gammas = transcript.challenge_vector_optimized::<F>(d);
    let r_address = transcript.challenge_vector_optimized::<F>(one_hot_params.log_k_chunk);
    let booleanity_params = BooleanitySumcheckParams {
        d,
        log_k_chunk: one_hot_params.log_k_chunk,
        log_t: r_cycle_bool.r.len(),
        r_cycle: r_cycle_bool.r,
        r_address,
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
pub fn compute_ra_evals<F: JoltField>(
    trace: &[usize],
    one_hot_params: &OneHotParams,
    r_cycle: &[F::Challenge],
) -> Vec<Vec<F>> {
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
