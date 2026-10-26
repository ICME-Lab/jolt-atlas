//! Interior / saturated split of the packed saturating-clamp buckets.
//!
//! A clamped node proves `out = SatClamp(acc)` with `acc` the (i64) pre-clamp
//! accumulation, opened once at the node's output point (`ClampAcc`). The
//! previous protocol looked `acc` up in a `w`-bit saturation table: `w/4`
//! committed one-hot chunks per bucket and a degree-`w/4 + 1` RaVirtual check,
//! for `w ∈ {40, 48, 56, 64}`. Almost every element is *interior*
//! (`acc ∈ i32`, `out = acc`); saturation is rare. This module splits the two
//! cases so the dense work is 32-bit and the `w`-bit work is sparse.
//!
//! Per bucket cycle `t` (see [`super::global_clamp`] for the packing) the
//! prover holds three dense vectors and commits their one-hot chunks:
//!
//! ```text
//! OUT(t) = out(t) + 2^31                          8 dense 4-bit chunks
//! IND(t) = [acc(t) ∉ i32]                         virtual: Hamming weight of the slack chunks
//! S(t)   = IND(t) · |acc(t) − out(t)|             w/4 sparse chunks (None where IND = 0)
//! ```
//!
//! **Split sumcheck** (one per bucket, cycle weight `W(t) = Σ_n γ_n·eq(r_n, t − off_n)`):
//!
//! ```text
//! Σ_t W(t)·[ OUT + β₁/M · IND·S·(2·OUT − M) + β₂ · IND·OUT·(OUT − M) + β₃ · IND·(1 − IND) ]
//!   = Σ_n γ_n · [ out_n(r_n) + 2^31 + β₁·(acc_n(r_n) − out_n(r_n)) ],      M = 2^32 − 1
//! ```
//!
//! whose right-hand side is linear in the nodes' existing openings. It reduces
//! to openings of `OUT`, `S`, `IND` at a fresh point `r'`. Since the `r_n` are
//! drawn after the chunks are committed, the identity at random `r_n`, `γ`, `β`
//! gives the pointwise identities (Schwartz–Zippel): `OUT = out + 2^31`,
//! `acc − out = IND·S·σ` with `σ = (2·OUT − M)/M`, `IND·OUT·(OUT − M) = 0`,
//! `IND ∈ {0, 1}`.
//!
//! **Chunk check**: the Booleanity sumcheck over all chunks carries a linear
//! term (see `joltworks::subprotocols::booleanity::LinearTerm`) that ties the
//! three virtual openings to the committed chunks — every chunk is opened once:
//!
//! ```text
//! Σ_k Σ_d ra_d(k, r') · (γ^d + β_o·2^{4(7−d)}·k)        d < 8      (output chunks)
//! Σ_k Σ_d ra_d(k, r') · (γ^{8+d} + β_s·2^{4(D−1−d)}·k)  d < D      (slack chunks)
//!   = Σ_{d<8} γ^d · 1 + Σ_{d<D} γ^{8+d} · IND(r') + β_o·OUT(r') + β_s·S(r')
//! ```
//!
//! i.e. every output chunk has Hamming weight 1, every slack chunk has Hamming
//! weight `IND`, and `OUT`/`S` are the chunk values. With the standard
//! Booleanity check on all chunks this gives, pointwise: `OUT ∈ [0, 2^32)` so
//! `out ∈ i32`; `IND = 0 ⇒ acc = out`; `IND = 1 ⇒ OUT ∈ {0, M}`, i.e.
//! `out ∈ {i32::MIN, i32::MAX}`, and `acc = out ± S` with `S ∈ [0, 2^w)` on
//! the saturated side (`out = MAX ⇒ acc = MAX + S ≥ MAX`, `out = MIN ⇒ acc =
//! MIN − S ≤ MIN`) — exactly `out = SatClamp(acc)`. (`w` is the node's static
//! clamp width, so `|acc| + 2^w` never wraps the field.)
//!
//! Compared to the table lookup: no RaVirtual (the most expensive check), the
//! dense chunk count drops from `w/4` to 8, and the `w/4` slack chunks are
//! almost empty.
use common::parallel::par_enabled;
pub use jolt_atlas_verifier::onnx_proof::clamp_split::*;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::ProverOpeningAccumulator,
    },
    subprotocols::{
        booleanity::{BooleanitySumcheckParams, BooleanitySumcheckProver},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceParams,
    },
    transcripts::Transcript,
    utils::thread::unsafe_allocate_zero_vec,
};
use rayon::prelude::*;

/// Prover of a bucket's split sumcheck.
pub struct SplitProver<F: JoltField> {
    params: SplitParams<F>,
    w: MultilinearPolynomial<F>,
    out: MultilinearPolynomial<F>,
    slack: MultilinearPolynomial<F>,
    ind: MultilinearPolynomial<F>,
}

impl<F: JoltField> SplitProver<F> {
    /// Build from resolved parameters and the bucket's split witness.
    pub fn new(params: SplitParams<F>, split: &BucketSplit) -> Self {
        let w = MultilinearPolynomial::from(params.cycle.evals());
        Self {
            params,
            w,
            out: MultilinearPolynomial::from(split.out.clone()),
            slack: MultilinearPolynomial::from(split.slack.clone()),
            ind: MultilinearPolynomial::from(split.ind.clone()),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for SplitProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let b = self.params.bucket_idx;
        accumulator.append_virtual(transcript, out_id(b), r.clone(), self.out.final_claim());
        accumulator.append_virtual(transcript, slack_id(b), r.clone(), self.slack.final_claim());
        accumulator.append_virtual(transcript, ind_id(b), r, self.ind.final_claim());
    }
}

impl<F: JoltField> SplitProver<F> {
    fn round_message(&mut self, _round: usize, previous_claim: F) -> UnivariatePoly<F> {
        let (b1, b2, b3, m) = self.params.coefficients();
        let half = self.w.len() / 2;
        let evals = (0..half)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|g| {
                let lin = |p: &MultilinearPolynomial<F>| {
                    let v0 = p.get_bound_coeff(2 * g);
                    (v0, p.get_bound_coeff(2 * g + 1) - v0)
                };
                let (w0, wd) = lin(&self.w);
                let (o0, od) = lin(&self.out);
                let (s0, sd) = lin(&self.slack);
                let (i0, id) = lin(&self.ind);
                let mut acc = [F::zero(); 4];
                // Evaluations at x = 0, 2, 3, 4.
                let (mut w, mut o, mut s, mut i) = (w0, o0, s0, i0);
                for (j, x) in [0u64, 2, 3, 4].into_iter().enumerate() {
                    if x != 0 {
                        // Advance from the previous point (0 → 2 is two steps).
                        let steps = if x == 2 { 2 } else { 1 };
                        for _ in 0..steps {
                            w += wd;
                            o += od;
                            s += sd;
                            i += id;
                        }
                    }
                    let g_val = o
                        + b1 * i * s * (o + o - m)
                        + b2 * i * o * (o - m)
                        + b3 * i * (F::one() - i);
                    acc[j] = w * g_val;
                }
                acc
            })
            .reduce(
                || [F::zero(); 4],
                |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]],
            );
        UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, r: F, _round: usize) {
        let r_j = F::challenge_from_field(r);
        self.w.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.out.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.slack.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.ind.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

impl<F: JoltField> ProveRounds<F> for SplitProver<F> {
    fn num_rounds(&self) -> usize {
        SumcheckInstanceParams::num_rounds(&self.params)
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
        Ok(self.round_message(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind, SumcheckInstanceParams::num_rounds(&self.params) - 1);
        Ok(())
    }
}

/// Prover of the value sumcheck.
pub struct ValueProver<F: JoltField> {
    params: ValueParams<F>,
    w: MultilinearPolynomial<F>,
    value: MultilinearPolynomial<F>,
}

impl<F: JoltField> ValueProver<F> {
    /// Build from resolved parameters and the packed values.
    pub fn new(params: ValueParams<F>, values: Vec<u64>) -> Self {
        let w = MultilinearPolynomial::from(params.cycle.evals());
        Self {
            params,
            w,
            value: MultilinearPolynomial::from(values),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ValueProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        accumulator.append_virtual(
            transcript,
            self.params.value_id,
            r,
            self.value.final_claim(),
        );
    }
}

impl<F: JoltField> ValueProver<F> {
    fn round_message(&mut self, _round: usize, previous_claim: F) -> UnivariatePoly<F> {
        let half = self.w.len() / 2;
        let [e0, e2] = (0..half)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|g| {
                let w0 = self.w.get_bound_coeff(2 * g);
                let w1 = self.w.get_bound_coeff(2 * g + 1);
                let v0 = self.value.get_bound_coeff(2 * g);
                let v1 = self.value.get_bound_coeff(2 * g + 1);
                [w0 * v0, (w1 + w1 - w0) * (v1 + v1 - v0)]
            })
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);
        UnivariatePoly::from_evals_and_hint(previous_claim, &[e0, e2])
    }

    fn bind(&mut self, r: F, _round: usize) {
        let r_j = F::challenge_from_field(r);
        self.w.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.value.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

impl<F: JoltField> ProveRounds<F> for ValueProver<F> {
    fn num_rounds(&self) -> usize {
        SumcheckInstanceParams::num_rounds(&self.params)
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
        Ok(self.round_message(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind, SumcheckInstanceParams::num_rounds(&self.params) - 1);
        Ok(())
    }
}

/// Build an exact bucket's chunk-check prover (Booleanity over the output chunks).
pub fn exact_chunk_check_provers<F: JoltField, T: Transcript>(
    params: BooleanitySumcheckParams<F>,
    split: &BucketSplit,
) -> Vec<Box<dyn SumcheckInstanceProver<F, T>>> {
    let values: Vec<u64> = split.out.iter().map(|&o| o as u64).collect();
    let indices = value_chunk_indices(&values, OUT_LOG_K);
    let g = compute_g(&indices, &params.r_cycle);
    vec![Box::new(BooleanitySumcheckProver::<F, u8>::gen(
        params, g, indices,
    ))]
}

/// Build a remainder bucket's chunk-check prover.
pub fn remainder_chunk_check_provers<F: JoltField, T: Transcript>(
    params: BooleanitySumcheckParams<F>,
    values: &[u64],
    width: usize,
) -> Vec<Box<dyn SumcheckInstanceProver<F, T>>> {
    let indices = value_chunk_indices(values, width);
    let g = compute_g(&indices, &params.r_cycle);
    vec![Box::new(BooleanitySumcheckProver::<F, u8>::gen(
        params, g, indices,
    ))]
}

// ---------------------------------------------------------------------------
// Bucket-level construction
// ---------------------------------------------------------------------------

/// Build a bucket's chunk-check prover from resolved parameters (no
/// accumulator / transcript access).
pub fn chunk_check_provers<F: JoltField, T: Transcript>(
    params: BooleanitySumcheckParams<F>,
    split: &BucketSplit,
) -> Vec<Box<dyn SumcheckInstanceProver<F, T>>> {
    let indices = split.chunk_indices();
    let g = compute_g(&indices, &params.r_cycle);
    vec![Box::new(BooleanitySumcheckProver::<F, u8>::gen(
        params, g, indices,
    ))]
}

/// `G[d][k] = Σ_t eq(r_cycle, t)·[chunk_d(t) = k]` for every chunk.
fn compute_g<F: JoltField>(chunks: &[Vec<Option<u8>>], r_cycle: &[F]) -> Vec<Vec<F>> {
    let eq: Vec<F> = EqPolynomial::evals(r_cycle);
    let k_chunk = one_hot_params(OUT_LOG_K).k_chunk;
    chunks
        .par_iter()
        .map(|idx| {
            let num_chunks = rayon::current_num_threads()
                .next_power_of_two()
                .min(idx.len());
            let chunk_size = (idx.len() / num_chunks).max(1);
            idx.par_chunks(chunk_size)
                .enumerate()
                .map(|(c, part)| {
                    let mut acc: Vec<F> = unsafe_allocate_zero_vec(k_chunk);
                    let base = c * chunk_size;
                    for (j, k) in part.iter().enumerate() {
                        if let Some(k) = k {
                            acc[*k as usize] += eq[base + j];
                        }
                    }
                    acc
                })
                .reduce(
                    || unsafe_allocate_zero_vec(k_chunk),
                    |mut a, b| {
                        a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                        a
                    },
                )
        })
        .collect()
}
