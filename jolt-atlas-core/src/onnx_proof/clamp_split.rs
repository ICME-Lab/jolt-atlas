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
//! almost empty (Dory's one-hot commitment is `O(nonzeros)`).
use super::{
    clamp_lookups::{acc_opening_id, clamp_intermediate},
    deferred_lookups::DEFERRED_PROOF_IDX,
    global_clamp::ClampBucket,
};
use atlas_onnx_tracer::{model::trace::Trace, model::Model, node::ComputationNode};
use common::{parallel::par_enabled, CommittedPoly, VirtualPoly};
use joltworks::subprotocols::booleanity::{LinearClaim, LinearTerm};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        one_hot_polynomial::OneHotPolynomial,
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        booleanity::{
            BooleanitySumcheckParams, BooleanitySumcheckProver, BooleanitySumcheckVerifier,
        },
        ps_shout::{CycleWeight, PackedSegment},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::thread::unsafe_allocate_zero_vec,
};
use rayon::prelude::*;

/// Address width of the output range check (`out + 2^31 ∈ [0, 2^32)`).
pub const OUT_LOG_K: usize = 32;
const OFFSET: i64 = 1 << 31;
/// `M = 2^32 − 1`: the offset output of `i32::MAX`.
const OUT_MAX: u64 = (1 << 32) - 1;

fn one_hot_params(log_k: usize) -> OneHotParams {
    OneHotParams::from_config_and_log_K(&OneHotConfig::default(), log_k)
}

/// Number of output chunks (`32 / LOG_K_CHUNK`).
pub fn num_out_chunks() -> usize {
    one_hot_params(OUT_LOG_K).instruction_d
}

/// Number of slack chunks for a `width`-bit bucket.
pub fn num_slack_chunks(width: usize) -> usize {
    one_hot_params(width).instruction_d
}

/// The bucket's committed chunk polynomials: output chunks, then slack chunks.
pub fn chunk_polys(bucket: &ClampBucket) -> Vec<CommittedPoly> {
    (0..num_out_chunks())
        .map(|d| CommittedPoly::GlobalClampOutRaD(bucket.idx, d))
        .chain(
            (0..num_slack_chunks(bucket.width))
                .map(|d| CommittedPoly::GlobalClampSlackRaD(bucket.idx, d)),
        )
        .collect()
}

fn split_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(DEFERRED_PROOF_IDX)
}

fn out_id(bucket_idx: usize) -> OpeningId {
    OpeningId::new(VirtualPoly::GlobalClampOut(bucket_idx), split_sumcheck_id())
}

fn slack_id(bucket_idx: usize) -> OpeningId {
    OpeningId::new(
        VirtualPoly::GlobalClampSlack(bucket_idx),
        split_sumcheck_id(),
    )
}

fn ind_id(bucket_idx: usize) -> OpeningId {
    OpeningId::new(VirtualPoly::GlobalClampInd(bucket_idx), split_sumcheck_id())
}

/// Opening id of the bucket's prover-declared "exact" flag.
pub fn exact_flag_id(bucket_idx: usize) -> OpeningId {
    OpeningId::new(
        VirtualPoly::GlobalClampExact(bucket_idx),
        split_sumcheck_id(),
    )
}

/// The bucket's output-chunk polynomials only.
pub fn out_chunk_polys(bucket: &ClampBucket) -> Vec<CommittedPoly> {
    (0..num_out_chunks())
        .map(|d| CommittedPoly::GlobalClampOutRaD(bucket.idx, d))
        .collect()
}

// ---------------------------------------------------------------------------
// Exact (public, unsaturated) outputs
// ---------------------------------------------------------------------------

/// Whether a tensor has no saturated element.
pub fn unsaturated(t: &[i32]) -> bool {
    !t.iter().any(|&v| v == i32::MAX || v == i32::MIN)
}

/// A model *output* node whose public output has no saturated element proves
/// its clamp *exactly* — `ClampAcc(r) = out(r)` — instead of through the
/// bucket: the verifier knows `out` (an `i32` tensor without `±2^31`), so
/// `SatClamp(acc) = out` is equivalent to `acc = out`. Such a node is a gap
/// in its clamp bucket (prover side: from the trace).
pub fn exact_output_prover(model: &Model, trace: &Trace, node_idx: usize) -> bool {
    model.outputs().contains(&node_idx)
        && trace
            .node_outputs
            .get(&node_idx)
            .is_some_and(|t| unsaturated(t.data()))
}

/// [`exact_output_prover`] from the verifier's public IO.
pub fn exact_output_verifier(
    model: &Model,
    io: &atlas_onnx_tracer::model::trace::ModelExecutionIO,
    node_idx: usize,
) -> bool {
    model.outputs().contains(&node_idx)
        && io
            .output_indices
            .iter()
            .position(|&i| i == node_idx)
            .and_then(|p| io.outputs.get(p))
            .is_some_and(|t| unsaturated(t.data()))
}

/// `bucket` restricted to the nodes that are proven through it (`live`);
/// `None` if no node is.
pub fn live_bucket(
    bucket: &ClampBucket,
    mut live: impl FnMut(usize) -> bool,
) -> Option<ClampBucket> {
    let mut b = bucket.clone();
    b.nodes.retain(|n| live(n.idx));
    (!b.nodes.is_empty()).then_some(b)
}

// ---------------------------------------------------------------------------
// Witness
// ---------------------------------------------------------------------------

/// A node's split witness over its (power-of-two padded) output.
#[derive(Clone, Debug)]
pub struct NodeSplit {
    /// `out + 2^31`.
    pub out: Vec<u32>,
    /// `|acc − out|` on saturated elements, `0` elsewhere.
    pub slack: Vec<u64>,
    /// `1` on saturated elements.
    pub ind: Vec<u8>,
}

impl NodeSplit {
    /// From the padded pre-clamp accumulation and the padded node output.
    /// (Deliberately no consistency check: a tampered trace must be caught by
    /// the verifier, not by the prover's witness generation.)
    pub fn new(acc: &[i64], out: &[i32]) -> Self {
        assert_eq!(acc.len(), out.len(), "clamp split: acc / out length");
        let (out_v, (slack, ind)): (Vec<u32>, (Vec<u64>, Vec<u8>)) = acc
            .par_iter()
            .zip(out.par_iter())
            .with_min_len(par_enabled())
            .map(|(&q, &o)| {
                let saturated = q > i32::MAX as i64 || q < i32::MIN as i64;
                let slack = if saturated { q.abs_diff(o as i64) } else { 0 };
                ((o as i64 + OFFSET) as u32, (slack, saturated as u8))
            })
            .unzip();
        Self {
            out: out_v,
            slack,
            ind,
        }
    }

    /// Re-derive the node's split witness from the trace.
    pub fn from_trace(node: &ComputationNode, trace: &Trace) -> Self {
        let acc = clamp_intermediate(node, trace);
        let out = Trace::layer_data(trace, node)
            .output
            .padded_next_power_of_two();
        Self::new(acc.data(), out.data())
    }
}

/// A bucket's assembled split witness (gaps: interior zeros).
#[derive(Clone, Debug)]
pub struct BucketSplit {
    /// Bucket clamp width (slack address bits).
    pub width: usize,
    /// `out + 2^31` per cycle.
    pub out: Vec<u32>,
    /// Saturation slack per cycle.
    pub slack: Vec<u64>,
    /// Saturation indicator per cycle.
    pub ind: Vec<u8>,
}

impl BucketSplit {
    /// Lay each node's split into the bucket's cycle space (`None`: the node
    /// is a gap — an exact output, see [`exact_output_prover`]).
    pub fn assemble(
        bucket: &ClampBucket,
        mut per_node: impl FnMut(usize) -> Option<NodeSplit>,
    ) -> Self {
        let len = bucket.cycle_len();
        let mut out = vec![OFFSET as u32; len];
        let mut slack = vec![0u64; len];
        let mut ind = vec![0u8; len];
        for n in &bucket.nodes {
            let Some(s) = per_node(n.idx) else {
                continue;
            };
            assert_eq!(s.out.len(), 1 << n.log_t, "node {} split length", n.idx);
            out[n.offset..n.offset + s.out.len()].copy_from_slice(&s.out);
            slack[n.offset..n.offset + s.slack.len()].copy_from_slice(&s.slack);
            ind[n.offset..n.offset + s.ind.len()].copy_from_slice(&s.ind);
        }
        let saturated: usize = ind.iter().map(|&i| i as usize).sum();
        tracing::info!(
            bucket = bucket.idx,
            width = bucket.width,
            log_t = bucket.log_t,
            nodes = bucket.nodes.len(),
            saturated,
            "clamp bucket split"
        );
        Self {
            width: bucket.width,
            out,
            slack,
            ind,
        }
    }

    /// From the trace (commit-time witness generation); exact outputs are gaps.
    pub fn from_trace(bucket: &ClampBucket, trace: &Trace, model: &Model) -> Self {
        Self::assemble(bucket, |idx| {
            (!exact_output_prover(model, trace, idx))
                .then(|| NodeSplit::from_trace(&model.graph.nodes[&idx], trace))
        })
    }

    /// The chunk polynomials of a bucket with no live node: never opened, so
    /// committed empty (all `None`, free under Dory's sparse commit).
    pub fn empty_witness_polys<F: JoltField>(
        bucket: &ClampBucket,
    ) -> Vec<(CommittedPoly, MultilinearPolynomial<F>)> {
        let k_chunk = one_hot_params(OUT_LOG_K).k_chunk;
        chunk_polys(bucket)
            .into_iter()
            .map(|poly| {
                (
                    poly,
                    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                        vec![None; bucket.cycle_len()],
                        k_chunk,
                    )),
                )
            })
            .collect()
    }

    /// The `d`-th chunk index of every cycle: output chunks (`d < 8`, always
    /// set), then slack chunks (`None` on interior cycles).
    pub fn chunk_indices(&self) -> Vec<Vec<Option<u8>>> {
        let out_params = one_hot_params(OUT_LOG_K);
        let slack_params = one_hot_params(self.width);
        let mut chunks: Vec<Vec<Option<u8>>> = (0..out_params.instruction_d)
            .map(|d| {
                self.out
                    .par_iter()
                    .with_min_len(par_enabled())
                    .map(|&o| Some(out_params.lookup_index_chunk(o as u64, d)))
                    .collect()
            })
            .collect();
        chunks.extend((0..slack_params.instruction_d).map(|d| {
            self.slack
                .par_iter()
                .zip(self.ind.par_iter())
                .with_min_len(par_enabled())
                .map(|(&s, &i)| (i == 1).then(|| slack_params.lookup_index_chunk(s, d)))
                .collect()
        }));
        chunks
    }

    /// Whether any cycle is saturated (else the bucket is proven exactly and
    /// its slack chunks stay empty and unopened).
    pub fn saturated(&self) -> bool {
        self.ind.contains(&1)
    }

    /// The bucket's committed one-hot chunk polynomials.
    pub fn witness_polys<F: JoltField>(
        &self,
        bucket: &ClampBucket,
    ) -> Vec<(CommittedPoly, MultilinearPolynomial<F>)> {
        let k_chunk = one_hot_params(OUT_LOG_K).k_chunk;
        let mut indices = self.chunk_indices();
        if !self.saturated() {
            for idx in indices.iter_mut().skip(num_out_chunks()) {
                idx.iter_mut().for_each(|x| *x = None);
            }
        }
        chunk_polys(bucket)
            .into_iter()
            .zip(indices)
            .map(|(poly, idx)| {
                let idx: Vec<Option<u16>> = idx.into_iter().map(|x| x.map(u16::from)).collect();
                (
                    poly,
                    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(idx, k_chunk)),
                )
            })
            .collect()
    }
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

// ---------------------------------------------------------------------------
// Split sumcheck
// ---------------------------------------------------------------------------

/// Parameters of a bucket's split sumcheck.
#[derive(Clone)]
pub struct SplitParams<F: JoltField> {
    /// Bucket index.
    pub bucket_idx: usize,
    /// Packed cycle weight `W(t)`.
    pub cycle: CycleWeight<F>,
    /// `[β₁, β₂, β₃]` (see the module docs).
    pub betas: [F; 3],
    /// `Σ_n γ_n·[out_n(r_n) + 2^31 + β₁·(acc_n(r_n) − out_n(r_n))]`.
    pub input_claim: F,
}

impl<F: JoltField> SplitParams<F> {
    /// Draw the bucket's challenges and resolve its input claim / cycle weight
    /// from the member nodes' openings (already in the accumulator on both sides).
    pub fn draw<T: Transcript>(
        bucket: &ClampBucket,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        let gammas: Vec<F> = transcript.challenge_scalar_powers(bucket.nodes.len());
        let betas: [F; 3] = [
            transcript.challenge_scalar(),
            transcript.challenge_scalar(),
            transcript.challenge_scalar(),
        ];
        let offset = F::from_u64(OFFSET as u64);
        let mut input_claim = F::zero();
        let mut segments = Vec::with_capacity(bucket.nodes.len());
        for (n, gamma) in bucket.nodes.iter().zip(&gammas) {
            let (r, out_claim) = accumulator.get_node_output_opening(n.idx);
            let (_, acc_claim) = accumulator.get_virtual_polynomial_opening(acc_opening_id(n.idx));
            assert_eq!(r.len(), n.log_t, "node {} opening point length", n.idx);
            input_claim += *gamma * (out_claim + offset + betas[0] * (acc_claim - out_claim));
            segments.push(PackedSegment {
                gamma: *gamma,
                prefix: n.offset >> n.log_t,
                prefix_len: bucket.log_t - n.log_t,
                r: r.r,
            });
        }
        Self {
            bucket_idx: bucket.idx,
            cycle: CycleWeight::Packed {
                log_T: bucket.log_t,
                segments,
            },
            betas,
            input_claim,
        }
    }

    /// `β₁/M`, `β₂`, `β₃` and `M` as field elements.
    fn coefficients(&self) -> (F, F, F, F) {
        let m = F::from_u64(OUT_MAX);
        (
            self.betas[0] * m.inverse().expect("2^32 − 1 is invertible"),
            self.betas[1],
            self.betas[2],
            m,
        )
    }

    /// `OUT + β₁/M·IND·S·(2·OUT − M) + β₂·IND·OUT·(OUT − M) + β₃·IND·(1 − IND)`.
    fn combine(&self, out: F, slack: F, ind: F) -> F {
        let (b1, b2, b3, m) = self.coefficients();
        out + b1 * ind * slack * (out + out - m)
            + b2 * ind * out * (out - m)
            + b3 * ind * (F::one() - ind)
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for SplitParams<F> {
    fn degree(&self) -> usize {
        4
    }

    fn num_rounds(&self) -> usize {
        self.cycle.log_T()
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.input_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

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

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
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
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.w.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.out.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.slack.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.ind.bind_parallel(r_j, BindingOrder::LowToHigh);
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

/// Verifier of a bucket's split sumcheck.
pub struct SplitVerifier<F: JoltField> {
    params: SplitParams<F>,
}

impl<F: JoltField> SplitVerifier<F> {
    /// Build from resolved parameters.
    pub fn new(params: SplitParams<F>) -> Self {
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for SplitVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let b = self.params.bucket_idx;
        let out = accumulator.get_virtual_polynomial_opening(out_id(b)).1;
        let slack = accumulator.get_virtual_polynomial_opening(slack_id(b)).1;
        let ind = accumulator.get_virtual_polynomial_opening(ind_id(b)).1;
        self.params.cycle.mle(&r.r) * self.params.combine(out, slack, ind)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let b = self.params.bucket_idx;
        accumulator.append_virtual(transcript, out_id(b), r.clone());
        accumulator.append_virtual(transcript, slack_id(b), r.clone());
        accumulator.append_virtual(transcript, ind_id(b), r);
    }
}

// ---------------------------------------------------------------------------
// Chunk checks: Booleanity with the Hamming / chunk-value linear term
// ---------------------------------------------------------------------------

/// Draw a bucket's chunk-check challenges and resolve the Booleanity
/// parameters, whose linear term carries the Hamming-weight and chunk-value
/// identities of the module docs (all chunks share one opening).
pub fn chunk_check_params<F: JoltField, T: Transcript>(
    bucket: &ClampBucket,
    accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut T,
) -> BooleanitySumcheckParams<F> {
    let polys = chunk_polys(bucket);
    let d = polys.len();
    let n_out = num_out_chunks();
    let n_slack = num_slack_chunks(bucket.width);
    let log_k_chunk = one_hot_params(OUT_LOG_K).log_k_chunk;

    let gamma_powers: Vec<F> = transcript.challenge_scalar_powers(d);
    let beta_out: F = transcript.challenge_scalar();
    let beta_slack: F = transcript.challenge_scalar();
    let bool_gammas = transcript.challenge_vector_optimized::<F>(d);
    let r_address = transcript.challenge_vector_optimized::<F>(log_k_chunk);

    let b = bucket.idx;
    let r_cycle = accumulator.get_virtual_polynomial_opening(out_id(b)).0.r;
    let weights: Vec<F> = (0..n_out)
        .map(|i| beta_out * F::from_u64(1u64 << (log_k_chunk * (n_out - 1 - i))))
        .chain(
            (0..n_slack)
                .map(|i| beta_slack * F::from_u64(1u64 << (log_k_chunk * (n_slack - 1 - i)))),
        )
        .collect();
    // Σ_{d<8} γ^d·1 + Σ_{d≥8} γ^d·IND(r') + β_o·OUT(r') + β_s·S(r')
    let claim = LinearClaim {
        constant: gamma_powers[..n_out].iter().copied().sum(),
        terms: vec![
            (ind_id(b), gamma_powers[n_out..].iter().copied().sum()),
            (out_id(b), beta_out),
            (slack_id(b), beta_slack),
        ],
    };
    BooleanitySumcheckParams {
        linear: Some(LinearTerm {
            gammas: gamma_powers,
            weights,
            claim,
        }),
        d,
        log_k_chunk,
        log_t: r_cycle.len(),
        gammas: bool_gammas,
        r_address: r_address.into_opening(),
        r_cycle,
        polynomial_types: polys,
        sumcheck_id: SumcheckId::Booleanity,
    }
}

// ---------------------------------------------------------------------------
// Packed value range check (rescale remainders): Σ_t W(t)·V(t) = Σ_n γ_n·V_n(r_n)
// ---------------------------------------------------------------------------

/// Opening id of a remainder bucket's packed value at the value sumcheck's output point.
pub fn remainder_value_id(bucket_idx: usize) -> OpeningId {
    OpeningId::new(
        VirtualPoly::GlobalRemainderValue(bucket_idx),
        split_sumcheck_id(),
    )
}

/// A packed bucket of small non-negative values (`< 2^width`) to range-check:
/// a degree-2 *value sumcheck* opens the packed value `V` at a fresh point
/// `r'`, and the chunks' Booleanity linear term proves `V(r') = Σ_d
/// 2^{shift_d}·Σ_k k·ra_d(k, r')` with Hamming weight 1 — so pointwise
/// `V ∈ [0, 2^width)`.
#[derive(Clone)]
pub struct ValueParams<F: JoltField> {
    /// Bucket index.
    pub bucket_idx: usize,
    /// Opening id of the packed value at the sumcheck's output point.
    pub value_id: OpeningId,
    /// Packed cycle weight.
    pub cycle: CycleWeight<F>,
    /// `Σ_n γ_n·V_n(r_n)`.
    pub input_claim: F,
}

impl<F: JoltField> SumcheckInstanceParams<F> for ValueParams<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.cycle.log_T()
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.input_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
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

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
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
        UniPoly::from_evals_and_hint(previous_claim, &[e0, e2])
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.w.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.value.bind_parallel(r_j, BindingOrder::LowToHigh);
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

/// Verifier of the value sumcheck.
pub struct ValueVerifier<F: JoltField> {
    params: ValueParams<F>,
}

impl<F: JoltField> ValueVerifier<F> {
    /// Build from resolved parameters.
    pub fn new(params: ValueParams<F>) -> Self {
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ValueVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let value = accumulator
            .get_virtual_polynomial_opening(self.params.value_id)
            .1;
        self.params.cycle.mle(&r.r) * value
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        accumulator.append_virtual(transcript, self.params.value_id, r);
    }
}

/// Draw a packed value range check's chunk-check challenges and resolve the
/// Booleanity parameters (linear term: Hamming weight 1 and the chunked
/// value `Σ_d 2^{shift_d}·Σ_k k·ra_d(k, r')` = the value opened at `value_id`).
pub fn value_chunk_check_params<F: JoltField, T: Transcript>(
    polys: Vec<CommittedPoly>,
    value_bits: usize,
    value_id: OpeningId,
    accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut T,
) -> BooleanitySumcheckParams<F> {
    let d = polys.len();
    let log_k_chunk = one_hot_params(value_bits).log_k_chunk;
    debug_assert_eq!(d, one_hot_params(value_bits).instruction_d);

    let gamma_powers: Vec<F> = transcript.challenge_scalar_powers(d);
    let beta: F = transcript.challenge_scalar();
    let bool_gammas = transcript.challenge_vector_optimized::<F>(d);
    let r_address = transcript.challenge_vector_optimized::<F>(log_k_chunk);

    let r_cycle = accumulator.get_virtual_polynomial_opening(value_id).0.r;
    let weights: Vec<F> = (0..d)
        .map(|i| beta * F::from_u64(1u64 << (log_k_chunk * (d - 1 - i))))
        .collect();
    let claim = LinearClaim {
        constant: gamma_powers.iter().copied().sum(),
        terms: vec![(value_id, beta)],
    };
    BooleanitySumcheckParams {
        linear: Some(LinearTerm {
            gammas: gamma_powers,
            weights,
            claim,
        }),
        d,
        log_k_chunk,
        log_t: r_cycle.len(),
        gammas: bool_gammas,
        r_address: r_address.into_opening(),
        r_cycle,
        polynomial_types: polys,
        sumcheck_id: SumcheckId::Booleanity,
    }
}

/// [`value_chunk_check_params`] for a remainder bucket.
pub fn remainder_chunk_check_params<F: JoltField, T: Transcript>(
    bucket: &ClampBucket,
    accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut T,
) -> BooleanitySumcheckParams<F> {
    value_chunk_check_params(
        bucket.committed_polys(),
        bucket.width,
        remainder_value_id(bucket.idx),
        accumulator,
        transcript,
    )
}

/// [`value_chunk_check_params`] for an *exact* clamp bucket: its output
/// chunks only, valued at `OUT(r')`.
pub fn exact_chunk_check_params<F: JoltField, T: Transcript>(
    bucket: &ClampBucket,
    accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut T,
) -> BooleanitySumcheckParams<F> {
    value_chunk_check_params(
        out_chunk_polys(bucket),
        OUT_LOG_K,
        out_id(bucket.idx),
        accumulator,
        transcript,
    )
}

/// Parameters of an exact bucket's output range check: the value sumcheck
/// `Σ_t W(t)·OUT(t) = Σ_n γ_n·(out_n(r_n) + 2^31)` (the same `γ_n` draw as
/// [`SplitParams::draw`]; no `β`s).
pub fn exact_value_params<F: JoltField, T: Transcript>(
    bucket: &ClampBucket,
    accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut T,
) -> ValueParams<F> {
    let gammas: Vec<F> = transcript.challenge_scalar_powers(bucket.nodes.len());
    let offset = F::from_u64(OFFSET as u64);
    let mut input_claim = F::zero();
    let mut segments = Vec::with_capacity(bucket.nodes.len());
    for (n, gamma) in bucket.nodes.iter().zip(&gammas) {
        let (r, out_claim) = accumulator.get_node_output_opening(n.idx);
        assert_eq!(r.len(), n.log_t, "node {} opening point length", n.idx);
        input_claim += *gamma * (out_claim + offset);
        segments.push(PackedSegment {
            gamma: *gamma,
            prefix: n.offset >> n.log_t,
            prefix_len: bucket.log_t - n.log_t,
            r: r.r,
        });
    }
    ValueParams {
        bucket_idx: bucket.idx,
        value_id: out_id(bucket.idx),
        cycle: CycleWeight::Packed {
            log_T: bucket.log_t,
            segments,
        },
        input_claim,
    }
}

/// Verifier side of an exact bucket's per-node identity `ClampAcc(r_n) = out_n(r_n)`.
pub fn verify_exact_nodes<F: JoltField>(
    bucket: &ClampBucket,
    accumulator: &dyn OpeningAccumulator<F>,
) -> Result<(), joltworks::utils::errors::ProofVerifyError> {
    for n in &bucket.nodes {
        let acc = accumulator
            .get_virtual_polynomial_opening(acc_opening_id(n.idx))
            .1;
        let out = accumulator.get_node_output_opening(n.idx).1;
        if acc != out {
            return Err(joltworks::utils::errors::ProofVerifyError::InvalidOpeningProof(
                format!(
                    "clamp bucket {} declared exact but node {}'s output differs from its accumulation",
                    bucket.idx, n.idx
                ),
            ));
        }
    }
    Ok(())
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

/// The `d`-th chunk index of every packed value (always set).
pub fn value_chunk_indices(values: &[u64], width: usize) -> Vec<Vec<Option<u8>>> {
    let params = one_hot_params(width);
    (0..params.instruction_d)
        .map(|d| {
            values
                .par_iter()
                .with_min_len(par_enabled())
                .map(|&v| Some(params.lookup_index_chunk(v, d)))
                .collect()
        })
        .collect()
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

/// Verifier counterpart of [`chunk_check_provers`].
pub fn chunk_check_verifiers<F: JoltField, T: Transcript>(
    params: BooleanitySumcheckParams<F>,
) -> Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> {
    vec![Box::new(BooleanitySumcheckVerifier::new(params))]
}
