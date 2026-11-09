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
//! term (see `atlas_sumcheck::booleanity::LinearTerm`) that ties the
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
use crate::node_view::{ComputationNode, ModelView};
use crate::onnx_proof::clamp_lookups::clamp_intermediate;
use crate::onnx_proof::global_clamp::ClampBucket;
use crate::onnx_proof::relation::{RelationSetup, RelationWitness};
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::poly::one_hot_polynomial::OneHotPolynomial;
use atlas_algebra::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
    },
    utils::thread::unsafe_allocate_zero_vec,
};
use atlas_sumcheck::booleanity::{prover::BooleanitySumcheckProver, BooleanitySumcheckParams};
use atlas_sumcheck::relation::prover::DynRelationWitness;
use atlas_trace::{Model, Trace};
use atlas_transcript::Transcript;
use common::parallel::par_enabled;
use common::CommittedPoly;
pub use jolt_atlas_verifier::onnx_proof::clamp_split::*;
use jolt_claims::SymbolicSumcheck;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rayon::prelude::*;

/// Prover of a bucket's split sumcheck.
pub struct SplitProver<F: JoltField> {
    params: SplitParams<F>,
    setup: RelationSetup<F>,
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
            setup: params.relation_setup(),
            params,
            w,
            out: MultilinearPolynomial::from(split.out.clone()),
            slack: MultilinearPolynomial::from(split.slack.clone()),
            ind: MultilinearPolynomial::from(split.ind.clone()),
        }
    }
}

impl<F: JoltField> RelationWitness<F> for SplitProver<F> {
    type Relation = SplitParams<F>;

    fn relation(&self) -> &SplitParams<F> {
        &self.params
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        vec![
            self.out.final_claim(),
            self.slack.final_claim(),
            self.ind.final_claim(),
        ]
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
        self.params.rounds()
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
        self.bind(bind, self.params.rounds() - 1);
        Ok(())
    }
}

/// Prover of the value sumcheck.
pub struct ValueProver<F: JoltField> {
    params: ValueParams<F>,
    setup: RelationSetup<F>,
    w: MultilinearPolynomial<F>,
    value: MultilinearPolynomial<F>,
}

impl<F: JoltField> ValueProver<F> {
    /// Build from resolved parameters and the packed values.
    pub fn new(params: ValueParams<F>, values: Vec<u64>) -> Self {
        let w = MultilinearPolynomial::from(params.cycle.evals());
        Self {
            setup: params.relation_setup(),
            params,
            w,
            value: MultilinearPolynomial::from(values),
        }
    }
}

impl<F: JoltField> RelationWitness<F> for ValueProver<F> {
    type Relation = ValueParams<F>;

    fn relation(&self) -> &ValueParams<F> {
        &self.params
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        vec![self.value.final_claim()]
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
        self.params.rounds()
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
        self.bind(bind, self.params.rounds() - 1);
        Ok(())
    }
}

/// Build an exact bucket's chunk-check prover (Booleanity over the output chunks).
pub fn exact_chunk_check_provers<F: JoltField, T: Transcript>(
    params: BooleanitySumcheckParams<F>,
    split: &BucketSplit,
) -> Vec<Box<dyn DynRelationWitness<F, T>>> {
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
) -> Vec<Box<dyn DynRelationWitness<F, T>>> {
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
) -> Vec<Box<dyn DynRelationWitness<F, T>>> {
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

// ---------------------------------------------------------------------------
// Witness
// ---------------------------------------------------------------------------

const OFFSET: i64 = 1 << 31;

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
        let out = TraceView::layer_data(trace, node).output.padded_pow2();
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
                .then(|| NodeSplit::from_trace(&model.view(idx), trace))
        })
    }

    /// The chunk polynomials of a bucket with no live node: never opened, so
    /// committed empty (all `None`).
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
        let indices = if self.saturated() {
            self.chunk_indices()
        } else {
            // Exact bucket: the slack chunks stay empty and are never opened.
            let values: Vec<u64> = self.out.iter().map(|&o| o as u64).collect();
            let mut idx = value_chunk_indices(&values, OUT_LOG_K);
            idx.extend((0..num_slack_chunks(self.width)).map(|_| vec![None; self.out.len()]));
            idx
        };
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

/// A model *output* node whose public output has no saturated element proves
/// its clamp *exactly* — `ClampAcc(r) = out(r)` — instead of through the
/// bucket: the verifier knows `out` (an `i32` tensor without `±2^31`), so
/// `SatClamp(acc) = out` is equivalent to `acc = out`. Such a node is a gap
/// in its clamp bucket (prover side: from the trace).
pub fn exact_output_prover(model: &Model, trace: &Trace, node_idx: usize) -> bool {
    model.output_indices().contains(&node_idx)
        && trace
            .node_outputs
            .get(node_idx)
            .is_some_and(|t| unsaturated(t.data()))
}
