//! The prover side of the packed saturating-clamp buckets' split
//! ([`jolt_atlas_verifier::onnx_proof::clamp_split`] states the protocol and
//! its soundness argument).
//!
//! Per bucket the prover assembles one [`BucketSplit`] from the trace: per
//! cycle `OUT = out + 2^31`, the saturation slack `S` and the indicator `IND`.
//! It commits `OUT` as [`num_out_digits`] dense base-[`DIGIT_BASE`] digit
//! columns (`U8Scalars`) and `S` as one-hot slack chunks, empty off saturated
//! cycles (and everywhere in a bucket with no saturated cycle, which is then
//! proven exactly and never opens them). It proves the split or exact value
//! sumcheck ([`SplitProver`], [`ValueProver`]), the digit range check of `OUT`
//! ([`digit_range_provers`]) and, for a saturated bucket, the slack chunk
//! check ([`chunk_check_provers`]).
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
use atlas_sumcheck::digit_range::{prover::DigitRangeProver, DigitRangeParams};
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
            setup: params.relation_setup().expect("2^32 − 1 is invertible"),
            params,
            w,
            out: split.out_poly(),
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
        let (b1, b2, b3, m) = self.params.coefficients().expect("2^32 − 1 is invertible");
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

/// Build a bucket's digit range-check prover over its `OUT` digit columns.
pub fn digit_range_provers<F: JoltField, T: Transcript>(
    params: DigitRangeParams<F>,
    split: &BucketSplit,
) -> Vec<Box<dyn DynRelationWitness<F, T>>> {
    let columns = split
        .out_digits()
        .into_iter()
        .map(MultilinearPolynomial::from)
        .collect();
    vec![Box::new(DigitRangeProver::new(params, columns))]
}

/// Build a saturated bucket's slack chunk-check prover from resolved parameters (no
/// accumulator / transcript access).
pub fn chunk_check_provers<F: JoltField, T: Transcript>(
    params: BooleanitySumcheckParams<F>,
    split: &BucketSplit,
) -> Vec<Box<dyn DynRelationWitness<F, T>>> {
    let indices = split.slack_chunk_indices();
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
    /// The witness tamper sweep's digit tamper of this bucket, if any ([`digit_tamper`]).
    #[cfg(test)]
    pub(crate) tamper: Option<(usize, digit_tamper::DigitTamperForm)>,
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
            #[cfg(test)]
            tamper: digit_tamper::for_bucket(bucket.idx),
        }
    }

    /// From the trace (commit-time witness generation); exact outputs are gaps.
    pub fn from_trace(bucket: &ClampBucket, trace: &Trace, model: &Model) -> Self {
        Self::assemble(bucket, |idx| {
            (!exact_output_prover(model, trace, idx))
                .then(|| NodeSplit::from_trace(&model.view(idx), trace))
        })
    }

    /// `OUT`'s base-[`DIGIT_BASE`] digits, one column per digit, most significant first.
    pub fn out_digits(&self) -> Vec<Vec<u8>> {
        let n = num_out_digits();
        let digit_bits = DIGIT_BASE.ilog2();
        let digits: Vec<Vec<u8>> = (0..n)
            .map(|j| {
                let shift = (n - 1 - j) as u32 * digit_bits;
                self.out
                    .par_iter()
                    .with_min_len(par_enabled())
                    .map(|&o| ((o >> shift) as u64 % DIGIT_BASE) as u8)
                    .collect()
            })
            .collect();
        #[cfg(test)]
        let digits = {
            let mut digits = digits;
            if let Some((cycle, form)) = self.tamper {
                digit_tamper::apply(&mut digits, cycle, form);
            }
            digits
        };
        digits
    }

    /// `OUT` per cycle as the split sumcheck's witness.
    pub fn out_poly<F: JoltField>(&self) -> MultilinearPolynomial<F> {
        #[cfg(test)]
        if let Some(raised) = self.raised_out() {
            return MultilinearPolynomial::from(raised);
        }
        MultilinearPolynomial::from(self.out.clone())
    }

    /// `OUT` per cycle as the exact value sumcheck's witness.
    pub fn out_values(&self) -> Vec<u64> {
        #[cfg(test)]
        if let Some(raised) = self.raised_out() {
            return raised;
        }
        self.out.iter().map(|&o| o as u64).collect()
    }

    /// Under a [`digit_tamper::DigitTamperForm::RaiseTop`] tamper, `OUT` with the tampered cycle
    /// raised by `2^32`, the value its tampered digits recompose.
    #[cfg(test)]
    fn raised_out(&self) -> Option<Vec<u64>> {
        let (cycle, digit_tamper::DigitTamperForm::RaiseTop) = self.tamper? else {
            return None;
        };
        let mut out: Vec<u64> = self.out.iter().map(|&o| o as u64).collect();
        out[cycle] += 1 << OUT_LOG_K;
        Some(out)
    }

    /// The `d`-th slack chunk index of every cycle (`None` on interior cycles).
    pub fn slack_chunk_indices(&self) -> Vec<Vec<Option<u8>>> {
        let slack_params = one_hot_params(self.width);
        (0..slack_params.instruction_d)
            .map(|d| {
                self.slack
                    .par_iter()
                    .zip(self.ind.par_iter())
                    .with_min_len(par_enabled())
                    .map(|(&s, &i)| (i == 1).then(|| slack_params.lookup_index_chunk(s, d)))
                    .collect()
            })
            .collect()
    }

    /// Whether any cycle is saturated (else the bucket is proven exactly and
    /// its slack chunks stay empty and unopened).
    pub fn saturated(&self) -> bool {
        self.ind.contains(&1)
    }

    /// The bucket's committed polynomials: `OUT`'s dense digit columns, then the one-hot slack
    /// chunks (empty in a bucket with no saturated cycle, which never opens them).
    pub fn witness_polys<F: JoltField>(
        &self,
        bucket: &ClampBucket,
    ) -> Vec<(CommittedPoly, MultilinearPolynomial<F>)> {
        let k_chunk = one_hot_params(self.width).k_chunk;
        let slack = if self.saturated() {
            self.slack_chunk_indices()
        } else {
            vec![vec![None; self.out.len()]; num_slack_chunks(self.width)]
        };
        let digits = out_digit_polys(bucket)
            .into_iter()
            .zip(self.out_digits())
            .map(|(poly, digits)| (poly, MultilinearPolynomial::from(digits)));
        let slack = slack_chunk_polys(bucket)
            .into_iter()
            .zip(slack)
            .map(|(poly, idx)| {
                let idx: Vec<Option<u16>> = idx.into_iter().map(|x| x.map(u16::from)).collect();
                (
                    poly,
                    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(idx, k_chunk)),
                )
            });
        digits.chain(slack).collect()
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

/// Test-only tamper of one clamp bucket's `OUT` digits, for the witness tamper sweep's
/// `ClampDigit` sites: the trace holds no digits, so a trace tamper cannot reach them. The sweep
/// sets the tamper before proving and clears it after; [`BucketSplit::assemble`] records it on
/// the bucket it names, and the digits and `OUT` witnesses read it from there.
#[cfg(test)]
pub(crate) mod digit_tamper {
    use super::DIGIT_BASE;
    use std::sync::Mutex;

    /// How the digits of one cycle are tampered.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum DigitTamperForm {
        /// `d_j + b` and the next more significant digit `− 1`, at the most significant `j`
        /// whose more significant neighbour is nonzero, on the first cycle from the tamper's
        /// that has one: `OUT` is unchanged and one digit is out of range.
        KeepValue,
        /// The most significant digit `+ b`: the digits recompose `OUT + 2^32`, and the value
        /// relations read that `OUT`.
        RaiseTop,
    }

    /// A digit tamper of `bucket` at `cycle`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct DigitTamper {
        pub bucket: usize,
        pub cycle: usize,
        pub form: DigitTamperForm,
    }

    static ACTIVE: Mutex<Option<DigitTamper>> = Mutex::new(None);

    /// Set (or, with `None`, clear) the tamper the next proof applies.
    pub(crate) fn set(tamper: Option<DigitTamper>) {
        *ACTIVE.lock().unwrap() = tamper;
    }

    /// The active tamper's cycle and form, if it names `bucket`.
    pub(crate) fn for_bucket(bucket: usize) -> Option<(usize, DigitTamperForm)> {
        ACTIVE
            .lock()
            .unwrap()
            .filter(|t| t.bucket == bucket)
            .map(|t| (t.cycle, t.form))
    }

    /// Apply `form` to the digit columns at `cycle`.
    pub(super) fn apply(digits: &mut [Vec<u8>], cycle: usize, form: DigitTamperForm) {
        let base = DIGIT_BASE as u8;
        match form {
            DigitTamperForm::KeepValue => {
                // The first cycle from `cycle` on with a nonzero digit above the least
                // significant (`OUT ≥ b`; a saturated `out = i32::MIN` has `OUT = 0`).
                let movable = (cycle..digits[0].len()).find_map(|t| {
                    (1..digits.len())
                        .find(|&j| digits[j - 1][t] >= 1)
                        .map(|j| (t, j))
                });
                if let Some((t, j)) = movable {
                    digits[j][t] += base;
                    digits[j - 1][t] -= 1;
                }
            }
            DigitTamperForm::RaiseTop => digits[0][cycle] += base,
        }
    }
}
