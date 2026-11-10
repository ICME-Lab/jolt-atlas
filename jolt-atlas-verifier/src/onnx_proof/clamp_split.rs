//! Interior / saturated split of the packed saturating-clamp buckets.
//!
//! A clamped node proves `out = SatClamp(acc)` with `acc` the (i64) pre-clamp
//! accumulation, opened once at the node's output point (`ClampAcc`). Almost
//! every element is *interior* (`acc ∈ i32`, `out = acc`); saturation is rare.
//! This module splits the two cases so the dense work is 32-bit and the
//! `w`-bit work is sparse.
//!
//! Per bucket cycle `t` (see [`super::global_clamp`] for the packing) the
//! prover holds three vectors and commits the first as dense digits and the
//! third as one-hot chunks:
//!
//! ```text
//! OUT(t) = out(t) + 2^31     32/log₂b dense base-b digits (U8Scalars in [0, b), b = DIGIT_BASE)
//! IND(t) = [acc(t) ∉ i32]    virtual: Hamming weight of the slack chunks
//! S(t)   = IND(t) · |acc(t) − out(t)|        w/4 sparse one-hot chunks (None where IND = 0)
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
//! drawn after the digits and chunks are committed, the identity at random
//! `r_n`, `γ`, `β` gives the pointwise identities (Schwartz–Zippel):
//! `OUT = out + 2^31`, `acc − out = IND·S·σ` with `σ = (2·OUT − M)/M`,
//! `IND·OUT·(OUT − M) = 0`, `IND ∈ {0, 1}`. A bucket whose prover-declared
//! exact flag is set runs the *exact* value sumcheck instead,
//! `Σ_t W(t)·OUT(t) = Σ_n γ_n·(out_n(r_n) + 2^31)`, with no slack, and its
//! member nodes tie `acc = out` ([`verify_exact_nodes`]).
//!
//! **Digit range** (every live bucket, `atlas_sumcheck::digit_range`): the
//! digits `d_j` of `OUT` are each in `[0, b)` and recompose `OUT(r')`,
//!
//! ```text
//! Σ_t eq(r', t) · [ Σ_j γ^j · ∏_{a<b}(d_j(t) − a) + β · Σ_j b^(n−1−j) · d_j(t) ] = β · OUT(r')
//! ```
//!
//! with `r'` drawn after the digits are committed, so pointwise every digit is
//! in `[0, b)` and `OUT = Σ_j b^(n−1−j)·d_j ∈ [0, 2^32)`, i.e. `out ∈ i32`.
//!
//! **Slack chunk check** (buckets not declared exact): the Booleanity
//! sumcheck over the slack chunks carries a linear term (see
//! `atlas_sumcheck::booleanity::LinearTerm`) tying `IND` and `S` to them, so
//! each slack chunk is opened once:
//!
//! ```text
//! Σ_k Σ_d ra_d(k, r') · (γ^d + β_s·2^{4(D−1−d)}·k)  =  Σ_d γ^d · IND(r') + β_s·S(r')      d < D
//! ```
//!
//! i.e. every slack chunk has Hamming weight `IND` and `S` is their value.
//! Together, pointwise: `IND = 0 ⇒ acc = out`; `IND = 1 ⇒ OUT ∈ {0, M}`, i.e.
//! `out ∈ {i32::MIN, i32::MAX}`, and `acc = out ± S` with `S ∈ [0, 2^w)` on
//! the saturated side (`out = MAX ⇒ acc = MAX + S ≥ MAX`, `out = MIN ⇒ acc =
//! MIN − S ≤ MIN`) — exactly `out = SatClamp(acc)`. (`w` is the node's static
//! clamp width, so `|acc| + 2^w` never wraps the field.)
//!
//! The dense digits cost a committed cell per row whatever the value; the
//! slack chunks are empty off saturated rows and cost nothing there.
use super::{
    clamp_lookups::acc_opening_id, deferred_lookups::DEFERRED_PROOF_IDX, global_clamp::ClampBucket,
};
use crate::node_view::ModelView;
use crate::onnx_proof::relation::not_derived;
use crate::onnx_proof::relation::{
    low_to_high_point, Produces, Relation, RelationChallenge, RelationDerived, RelationExpr,
    RelationId, RelationOpening, RelationSetup, RelationVerifier,
};
use atlas_algebra::field::{IntoOpening, JoltField};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_sumcheck::booleanity::{LinearClaim, LinearTerm};
use atlas_sumcheck::digit_range::DigitRangeParams;
use atlas_sumcheck::relation::DynRelationVerifier;
use atlas_sumcheck::{
    booleanity::BooleanitySumcheckParams,
    config::{OneHotConfig, OneHotParams},
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    ps_shout::{CycleWeight, PackedSegment},
};
use atlas_trace::Model;
use atlas_transcript::Transcript;
use common::{CommittedPoly, VirtualPoly};
use jolt_claims::{challenge, constant, derived, opening, SymbolicSumcheck};
use jolt_field::Ring;

/// Bit width of the output range check (`out + 2^31 ∈ [0, 2^32)`).
pub const OUT_LOG_K: usize = 32;
/// The base of `OUT`'s dense digits.
pub const DIGIT_BASE: u64 = 4;
const OFFSET: i64 = 1 << 31;
/// `M = 2^32 − 1`: the offset output of `i32::MAX`.
const OUT_MAX: u64 = (1 << 32) - 1;

/// One-hot parameters of a split chunk with `log_k`-bit addresses.
pub fn one_hot_params(log_k: usize) -> OneHotParams {
    OneHotParams::from_config_and_log_K(&OneHotConfig::default(), log_k)
}

/// Number of base-[`DIGIT_BASE`] digits of a `bits`-bit value, `⌈bits / log₂ b⌉`: the digits
/// bound the value by `b^n`, which is `2^bits` when `log₂ b` divides `bits`.
pub fn num_digits(bits: usize) -> usize {
    bits.div_ceil(DIGIT_BASE.ilog2() as usize)
}

/// Number of `OUT` digits (`32 / log₂ DIGIT_BASE`).
pub fn num_out_digits() -> usize {
    num_digits(OUT_LOG_K)
}

/// Number of slack chunks for a `width`-bit bucket.
pub fn num_slack_chunks(width: usize) -> usize {
    one_hot_params(width).instruction_d
}

/// The bucket's `OUT` digit columns, most significant first.
pub fn out_digit_polys(bucket: &ClampBucket) -> Vec<CommittedPoly> {
    (0..num_out_digits())
        .map(|j| CommittedPoly::GlobalClampOutDigit(bucket.idx, j))
        .collect()
}

/// The bucket's slack chunk polynomials.
pub fn slack_chunk_polys(bucket: &ClampBucket) -> Vec<CommittedPoly> {
    (0..num_slack_chunks(bucket.width))
        .map(|d| CommittedPoly::GlobalClampSlackRaD(bucket.idx, d))
        .collect()
}

/// The bucket's committed polynomials: the `OUT` digit columns, then the slack chunks.
pub fn chunk_polys(bucket: &ClampBucket) -> Vec<CommittedPoly> {
    [out_digit_polys(bucket), slack_chunk_polys(bucket)].concat()
}

fn split_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(DEFERRED_PROOF_IDX)
}

/// Opening of a bucket's packed output column.
pub fn out_id(bucket_idx: usize) -> OpeningId {
    OpeningId::new(VirtualPoly::GlobalClampOut(bucket_idx), split_sumcheck_id())
}

/// Opening of a bucket's packed slack column.
pub fn slack_id(bucket_idx: usize) -> OpeningId {
    OpeningId::new(
        VirtualPoly::GlobalClampSlack(bucket_idx),
        split_sumcheck_id(),
    )
}

/// Opening of a bucket's packed saturation-indicator column.
pub fn ind_id(bucket_idx: usize) -> OpeningId {
    OpeningId::new(VirtualPoly::GlobalClampInd(bucket_idx), split_sumcheck_id())
}

/// A member node's opening point has one coordinate per variable of its padded output.
fn check_point_len(
    node: &super::global_clamp::BucketNode,
    len: usize,
) -> Result<(), ProofVerifyError> {
    if len != node.log_t {
        return Err(ProofVerifyError::Malformed(format!(
            "node {} opening point has {len} coordinates, expected {}",
            node.idx, node.log_t
        )));
    }
    Ok(())
}

/// Opening id of the bucket's prover-declared "exact" flag.
pub fn exact_flag_id(bucket_idx: usize) -> OpeningId {
    OpeningId::new(
        VirtualPoly::GlobalClampExact(bucket_idx),
        split_sumcheck_id(),
    )
}

// ---------------------------------------------------------------------------
// Exact (public, unsaturated) outputs
// ---------------------------------------------------------------------------

/// Whether a tensor has no saturated element.
pub fn unsaturated(t: &[i32]) -> bool {
    !t.iter().any(|&v| v == i32::MAX || v == i32::MIN)
}

/// [`exact_output_prover`] from the verifier's public IO.
pub fn exact_output_verifier(
    model: &Model,
    io: &atlas_trace::trace::ModelExecutionIO,
    node_idx: usize,
) -> bool {
    model.output_indices().contains(&node_idx)
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
    /// `Err` if the accumulator lacks one of them.
    pub fn draw<T: Transcript>(
        bucket: &ClampBucket,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<Self, ProofVerifyError> {
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
            let (r, out_claim) = accumulator.get_node_output_opening(n.idx)?;
            let (_, acc_claim) =
                accumulator.get_virtual_polynomial_opening(acc_opening_id(n.idx))?;
            check_point_len(n, r.len())?;
            input_claim += *gamma * (out_claim + offset + betas[0] * (acc_claim - out_claim));
            segments.push(PackedSegment {
                gamma: *gamma,
                prefix: n.offset >> n.log_t,
                prefix_len: bucket.log_t - n.log_t,
                r: r.r,
            });
        }
        Ok(Self {
            bucket_idx: bucket.idx,
            cycle: CycleWeight::Packed {
                log_T: bucket.log_t,
                segments,
            },
            betas,
            input_claim,
        })
    }

    /// `β₁/M`, `β₂`, `β₃` and `M` as field elements. `Err` if `M = 2^32 − 1` is zero in `F`.
    pub fn coefficients(&self) -> Result<(F, F, F, F), ProofVerifyError> {
        let m = F::from_u64(OUT_MAX);
        let m_inverse = m.inverse().ok_or_else(|| {
            ProofVerifyError::Malformed("2^32 − 1 is not invertible in the field".into())
        })?;
        Ok((self.betas[0] * m_inverse, self.betas[1], self.betas[2], m))
    }
}

/// A bucket's split sumcheck as a relation:
///
/// ```text
/// input claim   Σ_n γ_n·[out_n(r_n) + 2^31 + β₁·(acc_n(r_n) − out_n(r_n))]
/// output claim  W(r′) · (OUT + β₁/M·IND·S·(2·OUT − M) + β₂·IND·OUT·(OUT − M) + β₃·IND·(1 − IND))
/// ```
///
/// at the low-to-high bound point `r′`, with `W` the packed cycle weight (see the module docs for
/// the S ⊆ R argument). The parameters carry the cycle weight; the setup carries the input claim
/// and the three coefficients.
impl<F: JoltField> SplitParams<F> {
    /// The relation these parameters instantiate, with its setup. `Err` as
    /// [`Self::coefficients`].
    pub fn into_verifier(self) -> Result<SplitVerifier<F>, ProofVerifyError> {
        let setup = self.relation_setup()?;
        Ok(RelationVerifier::at(self, setup))
    }

    /// The relation's setup: the input claim and the identities' coefficients. `Err` as
    /// [`Self::coefficients`].
    pub fn relation_setup(&self) -> Result<RelationSetup<F>, ProofVerifyError> {
        let (b1, b2, b3, _) = self.coefficients()?;
        Ok(RelationSetup {
            incoming_point: Vec::new(),
            challenges: vec![
                (RelationChallenge::Claim, self.input_claim),
                (RelationChallenge::Batching(0), b1),
                (RelationChallenge::Batching(1), b2),
                (RelationChallenge::Batching(2), b3),
            ],
        })
    }
}

impl<F: JoltField> SymbolicSumcheck for SplitParams<F> {
    crate::atlas_relation_types!();
    type Shape = Self;

    fn new(shape: Self) -> Self {
        shape
    }

    fn id() -> RelationId {
        RelationId::ClampSplit
    }

    fn rounds(&self) -> usize {
        self.cycle.log_T()
    }

    fn degree(&self) -> usize {
        4
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        challenge(RelationChallenge::Claim)
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        let b = self.bucket_idx;
        let out = || opening::<G, _, _, _>(RelationOpening::Id(out_id(b)));
        let slack = || opening::<G, _, _, _>(RelationOpening::Id(slack_id(b)));
        let ind = || opening::<G, _, _, _>(RelationOpening::Id(ind_id(b)));
        let m = || constant::<G, _, _, _>(G::from_u64(OUT_MAX));
        let beta = |i| challenge::<G, _, _, _>(RelationChallenge::Batching(i));
        derived::<G, _, _, _>(RelationDerived::CycleEq)
            * (out()
                + beta(0) * ind() * slack() * (out() + out() - m())
                + beta(1) * ind() * out() * (out() - m())
                + beta(2) * ind() * (constant(G::one()) - ind()))
    }
}

impl<F: JoltField> Produces for SplitParams<F> {
    fn produced(&self) -> Vec<OpeningId> {
        let b = self.bucket_idx;
        vec![out_id(b), slack_id(b), ind_id(b)]
    }
}

impl<F: JoltField> Relation<F> for SplitParams<F> {
    fn derive(
        &self,
        id: &RelationDerived,
        _setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Result<F, ProofVerifyError> {
        Ok(match id {
            RelationDerived::CycleEq => self.cycle.mle(&low_to_high_point(challenges)),
            other => return Err(not_derived(other)),
        })
    }
}

/// Verifier of a bucket's split sumcheck: its relation.
pub type SplitVerifier<F> = RelationVerifier<F, SplitParams<F>>;

// ---------------------------------------------------------------------------
// Digit range check of OUT
// ---------------------------------------------------------------------------

/// Draw a bucket's digit range check: `OUT`'s digit columns in range and recomposing `OUT(r')`,
/// the claim the split or exact value sumcheck produced. `Err` if the accumulator lacks it.
pub fn digit_range_params<F: JoltField, T: Transcript>(
    bucket: &ClampBucket,
    accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<DigitRangeParams<F>, ProofVerifyError> {
    DigitRangeParams::draw(
        DIGIT_BASE,
        out_digit_polys(bucket),
        out_id(bucket.idx),
        SumcheckId::DigitRange,
        accumulator,
        transcript,
    )
}

// ---------------------------------------------------------------------------
// Slack chunk check: Booleanity with the Hamming / chunk-value linear term
// ---------------------------------------------------------------------------

/// Draw a bucket's slack chunk-check challenges and resolve the Booleanity
/// parameters, whose linear term carries the Hamming-weight (`= IND`) and
/// chunk-value (`= S`) identities of the module docs (all slack chunks share
/// one opening). `Err` if the accumulator lacks the slack opening.
pub fn chunk_check_params<F: JoltField, T: Transcript>(
    bucket: &ClampBucket,
    accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<BooleanitySumcheckParams<F>, ProofVerifyError> {
    let polys = slack_chunk_polys(bucket);
    let d = polys.len();
    let log_k_chunk = one_hot_params(bucket.width).log_k_chunk;

    let gamma_powers: Vec<F> = transcript.challenge_scalar_powers(d);
    let beta_slack: F = transcript.challenge_scalar();
    let bool_gammas = transcript.challenge_vector_optimized::<F>(d);
    let r_address = transcript.challenge_vector_optimized::<F>(log_k_chunk);

    let b = bucket.idx;
    let r_cycle = accumulator.get_virtual_polynomial_opening(slack_id(b))?.0.r;
    let weights: Vec<F> = (0..d)
        .map(|i| beta_slack * F::from_u64(1u64 << (log_k_chunk * (d - 1 - i))))
        .collect();
    // Σ_d γ^d·IND(r') + β_s·S(r')
    let claim = LinearClaim {
        constant: F::zero(),
        terms: vec![
            (ind_id(b), gamma_powers.iter().copied().sum()),
            (slack_id(b), beta_slack),
        ],
    };
    Ok(BooleanitySumcheckParams {
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
    })
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
/// `r'`, and the digit range check ([`remainder_digit_range_params`]) proves
/// `V(r')` is the recomposition of [`num_digits`]`(width)` committed digits in
/// `[0, DIGIT_BASE)` — so pointwise `V ∈ [0, 2^width)` (for an even width;
/// `[0, 2^(width+1))` for an odd one at `DIGIT_BASE = 4`). The exact clamp
/// buckets use the same sumcheck for `OUT`.
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

/// The value sumcheck as a relation: `Σ_n γ_n·V_n(r_n)` against `W(r′)·V(r′)` at the
/// low-to-high bound point `r′`, with `W` the packed cycle weight. **S ⊆ R.** `W` places each
/// node's eq weight on its own segment, so the sum is the batched claims. The parameters carry the
/// cycle weight; the setup carries the input claim.
impl<F: JoltField> ValueParams<F> {
    /// The relation these parameters instantiate, with its setup.
    pub fn into_verifier(self) -> ValueVerifier<F> {
        let setup = self.relation_setup();
        RelationVerifier::at(self, setup)
    }

    /// The relation's setup: the input claim.
    pub fn relation_setup(&self) -> RelationSetup<F> {
        RelationSetup {
            incoming_point: Vec::new(),
            challenges: vec![(RelationChallenge::Claim, self.input_claim)],
        }
    }
}

impl<F: JoltField> SymbolicSumcheck for ValueParams<F> {
    crate::atlas_relation_types!();
    type Shape = Self;

    fn new(shape: Self) -> Self {
        shape
    }

    fn id() -> RelationId {
        RelationId::PackedValue
    }

    fn rounds(&self) -> usize {
        self.cycle.log_T()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        challenge(RelationChallenge::Claim)
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        derived::<G, _, _, _>(RelationDerived::CycleEq)
            * opening(RelationOpening::Id(self.value_id))
    }
}

impl<F: JoltField> Produces for ValueParams<F> {
    fn produced(&self) -> Vec<OpeningId> {
        vec![self.value_id]
    }
}

impl<F: JoltField> Relation<F> for ValueParams<F> {
    fn derive(
        &self,
        id: &RelationDerived,
        _setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Result<F, ProofVerifyError> {
        Ok(match id {
            RelationDerived::CycleEq => self.cycle.mle(&low_to_high_point(challenges)),
            other => return Err(not_derived(other)),
        })
    }
}

/// Verifier of the value sumcheck: its relation.
pub type ValueVerifier<F> = RelationVerifier<F, ValueParams<F>>;

/// A remainder bucket's digit columns, most significant first.
pub fn remainder_digit_polys(bucket: &ClampBucket) -> Vec<CommittedPoly> {
    (0..num_digits(bucket.width))
        .map(|j| CommittedPoly::GlobalRemainderDigit(bucket.idx, j))
        .collect()
}

/// Draw a remainder bucket's digit range check: its digit columns in range and recomposing the
/// packed value the value sumcheck opened. `Err` if the accumulator lacks that opening.
pub fn remainder_digit_range_params<F: JoltField, T: Transcript>(
    bucket: &ClampBucket,
    accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<DigitRangeParams<F>, ProofVerifyError> {
    DigitRangeParams::draw(
        DIGIT_BASE,
        remainder_digit_polys(bucket),
        remainder_value_id(bucket.idx),
        SumcheckId::DigitRange,
        accumulator,
        transcript,
    )
}

/// Parameters of an exact bucket's output range check: the value sumcheck
/// `Σ_t W(t)·OUT(t) = Σ_n γ_n·(out_n(r_n) + 2^31)` (the same `γ_n` draw as
/// [`SplitParams::draw`]; no `β`s).
/// `Err` if the accumulator lacks a claim it reads.
pub fn exact_value_params<F: JoltField, T: Transcript>(
    bucket: &ClampBucket,
    accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<ValueParams<F>, ProofVerifyError> {
    let gammas: Vec<F> = transcript.challenge_scalar_powers(bucket.nodes.len());
    let offset = F::from_u64(OFFSET as u64);
    let mut input_claim = F::zero();
    let mut segments = Vec::with_capacity(bucket.nodes.len());
    for (n, gamma) in bucket.nodes.iter().zip(&gammas) {
        let (r, out_claim) = accumulator.get_node_output_opening(n.idx)?;
        check_point_len(n, r.len())?;
        input_claim += *gamma * (out_claim + offset);
        segments.push(PackedSegment {
            gamma: *gamma,
            prefix: n.offset >> n.log_t,
            prefix_len: bucket.log_t - n.log_t,
            r: r.r,
        });
    }
    Ok(ValueParams {
        bucket_idx: bucket.idx,
        value_id: out_id(bucket.idx),
        cycle: CycleWeight::Packed {
            log_T: bucket.log_t,
            segments,
        },
        input_claim,
    })
}

/// A bucket's exact flag, read from its opening: `true` when the prover declared every
/// element unsaturated. `Err` if the flag is not a bit or was never appended.
pub fn exact_flag<F: JoltField>(
    bucket: &ClampBucket,
    accumulator: &dyn OpeningAccumulator<F>,
) -> Result<bool, ProofVerifyError> {
    let flag = accumulator
        .get_virtual_polynomial_opening(exact_flag_id(bucket.idx))?
        .1;
    if flag == F::one() {
        Ok(true)
    } else if flag == F::zero() {
        Ok(false)
    } else {
        Err(ProofVerifyError::InvalidOpeningProof(format!(
            "clamp bucket {}: exact flag must be 0 or 1",
            bucket.idx
        )))
    }
}

/// Parameters of a remainder bucket's range check: the value sumcheck over the packed
/// remainders, its input claim `Σ_n γ_n·R_n(r_n)` with `γ` drawn here.
/// `Err` if the accumulator lacks a claim it reads.
pub fn remainder_value_params<F: JoltField, T: Transcript>(
    bucket: &ClampBucket,
    accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<ValueParams<F>, ProofVerifyError> {
    let gammas: Vec<F> = transcript.challenge_scalar_powers(bucket.nodes.len());
    let (input_claim, cycle) = bucket.remainder_inputs(&gammas, accumulator)?;
    Ok(ValueParams {
        bucket_idx: bucket.idx,
        value_id: remainder_value_id(bucket.idx),
        cycle,
        input_claim,
    })
}

/// Verifier side of an exact bucket's per-node identity `ClampAcc(r_n) = out_n(r_n)`.
pub fn verify_exact_nodes<F: JoltField>(
    bucket: &ClampBucket,
    accumulator: &dyn OpeningAccumulator<F>,
) -> Result<(), atlas_algebra::utils::errors::ProofVerifyError> {
    for n in &bucket.nodes {
        let acc = accumulator
            .get_virtual_polynomial_opening(acc_opening_id(n.idx))?
            .1;
        let out = accumulator.get_node_output_opening(n.idx)?.1;
        if acc != out {
            return Err(atlas_algebra::utils::errors::ProofVerifyError::InvalidOpeningProof(
                format!(
                    "clamp bucket {} declared exact but node {}'s output differs from its accumulation",
                    bucket.idx, n.idx
                ),
            ));
        }
    }
    Ok(())
}

/// Verifier counterpart of [`chunk_check_provers`].
pub fn chunk_check_verifiers<F: JoltField, T: Transcript>(
    params: BooleanitySumcheckParams<F>,
) -> Vec<Box<dyn DynRelationVerifier<F, T>>> {
    vec![Box::new(params.into_verifier())]
}
