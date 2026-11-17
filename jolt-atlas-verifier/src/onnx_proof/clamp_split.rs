//! Packed range checks: the value sumcheck and the digit range check of a range bucket.
//!
//! A range bucket ([`global_clamp`](super::global_clamp)) packs values that must lie in
//! `[0, 2^width)`: rescale and division remainders, their slacks below a bound, softmax's row
//! gaps, and the offset output `out + 2^31` of every arithmetic node that is not a model output
//! (so that the node's output is an `i32`). Per bucket:
//!
//! 1. a degree-2 **value sumcheck** ([`ValueParams`]) reduces the members' claims, each at its
//!    own point, to one opening of the packed value `V` at a fresh point `r'`;
//! 2. a **digit range check** proves `V(r')` is the recomposition of [`num_digits`]`(width)`
//!    committed base-[`DIGIT_BASE`] digits, each in range.
use super::{deferred_lookups::DEFERRED_PROOF_IDX, global_clamp::ClampBucket};
use crate::onnx_proof::relation::not_derived;
use crate::onnx_proof::relation::{
    low_to_high_point, Produces, Relation, RelationChallenge, RelationDerived, RelationExpr,
    RelationId, RelationOpening, RelationSetup, RelationVerifier,
};
use atlas_algebra::field::JoltField;
use atlas_algebra::field::Ring;
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_claims::{challenge, derived, opening, SymbolicSumcheck};
use atlas_sumcheck::digit_range::DigitRangeParams;
use atlas_sumcheck::{
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    ps_shout::CycleWeight,
};
use atlas_transcript::Transcript;
use common::{CommittedPoly, VirtualPoly};

/// The base of a bucket's dense digits.
pub const DIGIT_BASE: u64 = 4;

/// Number of base-[`DIGIT_BASE`] digits of a `bits`-bit value, `⌈bits / log₂ b⌉`: the digits
/// bound the value by `b^n`, which is `2^bits` when `log₂ b` divides `bits`.
pub fn num_digits(bits: usize) -> usize {
    bits.div_ceil(DIGIT_BASE.ilog2() as usize)
}

fn bucket_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(DEFERRED_PROOF_IDX)
}

// ---------------------------------------------------------------------------
// Packed value range check: Σ_t W(t)·V(t) = Σ_n γ_n·V_n(r_n)
// ---------------------------------------------------------------------------

/// Opening id of a remainder bucket's packed value at the value sumcheck's output point.
pub fn remainder_value_id(bucket_idx: usize) -> OpeningId {
    OpeningId::new(
        VirtualPoly::GlobalRemainderValue(bucket_idx),
        bucket_sumcheck_id(),
    )
}

/// A packed bucket of small non-negative values (`< 2^width`) to range-check:
/// a degree-2 *value sumcheck* opens the packed value `V` at a fresh point
/// `r'`, and the digit range check ([`remainder_digit_range_params`]) proves
/// `V(r')` is the recomposition of [`num_digits`]`(width)` committed digits in
/// `[0, DIGIT_BASE)` — so pointwise `V ∈ [0, 2^width)` (for an even width;
/// `[0, 2^(width+1))` for an odd one at `DIGIT_BASE = 4`).
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

    fn id(&self) -> RelationId {
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
