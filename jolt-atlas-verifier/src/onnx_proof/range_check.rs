//! Packed range checks: the value sumcheck of a range bucket.
//!
//! A range bucket ([`range_bucket`](super::range_bucket)) packs values that must lie in
//! `[0, 2^width)`: rescale and division remainders, their slacks below a bound, softmax's row
//! gaps, and the offset output `out + 2^31` of every arithmetic node that is not a model output
//! (so that the node's output is an `i32`). The packed value `V` is not a polynomial of its
//! own: it is the recomposition `Σ_j b^(n−1−j)·d_j` of the bucket's [`num_digits`]`(width)`
//! committed base-[`DIGIT_BASE`] digit columns, most significant first. A degree-2 **value
//! sumcheck** ([`ValueParams`]) reduces the members' claims, each at its own point, to one
//! opening of every digit column at a fresh point.
//!
//! That a digit is in `[0, b)` is held by the commitment, which commits a digit column at
//! `log₂ b` bits a coefficient; the verifier requires that width when it reads the
//! commitments ([`read_commitments`](super::verifier::read_commitments)). With `n` digits the recomposition is in `[0, b^n)` and never wraps.
use super::range_bucket::RangeBucket;
use crate::onnx_proof::relation::not_derived;
use crate::onnx_proof::relation::{
    low_to_high_point, Produces, Relation, RelationChallenge, RelationDerived, RelationExpr,
    RelationId, RelationOpening, RelationSetup, RelationVerifier,
};
use atlas_algebra::field::JoltField;
use atlas_algebra::field::Ring;
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_claims::{challenge, constant, derived, opening, SymbolicSumcheck};
use atlas_sumcheck::{
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    ps_shout::CycleWeight,
    relation::DenseGroup,
};
use atlas_transcript::Transcript;
use common::CommittedPoly;

/// The base of a bucket's dense digits.
pub const DIGIT_BASE: u64 = 4;

/// Number of base-[`DIGIT_BASE`] digits of a `bits`-bit value, `⌈bits / log₂ b⌉`: the digits
/// bound the value by `b^n`, which is `2^bits` when `log₂ b` divides `bits`.
pub fn num_digits(bits: usize) -> usize {
    bits.div_ceil(DIGIT_BASE.ilog2() as usize)
}

/// The sumcheck the buckets' openings are named under: a node index no node has, since a
/// bucket spans nodes.
fn bucket_sumcheck_id() -> SumcheckId {
    SumcheckId::NodeExecution(usize::MAX)
}

// ---------------------------------------------------------------------------
// Packed value range check: Σ_t W(t)·V(t) = Σ_n γ_n·V_n(r_n)
// ---------------------------------------------------------------------------

/// `b^(n−1−j)` for `j < n`: the place value of each digit, most significant first.
pub fn place_values(n: usize) -> Vec<u64> {
    (0..n).map(|j| DIGIT_BASE.pow((n - 1 - j) as u32)).collect()
}

/// The value sumcheck of a packed bucket of small non-negative values (`< 2^width`): it opens
/// every digit column of the bucket at a fresh point `r'`, where their recomposition is the
/// packed value `V(r')`. The digits are in `[0, DIGIT_BASE)` by their commitment, so pointwise
/// `V ∈ [0, 2^width)` (for an even width; `[0, 2^(width+1))` for an odd one at
/// `DIGIT_BASE = 4`).
#[derive(Clone)]
pub struct ValueParams<F: JoltField> {
    /// Bucket index.
    pub bucket_idx: usize,
    /// The bucket's committed digit columns, most significant first.
    pub digits: Vec<CommittedPoly>,
    /// Packed cycle weight.
    pub cycle: CycleWeight<F>,
    /// `Σ_n γ_n·V_n(r_n)`.
    pub input_claim: F,
}

/// The value sumcheck as a relation: `Σ_n γ_n·V_n(r_n)` against `W(r′)·Σ_j b^(n−1−j)·d_j(r′)`
/// at the low-to-high bound point `r′`, with `W` the packed cycle weight and `d_j` the digit
/// columns. **S ⊆ R.** `W` places each node's eq weight on its own segment and the digits
/// recompose the packed value, so the sum is the batched claims. The parameters carry the
/// cycle weight; the setup carries the input claim.
impl<F: JoltField> ValueParams<F> {
    /// The relation these parameters instantiate, with its setup.
    pub fn into_verifier(self) -> ValueVerifier<F> {
        let setup = self.relation_setup();
        RelationVerifier::at(self, setup)
    }

    /// The openings of the digit columns at the sumcheck's output point.
    pub fn digit_ids(&self) -> Vec<OpeningId> {
        self.digits
            .iter()
            .map(|&digit| OpeningId::new(digit, bucket_sumcheck_id()))
            .collect()
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
    atlas_sumcheck::relation_types!();

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
        let value = place_values(self.digits.len())
            .into_iter()
            .zip(self.digit_ids())
            .fold(RelationExpr::zero(), |acc, (place, id)| {
                acc + constant::<G, _, _, _>(G::from_u64(place)) * opening(RelationOpening::Id(id))
            });
        derived::<G, _, _, _>(RelationDerived::CycleEq) * value
    }
}

impl<F: JoltField> Produces for ValueParams<F> {
    fn produced(&self) -> Vec<OpeningId> {
        self.digit_ids()
    }
}

impl<F: JoltField> Relation<F> for ValueParams<F> {
    /// The digit columns are opened at one point: the opening reduction takes them as one
    /// group.
    fn dense_groups(&self) -> Option<Vec<DenseGroup>> {
        Some(vec![DenseGroup {
            polynomials: self.digits.clone(),
            sumcheck: bucket_sumcheck_id(),
        }])
    }

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

/// A range bucket's digit columns, most significant first.
pub fn remainder_digit_polys(bucket: &RangeBucket) -> Vec<CommittedPoly> {
    (0..num_digits(bucket.width))
        .map(|j| CommittedPoly::GlobalRemainderDigit(bucket.idx, j))
        .collect()
}

/// Parameters of a range bucket's range check: the value sumcheck over the packed
/// remainders, its input claim `Σ_n γ_n·R_n(r_n)` with `γ` drawn here.
/// `Err` if the accumulator lacks a claim it reads.
pub fn remainder_value_params<F: JoltField, T: Transcript>(
    bucket: &RangeBucket,
    accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<ValueParams<F>, ProofVerifyError> {
    let gammas: Vec<F> = transcript.challenge_scalar_powers(bucket.nodes.len());
    let (input_claim, cycle) = bucket.remainder_inputs(&gammas, accumulator)?;
    Ok(ValueParams {
        bucket_idx: bucket.idx,
        digits: remainder_digit_polys(bucket),
        cycle,
        input_claim,
    })
}
