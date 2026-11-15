//! The batched sumcheck engine: the round loop every relation is proven with
//! ([`prover::prove_batch`]) and the compressed-round verifier every relation is checked with
//! ([`verify_compressed`]).
//!
//! The sumcheck reduces a claim $\sum_{x \in \{0,1\}^n} g(x) = C$ to one evaluation
//! $g(r_1, \ldots, r_n) = v$ in $n$ rounds: in round $i$ the prover sends a univariate
//! $s_i(X)$, the verifier checks it against the running sum, squeezes $r_i$ and recurses. A
//! batch runs several members as one sumcheck over the padded random linear combination
//! described on [`BatchPrelude`].
//!
//! Every challenge the engine draws is preceded by a proof of work of [`round_pow_bits`] for
//! the batch's degree. The engine's transcript writes are the deployed proof format's word
//! encodings ([`atlas_transcript::words`]): field elements as canonical big-endian bytes and
//! labels as 32-byte words.
//!
//! Prover items live in [`prover`]; everything else here is shared with the verifier.

#![forbid(unsafe_code)]
#![deny(
    clippy::indexing_slicing,
    clippy::get_unwrap,
    clippy::string_slice,
    clippy::fallible_impl_from,
    clippy::mem_forget,
    clippy::exit,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::host_endian_bytes,
    clippy::wildcard_enum_match_arm
)]

pub mod prover;
#[cfg(test)]
mod tests;

use atlas_algebra::poly::univariate::{CompressedPoly, UnivariatePolynomial};
use atlas_algebra::{
    field::{challenge::grind_bits_for_loss, JoltField},
    utils::errors::ProofVerifyError,
};
use atlas_transcript::{
    words::{LabelWithCount, WordTranscript},
    Transcript,
};

/// Transcript label of a sumcheck round polynomial.
pub const SUMCHECK_ROUND_TRANSCRIPT_LABEL: &[u8] = b"sumcheck_poly";
/// Transcript label of a batch member's input claim.
const SUMCHECK_CLAIM_TRANSCRIPT_LABEL: &[u8] = b"sumcheck_claim";

/// Security level the per-round proof of work targets.
pub const ROUND_SECURITY_BITS: u32 = 128;

/// Proof-of-work difficulty before the challenge of a sumcheck round whose polynomial has degree
/// `degree`. A cheating round polynomial survives only if the challenge hits one of at most
/// `degree` roots, probability `degree / |C|` per query for challenge set `C`
/// ([`JoltField::challenge_set_size`]); the grind raises each query's cost so the bound reaches
/// `2^-ROUND_SECURITY_BITS` ([`grind_bits_for_loss`]).
pub fn round_pow_bits<F: JoltField>(degree: usize) -> u32 {
    grind_bits_for_loss(
        degree.max(1) as u64,
        F::challenge_set_size(),
        ROUND_SECURITY_BITS,
    )
    .expect("round degrees are far below the proof-of-work limit")
}

/// A sumcheck claim: $\sum_{x \in \{0,1\}^n} g(x) = C$ for $n$ = `num_vars`, with every round
/// polynomial of degree at most `degree`.
#[derive(Clone, Debug)]
pub struct SumcheckClaim<F> {
    /// Number of Boolean variables in the summation.
    pub num_vars: usize,
    /// Maximum degree of each round polynomial.
    pub degree: usize,
    /// The claimed value of the sum.
    pub claimed_sum: F,
}

/// A sumcheck proof as the engine produces and verifies it: one round polynomial per round,
/// each with its linear coefficient omitted (the verifier recovers it from the running sum).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CompressedSumcheckProof<F: JoltField> {
    /// The round polynomials, in round order.
    pub round_polynomials: Vec<CompressedPoly<F>>,
}

/// Why the engine rejected a proof or refused to prove a batch.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SumcheckError<F: JoltField> {
    /// A round polynomial's `s(0) + s(1)` is not the running claim carried from the previous
    /// round.
    #[error("round {round}: expected sum {expected}, got {actual}")]
    RoundCheckFailed {
        /// Zero-indexed round number where the check failed.
        round: usize,
        expected: F,
        /// The computed round sum.
        actual: F,
    },

    /// A round polynomial exceeded the declared degree bound.
    #[error("degree bound exceeded: degree {got}, max {max}")]
    DegreeBoundExceeded {
        /// Actual degree of the offending round polynomial.
        got: usize,
        /// Maximum allowed degree from the claim.
        max: usize,
    },

    /// A batch with rounds to prove declared a maximum degree of zero.
    #[error("batch declares max degree 0 with {max_num_vars} rounds to prove")]
    ZeroBatchDegree {
        /// Number of rounds that would need round polynomials.
        max_num_vars: usize,
    },

    /// A batch member declared more rounds than the batch contains.
    #[error(
        "batch member {member} has {rounds} rounds, exceeding the batch's {max_num_vars} rounds"
    )]
    BatchMemberRoundsOutOfRange {
        /// Zero-indexed member position (declaration order).
        member: usize,
        /// Round count declared for this member.
        rounds: usize,
        /// The batch's total round count.
        max_num_vars: usize,
    },

    /// The field implementation cannot represent this batch's padding scale.
    #[error("batch member {member} requires unsupported padding exponent {exponent}")]
    BatchPaddingExponentOutOfRange {
        /// Zero-indexed member position (declaration order).
        member: usize,
        /// `max_num_vars - rounds`, which `Ring::mul_pow_2` cannot represent above 255.
        exponent: usize,
    },

    /// A batch member's activation window end overflowed `usize`.
    #[error(
        "batch member {member}: activation window overflow for offset {offset}, rounds {rounds}"
    )]
    BatchMemberWindowOverflow {
        /// Zero-indexed member position (declaration order).
        member: usize,
        /// The member's activation offset.
        offset: usize,
        /// The member's round count.
        rounds: usize,
    },

    /// A batch member's activation window ends past the batch's last round.
    #[error(
        "batch member {member}: window [{offset}, {offset} + {rounds}) exceeds {max_num_vars} rounds"
    )]
    BatchMemberWindowOutOfRange {
        /// Zero-indexed member position (declaration order).
        member: usize,
        /// The member's activation offset.
        offset: usize,
        /// The member's round count.
        rounds: usize,
        /// The batch's total round count.
        max_num_vars: usize,
    },

    /// Adding one to the declared degree overflowed `usize`.
    #[error("sumcheck degree {degree} cannot be represented with its coefficient count")]
    DegreeOverflow { degree: usize },

    /// A compressed round polynomial had no coefficients, so there is nothing to absorb or
    /// evaluate. Any valid compressed round polynomial has degree at least 1.
    #[error("round {round}: compressed round polynomial requires >= 2 coefficients, got {got}")]
    CompressedPolynomialTooShort {
        /// Zero-indexed round number where the malformed polynomial appeared.
        round: usize,
        /// Number of stored coefficients received.
        got: usize,
    },

    /// The number of round polynomials in the proof does not match the claim's variables.
    #[error("expected {expected} rounds, proof contains {got}")]
    WrongNumberOfRounds {
        /// Expected number of rounds (equal to `num_vars`).
        expected: usize,
        /// Actual number of round polynomials in the proof.
        got: usize,
    },

    /// The proof of work before a round challenge did not check.
    #[error("round {round}: proof of work failed")]
    ProofOfWork {
        /// Zero-indexed round number whose challenge was guarded.
        round: usize,
    },

    /// A batched prove was given a member slice whose length disagrees with the prelude.
    #[error("batch prelude describes {expected} members, got {got}")]
    BatchMemberCountMismatch {
        /// Member count in the batch prelude.
        expected: usize,
        /// Member count in the supplied slice.
        got: usize,
    },

    /// A batch member's round count disagrees with the batch prelude.
    #[error("batch member {member}: prelude says {expected} rounds, member reports {got}")]
    BatchMemberRoundsMismatch {
        /// Zero-indexed member position (declaration order).
        member: usize,
        /// Round count recorded in the batch prelude.
        expected: usize,
        /// Round count the member reports.
        got: usize,
    },

    /// The batch's round scheduler returned no round message for an active member. Reported
    /// rather than folded as that member's padding (which would surface only as a round-sum
    /// mismatch).
    #[error("batch member {member}: round scheduler produced no round message")]
    MissingRoundMessage {
        /// Zero-indexed member position (declaration order).
        member: usize,
    },

    /// A round handle came back carrying a member index the batch does not have. The engine
    /// assigns those indices, so the scheduler rewrote one; attributing the message by it
    /// would misfold the round.
    #[error("round scheduler returned member index {member}, but the batch has {members} members")]
    RoundMemberIndexOutOfRange {
        /// The out-of-range index carried by the returned handle.
        member: usize,
        /// Number of members in the batch.
        members: usize,
    },
}

/// One batch member: its input claim (the member's initial running claim), its batching
/// coefficient, its round count and its activation offset. Members are in declaration order,
/// the order their claims are absorbed and their coefficients drawn.
///
/// A member is active for rounds `[offset, offset + rounds)` and contributes the padding
/// polynomial specified by [`BatchPrelude`] outside that window. Tail-aligned members have
/// `offset = max_num_vars - rounds`; head-aligned ones `offset = 0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchMember<F> {
    pub input_claim: F,
    pub coefficient: F,
    pub rounds: usize,
    pub offset: usize,
}

impl<F> BatchMember<F> {
    pub(crate) fn is_active(&self, round: usize) -> bool {
        round >= self.offset && round < self.offset + self.rounds
    }
}

/// The head of a batched sumcheck: the members (declaration order), the combined claim and
/// the batch dimensions.
///
/// # Padding rules
///
/// Write `n = max_num_vars` and `W = [offset, offset + rounds)` for a member's active window.
/// The field selects the rule: invertible `2` gives constant extension; otherwise the summand
/// is zero-extended.
///
/// Under constant extension the input claim is multiplied by `2^(n - rounds)`. Each inactive
/// round contributes the constant `claim / 2` and halves the running member claim. Active
/// members receive that running claim and must return polynomials at its scale, including any
/// remaining padding. The output multiplier is one.
///
/// Under zero extension the summand is multiplied by `∏_{j ∉ W} (1 - x_j)`, so the input claim
/// has multiplier one. The engine keeps a native claim `m` (initially the input claim) and a
/// padding factor `p` (initially one). An inactive round contributes `p * m * (1 - X)` and
/// updates only `p *= 1 - r_j`. An active round passes `m` to the member, folds its native
/// polynomial `s` with multiplier `p`, and updates only `m = s(r_j)`. The returned member claim
/// is `p * m`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchPrelude<F> {
    pub members: Vec<BatchMember<F>>,
    /// The padded random linear combination of the members' input claims: the batch's initial
    /// running claim, under the padding rule above.
    pub claimed_sum: F,
    pub max_num_vars: usize,
    pub max_degree: usize,
}

impl<F: JoltField> BatchPrelude<F> {
    /// Combine `members` into the initial claim using the [padding rule](Self), rejecting
    /// invalid batch dimensions or activation windows.
    pub fn try_new(
        members: Vec<BatchMember<F>>,
        max_num_vars: usize,
        max_degree: usize,
    ) -> Result<Self, SumcheckError<F>> {
        validate_batch_dimensions(&members, max_num_vars, max_degree)?;
        let padding = PaddingRule::for_field();
        let claimed_sum = members
            .iter()
            .map(|member| {
                member.coefficient
                    * padding.pad_input_claim(member.input_claim, max_num_vars - member.rounds)
            })
            .sum();
        Ok(Self {
            members,
            claimed_sum,
            max_num_vars,
            max_degree,
        })
    }

    pub(crate) fn validate(&self) -> Result<(), SumcheckError<F>> {
        validate_batch_dimensions(&self.members, self.max_num_vars, self.max_degree)
    }
}

pub(crate) enum PaddingRule<F> {
    ConstantExtension { two_inv: F },
    ZeroExtension,
}

impl<F: JoltField> PaddingRule<F> {
    pub(crate) fn for_field() -> Self {
        match jolt_field::Field::inverse(&<F as jolt_field::Ring>::from_u64(2)) {
            Some(two_inv) => Self::ConstantExtension { two_inv },
            None => Self::ZeroExtension,
        }
    }

    pub(crate) fn pad_input_claim(&self, claim: F, exponent: usize) -> F {
        match self {
            Self::ConstantExtension { .. } => jolt_field::Ring::mul_pow_2(&claim, exponent),
            Self::ZeroExtension => claim,
        }
    }
}

fn validate_batch_dimensions<F: JoltField>(
    members: &[BatchMember<F>],
    max_num_vars: usize,
    max_degree: usize,
) -> Result<(), SumcheckError<F>> {
    if max_degree.checked_add(1).is_none() {
        return Err(SumcheckError::DegreeOverflow { degree: max_degree });
    }
    for (member, described) in members.iter().enumerate() {
        let exponent = max_num_vars.checked_sub(described.rounds).ok_or(
            SumcheckError::BatchMemberRoundsOutOfRange {
                member,
                rounds: described.rounds,
                max_num_vars,
            },
        )?;
        // `Ring::mul_pow_2` panics above 255.
        if exponent > 255 {
            return Err(SumcheckError::BatchPaddingExponentOutOfRange { member, exponent });
        }
        let window_end = described.offset.checked_add(described.rounds).ok_or(
            SumcheckError::BatchMemberWindowOverflow {
                member,
                offset: described.offset,
                rounds: described.rounds,
            },
        )?;
        if window_end > max_num_vars {
            return Err(SumcheckError::BatchMemberWindowOutOfRange {
                member,
                offset: described.offset,
                rounds: described.rounds,
                max_num_vars,
            });
        }
    }
    if max_num_vars > 0 && max_degree == 0 {
        return Err(SumcheckError::ZeroBatchDegree { max_num_vars });
    }
    Ok(())
}

/// Absorb a batch's input claims, each under `b"sumcheck_claim"`, on the prover and the
/// verifier alike (before the batching coefficients are drawn).
pub fn absorb_input_claims<F: JoltField, T: Transcript>(input_claims: &[F], transcript: &mut T) {
    for input_claim in input_claims {
        transcript.append_labeled(SUMCHECK_CLAIM_TRANSCRIPT_LABEL, input_claim);
    }
}

/// Verify a compressed sumcheck proof of `claim` and return the reduced claim `(v, r)`: the
/// final running sum and the round challenges.
///
/// Per round: the degree bound is enforced, the round polynomial is absorbed under
/// [`SUMCHECK_ROUND_TRANSCRIPT_LABEL`], the proof of work is checked and the challenge `r_i`
/// squeezed, and the running sum becomes the round polynomial at `r_i` (its linear term
/// recovered from the running sum). The caller checks `v` against the oracle.
///
/// When `claim.num_vars == 0` nothing is absorbed or checked and `v` is the claimed sum.
#[tracing::instrument(skip_all, name = "verify_compressed")]
pub fn verify_compressed<F: JoltField, T: Transcript>(
    claim: &SumcheckClaim<F>,
    proof: &CompressedSumcheckProof<F>,
    transcript: &mut T,
) -> Result<(F, Vec<F>), SumcheckError<F>> {
    if proof.round_polynomials.len() != claim.num_vars {
        return Err(SumcheckError::WrongNumberOfRounds {
            expected: claim.num_vars,
            got: proof.round_polynomials.len(),
        });
    }
    let pow_bits = round_pow_bits::<F>(claim.degree);
    let mut running_sum = claim.claimed_sum;
    let mut challenges = Vec::with_capacity(claim.num_vars);

    for (round, round_proof) in proof.round_polynomials.iter().enumerate() {
        if round_proof.degree() > claim.degree {
            return Err(SumcheckError::DegreeBoundExceeded {
                got: round_proof.degree(),
                max: claim.degree,
            });
        }
        let coeffs = round_proof.coeffs_except_linear_term();
        if coeffs.is_empty() {
            return Err(SumcheckError::CompressedPolynomialTooShort { round, got: 0 });
        }
        transcript.append(&LabelWithCount(
            SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            coeffs.len() as u64,
        ));
        for coeff in coeffs {
            transcript.append(coeff);
        }
        let r = draw_challenge::<F, T>(transcript, pow_bits)
            .map_err(|_| SumcheckError::ProofOfWork { round })?;
        running_sum = round_proof.evaluate_with_hint(running_sum, r);
        challenges.push(r);
    }

    Ok((running_sum, challenges))
}

/// A round challenge: the proof of work of difficulty `pow_bits`, then the challenge as a field
/// element.
fn draw_challenge<F: JoltField, T: Transcript>(
    transcript: &mut T,
    pow_bits: u32,
) -> Result<F, ProofVerifyError> {
    transcript.pow(pow_bits)?;
    Ok(transcript.challenge_scalar_optimized::<F>().into())
}
