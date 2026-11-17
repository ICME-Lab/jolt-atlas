//! The prover side of the engine: the batched round loop [`prove_batch`], which sends each
//! round message through the transcript, the member round interface [`ProveRounds`] and the
//! per-round traversal seam [`RoundScheduler`].

use atlas_algebra::field::JoltField;
use atlas_algebra::poly::univariate::{UnivariatePoly, UnivariatePolynomial};
use atlas_transcript::{
    words::{LabelWithCount, WordTranscript},
    ProvingTranscript,
};

use super::{
    draw_challenge, round_pow_bits, BatchPrelude, PaddingRule, SumcheckError,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};

/// One batch member's prove-side round interface, consumed by [`prove_batch`]'s round loop.
/// Object-safe: a batch's members are heterogeneous, and the engine takes them as
/// `&mut [&mut dyn ProveRounds<F>]`.
///
/// `round` indices are member-local (`0..num_rounds()`): a member is consulted only during its
/// active window `[offset, offset + rounds)`, and never learns the batch's challenges outside
/// it (its opening point is the length-`rounds` slice of the batch point starting at its
/// offset).
///
/// The round API is fused: [`prove_round`](Self::prove_round) delivers the previous active
/// round's challenge together with the current round's message request, so a member can bind
/// and evaluate in a single pass over its tables. `bind` is `None` exactly on the member's
/// first active round, and the final active round's challenge arrives through
/// [`finish_rounds`](Self::finish_rounds).
pub trait ProveRounds<F: JoltField> {
    fn num_rounds(&self) -> usize;

    /// Bind `bind` (the member's previous active round's challenge, `None` on its first active
    /// round), then compute the round polynomial for member-local `round`. `previous_claim` is
    /// the member's own running claim at the scale specified by [`BatchPrelude`]. The returned
    /// polynomial must satisfy `s(0) + s(1) == previous_claim`.
    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>>;

    /// Bind `bind`, the challenge of member-local `round`, into state the member shares with
    /// other members of its batch. A [`RoundScheduler`] calls this for every member, one at a
    /// time, before the round's [`prove_round`]s (and before the [`finish_rounds`]s), which it
    /// may then run in parallel: those take no exclusive access to shared state. A member
    /// with shared state binds it here and nowhere else; the default has none.
    ///
    /// [`prove_round`]: Self::prove_round
    /// [`finish_rounds`]: Self::finish_rounds
    fn bind_shared(&mut self, _bind: F, _round: usize) {}

    /// Bind the member's final active round's challenge. Called once after the batch's round
    /// loop, for every member that was ever active; the member is fully bound afterwards.
    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>>;
}

/// The engine error for a round message whose `s(0) + s(1)` is `actual` instead of
/// `previous_claim`.
pub fn round_check_failed<F: JoltField>(
    round: usize,
    previous_claim: F,
) -> impl FnOnce(F) -> SumcheckError<F> {
    move |actual| SumcheckError::RoundCheckFailed {
        round,
        expected: previous_claim,
        actual,
    }
}

/// One active member's round request, handed to the [`RoundScheduler`].
pub struct MemberRound<'a, F: JoltField> {
    pub index: usize,
    pub local_round: usize,
    pub bind: Option<F>,
    pub claim: F,
    pub member: &'a mut dyn ProveRounds<F>,
    pub message: Option<UnivariatePoly<F>>,
}

impl<F: JoltField> MemberRound<'_, F> {
    pub fn run(&mut self) -> Result<(), SumcheckError<F>> {
        self.message = Some(
            self.member
                .prove_round(self.bind, self.local_round, self.claim)?,
        );
        Ok(())
    }
}

/// One ever-active member and its final bind, for after the round loop.
pub struct MemberFinish<'a, F: JoltField> {
    pub bind: F,
    pub member: &'a mut dyn ProveRounds<F>,
}

impl<F: JoltField> MemberFinish<'_, F> {
    pub fn run(&mut self) -> Result<(), SumcheckError<F>> {
        self.member.finish_rounds(self.bind)
    }
}

/// How a batch's rounds visit its members. Order and transport are free; activity, padding,
/// fold, round-sum checks and transcript stay in [`prove_batch`]. Leaving a handle's message
/// unset is [`SumcheckError::MissingRoundMessage`].
pub trait RoundScheduler<F: JoltField> {
    fn batch_prove_round(
        &mut self,
        work: &mut [MemberRound<'_, F>],
    ) -> Result<(), SumcheckError<F>>;

    fn batch_finish_rounds(
        &mut self,
        finishes: &mut [MemberFinish<'_, F>],
    ) -> Result<(), SumcheckError<F>>;
}

/// A proved batch: the round challenges (the batch opening point), the final combined running
/// claim (what the verifier's reduction reproduces), and each member's final bound claim in
/// declaration order, including the output multiplier documented on [`BatchPrelude`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProvedBatch<F> {
    pub challenges: Vec<F>,
    pub final_claim: F,
    pub member_claims: Vec<F>,
}

/// Send round `round`'s message and draw its challenge behind a proof of work of `pow_bits`.
/// `coefficients` are the round polynomial's, in increasing order, one per slot up to the
/// batch's degree: the message is all of them but the linear one, so every round of a batch
/// sends the same number of coefficients, which is what [`super::verify_compressed`] reads.
fn send_round<F: JoltField, T: ProvingTranscript>(
    round: usize,
    coefficients: &[F],
    pow_bits: u32,
    transcript: &mut T,
) -> Result<F, SumcheckError<F>> {
    let message: Vec<F> = coefficients
        .iter()
        .enumerate()
        .filter(|(power, _)| *power != 1)
        .map(|(_, coefficient)| *coefficient)
        .collect();
    transcript.append(&LabelWithCount(
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        message.len() as u64,
    ));
    transcript.send_scalars(&message);
    draw_challenge::<F, T>(transcript, pow_bits).map_err(|_| SumcheckError::ProofOfWork { round })
}

/// Prove one batched sumcheck. Per round: combine the active members' round polynomials with
/// their batching coefficients and the padding rule of [`BatchPrelude`], self-check
/// `s(0) + s(1)` against the running claim, send the round message and draw the challenge
/// behind the proof of work of [`round_pow_bits`] for `prelude.max_degree`, and
/// stash the challenge as each active member's pending bind, delivered with its next
/// `prove_round` or, after the loop, through `finish_rounds`.
///
/// The caller absorbs the input claims and draws the batching coefficients before
/// ([`super::absorb_input_claims`]) and appends the members' output claims after.
#[tracing::instrument(
    skip_all,
    name = "prove_batch",
    fields(num_rounds = prelude.max_num_vars, members = members.len())
)]
pub fn prove_batch<F: JoltField, T: ProvingTranscript>(
    prelude: &BatchPrelude<F>,
    members: &mut [&mut dyn ProveRounds<F>],
    scheduler: &mut dyn RoundScheduler<F>,
    transcript: &mut T,
) -> Result<ProvedBatch<F>, SumcheckError<F>> {
    if members.len() != prelude.members.len() {
        return Err(SumcheckError::BatchMemberCountMismatch {
            expected: prelude.members.len(),
            got: members.len(),
        });
    }
    for (index, (member, described)) in members.iter().zip(&prelude.members).enumerate() {
        if member.num_rounds() != described.rounds {
            return Err(SumcheckError::BatchMemberRoundsMismatch {
                member: index,
                expected: described.rounds,
                got: member.num_rounds(),
            });
        }
    }
    prelude.validate()?;
    let max_num_vars = prelude.max_num_vars;
    let pow_bits = round_pow_bits::<F>(prelude.max_degree);

    let coefficient_count = prelude.max_degree + 1;
    let mut running_claim = prelude.claimed_sum;
    let mut challenges = Vec::with_capacity(max_num_vars);
    // Each active member's not-yet-delivered previous-round challenge: filled when the round
    // challenge is drawn, consumed by the member's next `prove_round` (or `finish_rounds`).
    let mut pending_binds: Vec<Option<F>> = vec![None; members.len()];

    let padding_rule = PaddingRule::for_field();
    // `padding` stays one under constant extension, where `native` carries the member's
    // running claim at the padded scale.
    struct MemberClaim<F> {
        native: F,
        padding: F,
    }
    let mut claims: Vec<MemberClaim<F>> = prelude
        .members
        .iter()
        .map(|member| MemberClaim {
            native: padding_rule.pad_input_claim(member.input_claim, max_num_vars - member.rounds),
            padding: F::one(),
        })
        .collect();

    for round in 0..max_num_vars {
        let _round_span = tracing::info_span!("sumcheck_round", round).entered();

        let mut batched_coefficients = vec![F::zero(); coefficient_count];
        let mut work: Vec<MemberRound<'_, F>> = Vec::with_capacity(members.len());
        for (index, ((member, described), (claim, pending_bind))) in members
            .iter_mut()
            .zip(&prelude.members)
            .zip(claims.iter_mut().zip(pending_binds.iter_mut()))
            .enumerate()
        {
            if !described.is_active(round) {
                match &padding_rule {
                    PaddingRule::ConstantExtension { two_inv } => {
                        claim.native *= *two_inv;
                        if let Some(constant) = batched_coefficients.first_mut() {
                            *constant += described.coefficient * claim.native;
                        }
                    }
                    PaddingRule::ZeroExtension => {
                        let contribution = described.coefficient * claim.padding * claim.native;
                        if let Some(constant) = batched_coefficients.first_mut() {
                            *constant += contribution;
                        }
                        if let Some(linear) = batched_coefficients.get_mut(1) {
                            *linear -= contribution;
                        }
                    }
                }
                continue;
            }
            work.push(MemberRound {
                index,
                local_round: round - described.offset,
                bind: pending_bind.take(),
                claim: claim.native,
                member: &mut **member,
                message: None,
            });
        }
        scheduler.batch_prove_round(&mut work)?;

        for item in &work {
            let poly = item
                .message
                .as_ref()
                .ok_or(SumcheckError::MissingRoundMessage { member: item.index })?;
            let poly_degree = UnivariatePolynomial::degree(poly);
            if poly_degree > prelude.max_degree {
                return Err(SumcheckError::DegreeBoundExceeded {
                    got: poly_degree,
                    max: prelude.max_degree,
                });
            }
            let out_of_range = || SumcheckError::RoundMemberIndexOutOfRange {
                member: item.index,
                members: prelude.members.len(),
            };
            let described = prelude.members.get(item.index).ok_or_else(out_of_range)?;
            let claim = claims.get(item.index).ok_or_else(out_of_range)?;
            let scale = described.coefficient * claim.padding;
            for (slot, coefficient) in batched_coefficients.iter_mut().zip(poly.coefficients()) {
                *slot += scale * *coefficient;
            }
        }

        let batched_poly = UnivariatePoly::new(batched_coefficients);
        let round_sum = batched_poly.evaluate(F::zero()) + batched_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: running_claim,
                actual: round_sum,
            });
        }

        let challenge = send_round(round, batched_poly.coefficients(), pow_bits, transcript)?;
        running_claim = batched_poly.evaluate(challenge);
        challenges.push(challenge);

        if matches!(padding_rule, PaddingRule::ZeroExtension) {
            for (claim, described) in claims.iter_mut().zip(&prelude.members) {
                if !described.is_active(round) {
                    claim.padding *= F::one() - challenge;
                }
            }
        }
        // Attribution is by member index, not position: a reordering traversal makes
        // declaration-order zip impossible here.
        for item in &work {
            if let Some(poly) = &item.message {
                let out_of_range = || SumcheckError::RoundMemberIndexOutOfRange {
                    member: item.index,
                    members: prelude.members.len(),
                };
                claims.get_mut(item.index).ok_or_else(out_of_range)?.native =
                    poly.evaluate(challenge);
                *pending_binds.get_mut(item.index).ok_or_else(out_of_range)? = Some(challenge);
            }
        }
    }
    let member_claims: Vec<F> = claims
        .into_iter()
        .map(|claim| claim.padding * claim.native)
        .collect();

    let mut finishes: Vec<MemberFinish<'_, F>> = Vec::with_capacity(members.len());
    for (member, bind) in members.iter_mut().zip(pending_binds.iter()) {
        if let Some(bind) = *bind {
            finishes.push(MemberFinish {
                bind,
                member: &mut **member,
            });
        }
    }
    scheduler.batch_finish_rounds(&mut finishes)?;

    Ok(ProvedBatch {
        challenges,
        final_claim: running_claim,
        member_claims,
    })
}
