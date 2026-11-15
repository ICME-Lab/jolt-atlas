//! The engine's tests, over `Fp128` and the Blake2b sponge transcripts: the compressed verifier's
//! rejections, the batched prover against the verifier on twin transcripts (padding and head
//! alignment included), the round-scheduler seam, and the batch prelude's validation.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::indexing_slicing,
    clippy::wildcard_enum_match_arm
)]

use atlas_algebra::field::fp128::Fp128;
use atlas_algebra::poly::univariate::{CompressedPoly, UnivariatePoly};
use atlas_transcript::{
    Blake2bProver, Blake2bVerifier, ProvingTranscript, Transcript, VerifyingTranscript,
};
use jolt_field::{Field as _, Ring as _};
use num_traits::One;

use super::prover::{
    prove_batch, ClearSumcheckRecorder, MemberFinish, MemberRound, ProveRounds, ProvedBatch,
    RoundScheduler,
};
use super::{
    absorb_input_claims, verify_compressed, BatchMember, BatchPrelude, CompressedSumcheckProof,
    SumcheckClaim, SumcheckError,
};

type F = Fp128;

fn f(x: u64) -> F {
    F::from_u64(x)
}

/// The multilinear extension of `evals` at `point`, as `Σ_x evals[x] · eq(x, point)` with the
/// first coordinate the most significant bit of `x`.
fn mle(evals: &[F], point: &[F]) -> F {
    let n = point.len();
    evals
        .iter()
        .enumerate()
        .map(|(x, &value)| {
            point.iter().enumerate().fold(value, |acc, (j, &r)| {
                if (x >> (n - 1 - j)) & 1 == 1 {
                    acc * r
                } else {
                    acc * (F::one() - r)
                }
            })
        })
        .sum()
}

/// A dense multilinear member bound high-to-low, with a prescribed total sum.
struct DenseMember {
    evals: Vec<F>,
    original: Vec<F>,
    num_rounds: usize,
}

impl DenseMember {
    fn with_sum(num_rounds: usize, sum: F, seed: u64) -> Self {
        let mut evals: Vec<F> = (0..1u64 << num_rounds)
            .map(|i| f(seed + 31 * i + 11))
            .collect();
        let current: F = evals.iter().copied().sum();
        evals[0] += sum - current;
        Self {
            original: evals.clone(),
            evals,
            num_rounds,
        }
    }

    fn final_eval(&self) -> F {
        assert_eq!(self.evals.len(), 1, "member is fully bound");
        self.evals[0]
    }

    fn bind(&mut self, challenge: F) {
        let half = self.evals.len() / 2;
        for i in 0..half {
            self.evals[i] = self.evals[i] + challenge * (self.evals[i + half] - self.evals[i]);
        }
        self.evals.truncate(half);
    }
}

impl ProveRounds<F> for DenseMember {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.evals.len() / 2;
        let eval_0: F = self.evals[..half].iter().copied().sum();
        let eval_1: F = self.evals[half..].iter().copied().sum();
        assert_eq!(eval_0 + eval_1, previous_claim);
        Ok(UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

/// Declaration-order traversal.
struct Sequential;

impl RoundScheduler<F> for Sequential {
    fn batch_prove_round(
        &mut self,
        work: &mut [MemberRound<'_, F>],
    ) -> Result<(), SumcheckError<F>> {
        work.iter_mut().try_for_each(|item| item.run())
    }

    fn batch_finish_rounds(
        &mut self,
        finishes: &mut [MemberFinish<'_, F>],
    ) -> Result<(), SumcheckError<F>> {
        finishes.iter_mut().try_for_each(|item| item.run())
    }
}

/// Reverse-order traversal.
struct Reversed;

impl RoundScheduler<F> for Reversed {
    fn batch_prove_round(
        &mut self,
        work: &mut [MemberRound<'_, F>],
    ) -> Result<(), SumcheckError<F>> {
        work.iter_mut().rev().try_for_each(|item| item.run())
    }

    fn batch_finish_rounds(
        &mut self,
        finishes: &mut [MemberFinish<'_, F>],
    ) -> Result<(), SumcheckError<F>> {
        finishes.iter_mut().rev().try_for_each(|item| item.run())
    }
}

fn claim(num_vars: usize, degree: usize, claimed_sum: F) -> SumcheckClaim<F> {
    SumcheckClaim {
        num_vars,
        degree,
        claimed_sum,
    }
}

/// Prove `members` (described by `rounds` and `offsets`, with input claims `sums`) as one batch:
/// absorb the claims, draw one coefficient per member, run the engine.
fn prove_members(
    label: &'static [u8],
    members: &mut [&mut dyn ProveRounds<F>],
    sums: &[F],
    offsets: &[usize],
    max_num_vars: usize,
    scheduler: &mut dyn RoundScheduler<F>,
) -> (
    Blake2bProver,
    BatchPrelude<F>,
    ProvedBatch<F>,
    CompressedSumcheckProof<F>,
) {
    let mut transcript = Blake2bProver::new(label);
    absorb_input_claims(sums, &mut transcript);
    let described = members
        .iter()
        .zip(sums)
        .zip(offsets)
        .map(|((member, &input_claim), &offset)| BatchMember {
            input_claim,
            coefficient: transcript.challenge_scalar::<F>(),
            rounds: member.num_rounds(),
            offset,
        })
        .collect();
    let prelude = BatchPrelude::try_new(described, max_num_vars, 1).unwrap();
    let mut recorder = ClearSumcheckRecorder::new();
    let proved = prove_batch(&prelude, members, scheduler, &mut recorder, &mut transcript).unwrap();
    (transcript, prelude, proved, recorder.into_proof())
}

/// Replay the batch head on a verifier transcript, verify the proof, and check both transcripts
/// end in the same state (the next challenge agrees).
fn verify_twin(
    label: &'static [u8],
    mut prover: Blake2bProver,
    sums: &[F],
    claimed_sum: F,
    max_num_vars: usize,
    proof: &CompressedSumcheckProof<F>,
) -> (F, Vec<F>) {
    let mut verifier = Blake2bVerifier::new(label, prover.narg().to_vec());
    absorb_input_claims(sums, &mut verifier);
    let _: Vec<F> = sums.iter().map(|_| verifier.challenge_scalar()).collect();
    let reduced =
        verify_compressed(&claim(max_num_vars, 1, claimed_sum), proof, &mut verifier).unwrap();
    assert_eq!(
        prover.challenge_scalar::<F>(),
        verifier.challenge_scalar::<F>(),
        "prover and verifier transcripts diverged"
    );
    reduced
}

#[test]
fn single_member_reduces_to_the_multilinear_evaluation() {
    let sum = f(1234);
    let mut member = DenseMember::with_sum(3, sum, 5);
    let original = member.original.clone();
    let label = b"single-member";
    let (transcript, prelude, proved, proof) =
        prove_members(label, &mut [&mut member], &[sum], &[0], 3, &mut Sequential);
    let (value, point) = verify_twin(label, transcript, &[sum], prelude.claimed_sum, 3, &proof);

    assert_eq!(point, proved.challenges);
    assert_eq!(value, proved.final_claim);
    // The batch of one has coefficient c: the reduced value is c · g(r).
    let coefficient = prelude.members[0].coefficient;
    assert_eq!(value, coefficient * mle(&original, &point));
    assert_eq!(member.final_eval(), mle(&original, &point));
}

#[test]
fn tampered_round_does_not_reduce_to_the_evaluation() {
    let sum = f(77);
    let mut member = DenseMember::with_sum(2, sum, 3);
    let original = member.original.clone();
    let label = b"tampered-round";
    let (transcript, prelude, _, mut proof) =
        prove_members(label, &mut [&mut member], &[sum], &[0], 2, &mut Sequential);
    let mut coeffs = proof.round_polynomials[0]
        .coeffs_except_linear_term()
        .to_vec();
    coeffs[0] += F::one();
    proof.round_polynomials[0] = CompressedPoly::new(coeffs);

    let mut verifier = Blake2bVerifier::new(label, transcript.narg().to_vec());
    absorb_input_claims(&[sum], &mut verifier);
    let _: F = verifier.challenge_scalar();
    let (value, point) =
        verify_compressed(&claim(2, 1, prelude.claimed_sum), &proof, &mut verifier).unwrap();
    assert_ne!(
        value,
        prelude.members[0].coefficient * mle(&original, &point)
    );
}

#[test]
fn verify_rejects_wrong_round_count() {
    let proof = CompressedSumcheckProof {
        round_polynomials: Vec::new(),
    };
    let mut transcript = Blake2bVerifier::new(b"sumcheck-test", Vec::new());
    assert!(matches!(
        verify_compressed(&claim(1, 1, f(0)), &proof, &mut transcript),
        Err(SumcheckError::WrongNumberOfRounds {
            expected: 1,
            got: 0
        })
    ));
}

#[test]
fn verify_rejects_degree_bound_exceeded() {
    let proof = CompressedSumcheckProof {
        round_polynomials: vec![CompressedPoly::new(vec![f(1), f(2)])],
    };
    let mut transcript = Blake2bVerifier::new(b"sumcheck-test", Vec::new());
    assert!(matches!(
        verify_compressed(&claim(1, 1, f(3)), &proof, &mut transcript),
        Err(SumcheckError::DegreeBoundExceeded { got: 2, max: 1 })
    ));
}

#[test]
fn verify_rejects_empty_round_polynomial() {
    let proof = CompressedSumcheckProof {
        round_polynomials: vec![CompressedPoly::new(Vec::new())],
    };
    let mut transcript = Blake2bVerifier::new(b"sumcheck-test", Vec::new());
    assert!(matches!(
        verify_compressed(&claim(1, 1, f(0)), &proof, &mut transcript),
        Err(SumcheckError::CompressedPolynomialTooShort { round: 0, got: 0 })
    ));
}

#[test]
fn verify_rejects_a_missing_proof_of_work() {
    let sum = f(5);
    let mut member = DenseMember::with_sum(1, sum, 9);
    let (_, prelude, _, proof) = prove_members(
        b"missing-pow",
        &mut [&mut member],
        &[sum],
        &[0],
        1,
        &mut Sequential,
    );
    // A verifier transcript without the prover's NARG bytes has no nonce to check.
    let mut verifier = Blake2bVerifier::new(b"missing-pow", Vec::new());
    absorb_input_claims(&[sum], &mut verifier);
    let _: F = verifier.challenge_scalar();
    assert!(matches!(
        verify_compressed(&claim(1, 1, prelude.claimed_sum), &proof, &mut verifier),
        Err(SumcheckError::ProofOfWork { round: 0 })
    ));
}

#[test]
fn prove_batch_rejects_zero_max_degree() {
    let sum = f(42);
    let mut member = DenseMember::with_sum(2, sum, 7);
    let prelude = BatchPrelude {
        members: vec![BatchMember {
            input_claim: sum,
            coefficient: F::one(),
            rounds: 2,
            offset: 0,
        }],
        claimed_sum: sum,
        max_num_vars: 2,
        max_degree: 0,
    };
    let mut transcript = Blake2bProver::new(b"zero-degree-batch");
    let result = prove_batch(
        &prelude,
        &mut [&mut member],
        &mut Sequential,
        &mut ClearSumcheckRecorder::new(),
        &mut transcript,
    );
    assert!(matches!(
        result,
        Err(SumcheckError::ZeroBatchDegree { max_num_vars: 2 })
    ));
}

/// A 3-round and a tail-aligned 1-round member: the verifier's reduction lands on the prover's
/// final claim and challenges, and the short member's claim is padded by `2^2`.
#[test]
fn prove_batch_twin_with_padding() {
    let (sum_long, sum_short) = (f(1234), f(777));
    let mut long = DenseMember::with_sum(3, sum_long, 5);
    let mut short = DenseMember::with_sum(1, sum_short, 91);
    let label = b"prove-batch-twin";
    let (transcript, prelude, proved, proof) = prove_members(
        label,
        &mut [&mut long, &mut short],
        &[sum_long, sum_short],
        &[0, 2],
        3,
        &mut Sequential,
    );
    assert_eq!(
        proved.member_claims,
        vec![long.final_eval(), short.final_eval()]
    );
    let [coeff_long, coeff_short] = [0, 1].map(|i| prelude.members[i].coefficient);
    assert_eq!(
        prelude.claimed_sum,
        coeff_long * sum_long + coeff_short * sum_short * f(4)
    );

    let (value, point) = verify_twin(
        label,
        transcript,
        &[sum_long, sum_short],
        prelude.claimed_sum,
        3,
        &proof,
    );
    assert_eq!(value, proved.final_claim);
    assert_eq!(point, proved.challenges);
    assert_eq!(
        value,
        coeff_long * mle(&long.original, &point) + coeff_short * mle(&short.original, &point[2..])
    );
}

/// A head-aligned 1-round member is active in the first round and halves through the trailing
/// rounds: its table sums to `2^2 · input_claim`, and its final batch claim is the bound value
/// with that padding halved back out.
#[test]
fn prove_batch_twin_head_aligned_member() {
    let (sum_long, sum_short) = (f(1234), f(777));
    let mut long = DenseMember::with_sum(3, sum_long, 5);
    let mut short = DenseMember::with_sum(1, sum_short * f(4), 91);
    let label = b"prove-batch-head-twin";
    let (transcript, prelude, proved, proof) = prove_members(
        label,
        &mut [&mut long, &mut short],
        &[sum_long, sum_short],
        &[0, 0],
        3,
        &mut Sequential,
    );
    let quarter = f(4).inverse().unwrap();
    assert_eq!(
        proved.member_claims,
        vec![long.final_eval(), short.final_eval() * quarter]
    );

    let (value, point) = verify_twin(
        label,
        transcript,
        &[sum_long, sum_short],
        prelude.claimed_sum,
        3,
        &proof,
    );
    assert_eq!(value, proved.final_claim);
    assert_eq!(point, proved.challenges);
}

fn two_member_batch(
    scheduler: &mut dyn RoundScheduler<F>,
) -> Result<(ProvedBatch<F>, CompressedSumcheckProof<F>, F), SumcheckError<F>> {
    let (sum_long, sum_short) = (f(4242), f(99));
    let mut long = DenseMember::with_sum(3, sum_long, 7);
    let mut short = DenseMember::with_sum(1, sum_short, 13);
    let mut transcript = Blake2bProver::new(b"traversal-seam");
    absorb_input_claims(&[sum_long, sum_short], &mut transcript);
    let (coeff_long, coeff_short) = (transcript.challenge_scalar(), transcript.challenge_scalar());
    let prelude = BatchPrelude::try_new(
        vec![
            BatchMember {
                input_claim: sum_long,
                coefficient: coeff_long,
                rounds: 3,
                offset: 0,
            },
            BatchMember {
                input_claim: sum_short,
                coefficient: coeff_short,
                rounds: 1,
                offset: 2,
            },
        ],
        3,
        1,
    )
    .unwrap();
    let mut recorder = ClearSumcheckRecorder::new();
    let proved = prove_batch(
        &prelude,
        &mut [&mut long, &mut short],
        scheduler,
        &mut recorder,
        &mut transcript,
    )?;
    Ok((proved, recorder.into_proof(), transcript.challenge_scalar()))
}

#[test]
fn proof_is_invariant_under_a_reordering_traversal() {
    let sequential = two_member_batch(&mut Sequential).unwrap();
    let reversed = two_member_batch(&mut Reversed).unwrap();
    assert_eq!(sequential.0, reversed.0);
    assert_eq!(sequential.1, reversed.1);
    assert_eq!(sequential.2, reversed.2);
}

#[test]
fn skipping_an_active_member_is_reported_against_that_member() {
    struct SkipsMemberOne;
    impl RoundScheduler<F> for SkipsMemberOne {
        fn batch_prove_round(
            &mut self,
            work: &mut [MemberRound<'_, F>],
        ) -> Result<(), SumcheckError<F>> {
            work.iter_mut()
                .filter(|item| item.index != 1)
                .try_for_each(|item| item.run())
        }

        fn batch_finish_rounds(
            &mut self,
            _finishes: &mut [MemberFinish<'_, F>],
        ) -> Result<(), SumcheckError<F>> {
            Ok(())
        }
    }

    let error = two_member_batch(&mut SkipsMemberOne).unwrap_err();
    assert!(
        matches!(error, SumcheckError::MissingRoundMessage { member: 1 }),
        "expected MissingRoundMessage for member 1, got {error:?}"
    );
}

#[test]
fn a_rewritten_member_index_is_rejected() {
    struct RewritesIndex;
    impl RoundScheduler<F> for RewritesIndex {
        fn batch_prove_round(
            &mut self,
            work: &mut [MemberRound<'_, F>],
        ) -> Result<(), SumcheckError<F>> {
            for item in work.iter_mut() {
                item.run()?;
                item.index = usize::MAX;
            }
            Ok(())
        }

        fn batch_finish_rounds(
            &mut self,
            _finishes: &mut [MemberFinish<'_, F>],
        ) -> Result<(), SumcheckError<F>> {
            Ok(())
        }
    }

    let error = two_member_batch(&mut RewritesIndex).unwrap_err();
    assert!(
        matches!(
            error,
            SumcheckError::RoundMemberIndexOutOfRange {
                member: usize::MAX,
                members: 2
            }
        ),
        "expected RoundMemberIndexOutOfRange, got {error:?}"
    );
}

fn member(rounds: usize, offset: usize) -> BatchMember<F> {
    BatchMember {
        input_claim: F::one(),
        coefficient: F::one(),
        rounds,
        offset,
    }
}

#[test]
fn prelude_rejects_rounds_larger_than_batch() {
    assert!(matches!(
        BatchPrelude::try_new(vec![member(3, 0)], 2, 1),
        Err(SumcheckError::BatchMemberRoundsOutOfRange {
            member: 0,
            rounds: 3,
            max_num_vars: 2
        })
    ));
}

#[test]
fn prelude_rejects_window_overflow() {
    assert!(matches!(
        BatchPrelude::try_new(vec![member(1, usize::MAX)], 2, 1),
        Err(SumcheckError::BatchMemberWindowOverflow {
            member: 0,
            offset: usize::MAX,
            rounds: 1
        })
    ));
}

#[test]
fn prelude_rejects_window_outside_batch() {
    assert!(matches!(
        BatchPrelude::try_new(vec![member(2, 1)], 2, 1),
        Err(SumcheckError::BatchMemberWindowOutOfRange {
            member: 0,
            offset: 1,
            rounds: 2,
            max_num_vars: 2
        })
    ));
}

#[test]
fn prelude_rejects_degree_overflow() {
    assert!(matches!(
        BatchPrelude::<F>::try_new(Vec::new(), 0, usize::MAX),
        Err(SumcheckError::DegreeOverflow { degree: usize::MAX })
    ));
}
