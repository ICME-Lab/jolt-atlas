use ark_bn254::Fr;
use ark_ff::{One, Zero};
use joltworks::transcripts::Transcript;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundPolynomial<const D: usize> {
    pub coeffs: [Fr; D],
}

impl<const D: usize> RoundPolynomial<D> {
    pub fn eval(&self, point: Fr) -> Fr {
        self.coeffs
            .iter()
            .rev()
            .fold(Fr::from(0_u64), |acc, coeff| acc * point + coeff)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumCheckRounds<const D: usize> {
    pub round_polys: Vec<RoundPolynomial<D>>,
    pub final_claim: Fr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedSumcheck {
    pub final_claim: Fr,
    pub point: Vec<Fr>,
    pub challenges: Vec<<Fr as joltworks::field::JoltField>::Challenge>,
}

pub fn append_round_statement<T, const D: usize>(
    transcript: &mut T,
    claim: Fr,
    round: &RoundPolynomial<D>,
) where
    T: Transcript,
{
    transcript.append_message(b"q3/sc/round/v1");
    transcript.append_scalar(&claim);
    transcript.append_scalars(&round.coeffs);
}

pub fn verify_sumcheck_rounds<T, const D: usize>(
    mut claim: Fr,
    rounds: &SumCheckRounds<D>,
    expected_rounds: usize,
    transcript: &mut T,
) -> Option<VerifiedSumcheck>
where
    T: Transcript,
{
    // Generic sumcheck verifier for a univariate round-message transcript.
    // For each round polynomial g_i, the previous claim must satisfy:
    //
    //   claim_i = g_i(0) + g_i(1)
    //
    // After absorbing g_i, the transcript challenge r_i updates the claim to
    // g_i(r_i).  The concrete op verifier checks the final claim against its
    // own relation evaluated at the derived point (r_0, ..., r_{n-1}).
    let mut round_polys = rounds.round_polys.iter();
    let mut challenges = Vec::with_capacity(expected_rounds);

    for _ in 0..expected_rounds {
        let round = round_polys.next()?;
        (claim == round.eval(Fr::zero()) + round.eval(Fr::one())).then_some(())?;

        append_round_statement(transcript, claim, round);
        let challenge = transcript.challenge_scalar_optimized::<Fr>();
        claim = round.eval(challenge.into());
        challenges.push(challenge);
    }
    round_polys.next().is_none().then_some(())?;

    (claim == rounds.final_claim).then_some(VerifiedSumcheck {
        final_claim: claim,
        point: challenges.iter().copied().map(Into::into).collect(),
        challenges,
    })
}

pub fn sumcheck_initial_claim<const D: usize>(rounds: &SumCheckRounds<D>) -> Fr {
    rounds
        .round_polys
        .first()
        .map(|round| round.eval(Fr::zero()) + round.eval(Fr::one()))
        .unwrap_or(rounds.final_claim)
}
