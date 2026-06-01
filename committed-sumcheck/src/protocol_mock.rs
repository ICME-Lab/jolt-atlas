//! Protocol-shaped committed sumcheck mock.
//!
//! This module intentionally does not implement the cryptographic details. It
//! exists to pin down the control flow we want the real implementation to
//! follow.

use ark_bn254::Fr;
use ark_ff::{One, Zero};
use ark_std::UniformRand;
use joltworks::{field::JoltField, transcripts::Transcript};
use rand_core::CryptoRngCore;

use crate::{
    committed_round::absorb_round_poly_commitments,
    pedersen::{commit, Commitment, Opening, PedersenParams},
    round::RoundRelation,
    schnorr::{self, EqualityProof},
    sumcheck::SumCheck,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifierClaim {
    pub commitment: Commitment,
}

impl VerifierClaim {
    pub fn zero() -> Self {
        Self {
            commitment: Commitment(Zero::zero()),
        }
    }
}

impl From<ProverClaim> for VerifierClaim {
    fn from(claim: ProverClaim) -> Self {
        Self {
            commitment: claim.commitment,
        }
    }
}

impl std::ops::Add for VerifierClaim {
    type Output = VerifierClaim;

    fn add(self, rhs: VerifierClaim) -> Self::Output {
        VerifierClaim {
            commitment: Commitment(self.commitment.0 + rhs.commitment.0),
        }
    }
}

impl std::ops::Mul<Fr> for VerifierClaim {
    type Output = VerifierClaim;

    fn mul(self, rhs: Fr) -> Self::Output {
        VerifierClaim {
            commitment: Commitment(self.commitment.0 * rhs),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProverClaim {
    pub commitment: Commitment,
    pub opening: Opening,
}

impl ProverClaim {
    pub fn zero() -> Self {
        Self {
            commitment: Commitment(Zero::zero()),
            opening: Opening {
                value: Fr::zero(),
                blinding: Fr::zero(),
            },
        }
    }
}

impl std::ops::Add for ProverClaim {
    type Output = ProverClaim;

    fn add(self, rhs: ProverClaim) -> Self::Output {
        ProverClaim {
            commitment: Commitment(self.commitment.0 + rhs.commitment.0),
            opening: Opening {
                value: self.opening.value + rhs.opening.value,
                blinding: self.opening.blinding + rhs.opening.blinding,
            },
        }
    }
}

impl std::ops::Mul<Fr> for ProverClaim {
    type Output = ProverClaim;

    fn mul(self, rhs: Fr) -> Self::Output {
        ProverClaim {
            commitment: Commitment(self.commitment.0 * rhs),
            opening: Opening {
                value: self.opening.value * rhs,
                blinding: self.opening.blinding * rhs,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProverRoundPoly<const D: usize> {
    pub coeffs: [ProverClaim; D],
}

impl<const D: usize> ProverRoundPoly<D> {
    pub fn eval(&self, point: Fr) -> ProverClaim {
        self.coeffs
            .iter()
            .rev()
            .fold(ProverClaim::zero(), |acc, coeff| acc * point + *coeff)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierRoundPoly<const D: usize> {
    pub coeffs: [VerifierClaim; D],
}

impl<const D: usize> VerifierRoundPoly<D> {
    pub fn eval(&self, point: Fr) -> VerifierClaim {
        self.coeffs
            .iter()
            .rev()
            .fold(VerifierClaim::zero(), |acc, coeff| acc * point + *coeff)
    }
}

impl<const D: usize> From<&ProverRoundPoly<D>> for VerifierRoundPoly<D> {
    fn from(round: &ProverRoundPoly<D>) -> Self {
        Self {
            coeffs: round.coeffs.map(VerifierClaim::from),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockRoundProof<const D: usize> {
    pub round: VerifierRoundPoly<D>,
    pub equality: EqualityProof,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockSumCheckProof<const D: usize> {
    pub rounds: Vec<MockRoundProof<D>>,
    pub final_claim: VerifierClaim,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumCheckRounds<const D: usize> {
    pub proof: MockSumCheckProof<D>,
    pub challenges: Vec<<Fr as JoltField>::Challenge>,
}

#[derive(Debug, Clone)]
pub struct SumCheckPolynomial<R, const LANES: usize, const D: usize>
where
    R: RoundRelation<Fr, LANES> + Sync,
{
    inner: SumCheck<Fr, R, LANES>,
    pub challenges: Vec<<Fr as JoltField>::Challenge>,
}

impl<R, const LANES: usize, const D: usize> SumCheckPolynomial<R, LANES, D>
where
    R: RoundRelation<Fr, LANES> + Sync,
{
    pub fn new(inner: SumCheck<Fr, R, LANES>) -> Self {
        Self {
            inner,
            challenges: Vec::new(),
        }
    }

    pub fn num_rounds(&self) -> usize {
        self.inner.remaining_rounds()
    }

    pub fn round_poly(&self, round: usize) -> Option<[Fr; D]> {
        let _ = round;
        if self.inner.is_complete() {
            return None;
        }
        self.inner.round_poly().coeffs.try_into().ok()
    }

    pub fn bind(&mut self, r: <Fr as JoltField>::Challenge) {
        self.inner.bind(r);
        self.challenges.push(r);
    }
}

pub fn commit_round_poly<R, const D: usize>(
    params: &PedersenParams,
    coeffs: [Fr; D],
    rng: &mut R,
) -> ProverRoundPoly<D>
where
    R: CryptoRngCore,
{
    ProverRoundPoly {
        coeffs: coeffs.map(|value| {
            let opening = Opening {
                value,
                blinding: Fr::rand(rng),
            };
            ProverClaim {
                commitment: commit(params, &opening),
                opening,
            }
        }),
    }
}

pub fn commit_claim(params: &PedersenParams, opening: Opening) -> ProverClaim {
    ProverClaim {
        commitment: commit(params, &opening),
        opening,
    }
}

pub fn append_claim<T>(transcript: &mut T, claim: VerifierClaim)
where
    T: Transcript,
{
    transcript.append_message(b"cs/claim/v1");
    transcript.append_point(&claim.commitment.0);
}

pub fn append_prover_round<T, const D: usize>(
    params: &PedersenParams,
    transcript: &mut T,
    round: &ProverRoundPoly<D>,
) where
    T: Transcript,
{
    let commitments = round
        .coeffs
        .iter()
        .map(|claim| claim.commitment)
        .collect::<Vec<_>>();
    absorb_round_poly_commitments(params, &commitments, transcript);
}

pub fn append_verifier_round<T, const D: usize>(
    params: &PedersenParams,
    transcript: &mut T,
    round: &VerifierRoundPoly<D>,
) where
    T: Transcript,
{
    let commitments = round
        .coeffs
        .iter()
        .map(|claim| claim.commitment)
        .collect::<Vec<_>>();
    absorb_round_poly_commitments(params, &commitments, transcript);
}

pub fn prove_equality<T, Rng>(
    params: &PedersenParams,
    left: ProverClaim,
    right: ProverClaim,
    transcript: &mut T,
    rng: &mut Rng,
) -> Option<EqualityProof>
where
    T: Transcript,
    Rng: CryptoRngCore,
{
    schnorr::prove_equality(
        params,
        &left.commitment,
        &right.commitment,
        &left.opening,
        &right.opening,
        transcript,
        rng,
    )
}

pub fn verify_equality<T>(
    params: &PedersenParams,
    left: VerifierClaim,
    right: VerifierClaim,
    proof: &EqualityProof,
    transcript: &mut T,
) -> Option<()>
where
    T: Transcript,
{
    schnorr::verify_equality(
        params,
        &left.commitment,
        &right.commitment,
        proof,
        transcript,
    )
    .then_some(())
}

pub fn prove_sumcheck_rounds<R, T, Rng, const LANES: usize, const D: usize>(
    g: &mut SumCheckPolynomial<R, LANES, D>,
    mut claim_i: ProverClaim,
    params: &PedersenParams,
    transcript: &mut T,
    rng: &mut Rng,
) -> Option<SumCheckRounds<D>>
where
    R: RoundRelation<Fr, LANES> + Sync,
    T: Transcript,
    Rng: CryptoRngCore,
{
    let mut round_proofs = Vec::with_capacity(g.num_rounds());
    let mut challenges = Vec::with_capacity(g.num_rounds());

    for i in 0..g.num_rounds() {
        append_claim(transcript, claim_i.into());

        let round_i = commit_round_poly(params, g.round_poly(i)?, rng);
        append_prover_round(params, transcript, &round_i);

        let equality_i = prove_equality(
            params,
            claim_i,
            round_i.eval(Fr::zero()) + round_i.eval(Fr::one()),
            transcript,
            rng,
        )?;

        let r_i = transcript.challenge_scalar_optimized::<Fr>();
        g.bind(r_i);
        claim_i = round_i.eval(r_i.into());
        challenges.push(r_i);

        round_proofs.push(MockRoundProof {
            round: VerifierRoundPoly::from(&round_i),
            equality: equality_i,
        });
    }

    Some(SumCheckRounds {
        proof: MockSumCheckProof {
            rounds: round_proofs,
            final_claim: claim_i.into(),
        },
        challenges,
    })
}

pub fn verify_sumcheck_rounds<T, const D: usize>(
    mut claim_i: VerifierClaim,
    proof: &MockSumCheckProof<D>,
    params: &PedersenParams,
    transcript: &mut T,
) -> Option<(VerifierClaim, Vec<<Fr as JoltField>::Challenge>)>
where
    T: Transcript,
{
    let mut challenges = Vec::with_capacity(proof.rounds.len());

    for round_proof in &proof.rounds {
        append_claim(transcript, claim_i);

        let round_i = &round_proof.round;
        append_verifier_round(params, transcript, round_i);

        verify_equality(
            params,
            claim_i,
            round_i.eval(Fr::zero()) + round_i.eval(Fr::one()),
            &round_proof.equality,
            transcript,
        )?;

        let r_i = transcript.challenge_scalar_optimized::<Fr>();
        claim_i = round_i.eval(r_i.into());
        challenges.push(r_i);
    }

    Some((claim_i, challenges))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ops::hadamard::Hadamard, round::DenseMleTable};
    use joltworks::transcripts::Blake2bTranscript;
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

    #[test]
    fn proves_and_verifies_sumcheck_rounds() {
        let mut rng = ChaCha20Rng::from_seed([31; 32]);
        let params = PedersenParams::random(&mut rng);
        let eq_point = [
            <Fr as JoltField>::Challenge::from(2_u128),
            <Fr as JoltField>::Challenge::from(5_u128),
        ];
        let lhs = [1, 3, 2, 5].map(Fr::from);
        let rhs = [7, 11, 13, 17].map(Fr::from);

        let prover_relation = Hadamard::new::<Fr>(
            DenseMleTable::new(lhs.to_vec()),
            DenseMleTable::new(rhs.to_vec()),
        );
        let verifier_relation = Hadamard::new::<Fr>(
            DenseMleTable::new(lhs.to_vec()),
            DenseMleTable::new(rhs.to_vec()),
        );
        let prover_sumcheck = SumCheck::<Fr, _, 3>::new(&eq_point, prover_relation);
        let verifier_sumcheck = SumCheck::<Fr, _, 3>::new(&eq_point, verifier_relation);

        let initial_poly = prover_sumcheck.round_poly();
        let initial_opening = Opening {
            value: initial_poly.evaluate(Fr::zero()) + initial_poly.evaluate(Fr::one()),
            blinding: Fr::rand(&mut rng),
        };
        let initial_claim = commit_claim(&params, initial_opening);

        let mut prover_polynomial = SumCheckPolynomial::<_, 3, 4>::new(prover_sumcheck);
        let mut prover_transcript = Blake2bTranscript::default();
        let output = prove_sumcheck_rounds(
            &mut prover_polynomial,
            initial_claim,
            &params,
            &mut prover_transcript,
            &mut rng,
        )
        .expect("honest prover creates sumcheck rounds");

        let mut verifier_transcript = Blake2bTranscript::default();
        let (final_claim, verifier_challenges) = verify_sumcheck_rounds(
            VerifierClaim::from(initial_claim),
            &output.proof,
            &params,
            &mut verifier_transcript,
        )
        .expect("honest proof verifies");

        assert_eq!(output.challenges, verifier_challenges);
        assert_eq!(output.proof.final_claim, final_claim);

        let mut verifier_polynomial = SumCheckPolynomial::<_, 3, 4>::new(verifier_sumcheck);
        for challenge in verifier_challenges {
            verifier_polynomial.bind(challenge);
        }
        assert_eq!(prover_polynomial.challenges, verifier_polynomial.challenges);
    }
}
