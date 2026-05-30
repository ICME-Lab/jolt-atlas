//! Pedersen commitments for round-polynomial coefficients and adjacent-round
//! consistency proofs.

use ark_bn254::{Fr, G1Projective};
use ark_std::UniformRand;
use joltworks::transcripts::Transcript;
use rand_core::CryptoRngCore;

use crate::{
    pedersen::{commit, Commitment, Opening, PedersenParams},
    round::RoundPoly,
    schnorr::EqualityProof,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedRoundPoly {
    pub commitments: Vec<Commitment>,
    pub openings: Vec<Opening>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundConsistencyProof {
    pub equality: EqualityProof,
}

pub fn commit_round_poly<R>(
    params: &PedersenParams,
    poly: &RoundPoly<Fr>,
    rng: &mut R,
) -> CommittedRoundPoly
where
    R: CryptoRngCore,
{
    let openings = poly
        .coeffs
        .iter()
        .map(|&value| Opening {
            value,
            blinding: Fr::rand(rng),
        })
        .collect::<Vec<_>>();
    let commitments = openings
        .iter()
        .map(|opening| commit(params, opening))
        .collect::<Vec<_>>();
    CommittedRoundPoly {
        commitments,
        openings,
    }
}

pub fn scalar_round_poly(commitment: Commitment, opening: Opening) -> CommittedRoundPoly {
    CommittedRoundPoly {
        commitments: vec![commitment],
        openings: vec![opening],
    }
}

pub fn evaluate_round_opening(round: &CommittedRoundPoly, r: Fr) -> Opening {
    let mut value = Fr::from(0_u64);
    let mut blinding = Fr::from(0_u64);
    for (opening, r_power) in round.openings.iter().zip(powers(r, round.openings.len())) {
        value += opening.value * r_power;
        blinding += opening.blinding * r_power;
    }
    Opening { value, blinding }
}

pub fn evaluate_round_commitments(commitments: &[Commitment], r: Fr) -> Commitment {
    Commitment(
        commitments
            .iter()
            .zip(powers(r, commitments.len()))
            .map(|(commitment, r_power)| commitment.0 * r_power)
            .sum::<G1Projective>(),
    )
}

pub fn absorb_round_poly_commitments<T>(
    params: &PedersenParams,
    commitments: &[Commitment],
    transcript: &mut T,
) where
    T: Transcript,
{
    transcript.append_message(b"cs/round-poly-commitments/v1");
    transcript.append_point(&params.value_generator);
    transcript.append_point(&params.blinding_generator);
    transcript.append_u64(commitments.len() as u64);
    transcript.append_points(&commitment_points(commitments));
}

pub fn challenge_round_poly<T>(
    params: &PedersenParams,
    commitments: &[Commitment],
    transcript: &mut T,
) -> Fr
where
    T: Transcript,
{
    absorb_round_poly_commitments(params, commitments, transcript);
    transcript.challenge_scalar::<Fr>()
}

pub fn challenge_round_poly_optimized<T>(
    params: &PedersenParams,
    commitments: &[Commitment],
    transcript: &mut T,
) -> <Fr as joltworks::field::JoltField>::Challenge
where
    T: Transcript,
{
    absorb_round_poly_commitments(params, commitments, transcript);
    transcript.challenge_scalar_optimized::<Fr>()
}

pub fn prove_round_consistency<T, R>(
    params: &PedersenParams,
    left: &CommittedRoundPoly,
    right: &CommittedRoundPoly,
    r: Fr,
    transcript: &mut T,
    rng: &mut R,
) -> Option<RoundConsistencyProof>
where
    T: Transcript,
    R: CryptoRngCore,
{
    if left.commitments.len() != left.openings.len()
        || right.commitments.len() != right.openings.len()
        || left.openings.is_empty()
        || right.openings.is_empty()
    {
        return None;
    }

    let nonce_blinding = Fr::rand(rng);
    let nonce_commitment = params.blinding_generator * nonce_blinding;
    absorb_round_consistency_statement(
        params,
        &left.commitments,
        &right.commitments,
        r,
        &nonce_commitment,
        transcript,
    );
    let challenge = transcript.challenge_scalar::<Fr>();

    // left_blinding = Σ_j rho_{i,j} r^j, the blinding of g_i(r)'s committed value.
    let left_blinding = left
        .openings
        .iter()
        .zip(powers(r, left.openings.len()))
        .map(|(opening, r_power)| opening.blinding * r_power)
        .sum::<Fr>();

    // right_blinding = rho_{i+1,0} + Σ_j rho_{i+1,j}, the blinding of
    // g_{i+1}(0) + g_{i+1}(1).
    let right_blinding = right
        .openings
        .iter()
        .map(|opening| opening.blinding)
        .sum::<Fr>()
        + right.openings[0].blinding;
    let delta_blinding = left_blinding - right_blinding;

    let proof = RoundConsistencyProof {
        equality: EqualityProof {
            nonce_commitment,
            blinding_difference_response: nonce_blinding + challenge * delta_blinding,
        },
    };
    absorb_round_consistency_response(&proof, transcript);
    Some(proof)
}

pub fn verify_round_consistency<T>(
    params: &PedersenParams,
    left_commitments: &[Commitment],
    right_commitments: &[Commitment],
    r: Fr,
    proof: &RoundConsistencyProof,
    transcript: &mut T,
) -> bool
where
    T: Transcript,
{
    if left_commitments.is_empty() || right_commitments.is_empty() {
        return false;
    }

    absorb_round_consistency_statement(
        params,
        left_commitments,
        right_commitments,
        r,
        &proof.equality.nonce_commitment,
        transcript,
    );
    let challenge = transcript.challenge_scalar::<Fr>();

    // C_l = Σ_j C_{i,j} r^j commits to g_i(r).
    let left_commitment = evaluate_round_commitments(left_commitments, r).0;

    // C_r = C_{i+1,0} + Σ_j C_{i+1,j} commits to g_{i+1}(0) + g_{i+1}(1).
    let right_commitment = right_commitments
        .iter()
        .map(|commitment| commitment.0)
        .sum::<G1Projective>()
        + right_commitments[0].0;
    let delta_commitment = left_commitment - right_commitment;

    let accepted = params.blinding_generator * proof.equality.blinding_difference_response
        == proof.equality.nonce_commitment + delta_commitment * challenge;
    if accepted {
        absorb_round_consistency_response(proof, transcript);
    }
    accepted
}

pub fn absorb_round_consistency_response<T: Transcript>(
    proof: &RoundConsistencyProof,
    transcript: &mut T,
) {
    transcript.append_message(b"cs/round-consistency-response/v1");
    transcript.append_scalar(&proof.equality.blinding_difference_response);
}

fn absorb_round_consistency_statement<T: Transcript>(
    params: &PedersenParams,
    left_commitments: &[Commitment],
    right_commitments: &[Commitment],
    r: Fr,
    nonce_commitment: &G1Projective,
    transcript: &mut T,
) {
    transcript.append_message(b"cs/round-consistency/v1");
    transcript.append_point(&params.value_generator);
    transcript.append_point(&params.blinding_generator);
    transcript.append_u64(left_commitments.len() as u64);
    transcript.append_u64(right_commitments.len() as u64);
    transcript.append_points(&commitment_points(left_commitments));
    transcript.append_points(&commitment_points(right_commitments));
    transcript.append_scalar(&r);
    transcript.append_point(nonce_commitment);
}

fn commitment_points(commitments: &[Commitment]) -> Vec<G1Projective> {
    commitments.iter().map(|commitment| commitment.0).collect()
}

fn powers(r: Fr, len: usize) -> Vec<Fr> {
    let mut powers = Vec::with_capacity(len);
    let mut power = Fr::from(1_u64);
    for _ in 0..len {
        powers.push(power);
        power *= r;
    }
    powers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pedersen::PedersenParams;
    use joltworks::transcripts::Blake2bTranscript;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn round_consistency_proof_accepts_consistent_coefficients() {
        let mut rng = ChaCha20Rng::from_seed([12; 32]);
        let params = PedersenParams::random(&mut rng);
        let r = Fr::from(5_u64);
        let left = RoundPoly {
            coeffs: vec![Fr::from(3_u64), Fr::from(4_u64), Fr::from(2_u64)],
        };
        let left_at_r = left.evaluate(r);
        let right = RoundPoly {
            coeffs: vec![Fr::from(7_u64), left_at_r - Fr::from(14_u64)],
        };
        assert_eq!(
            left_at_r,
            right.evaluate(Fr::from(0_u64)) + right.evaluate(Fr::from(1_u64))
        );

        let left = commit_round_poly(&params, &left, &mut rng);
        let right = commit_round_poly(&params, &right, &mut rng);

        let mut prover_transcript = Blake2bTranscript::default();
        let proof =
            prove_round_consistency(&params, &left, &right, r, &mut prover_transcript, &mut rng)
                .expect("consistent round link");

        let mut verifier_transcript = Blake2bTranscript::default();
        assert!(verify_round_consistency(
            &params,
            &left.commitments,
            &right.commitments,
            r,
            &proof,
            &mut verifier_transcript,
        ));
    }

    #[test]
    fn round_challenge_absorbs_coefficient_commitments() {
        let mut rng = ChaCha20Rng::from_seed([14; 32]);
        let params = PedersenParams::random(&mut rng);
        let poly = RoundPoly {
            coeffs: vec![Fr::from(3_u64), Fr::from(4_u64), Fr::from(2_u64)],
        };
        let committed = commit_round_poly(&params, &poly, &mut rng);

        let mut prover_transcript = Blake2bTranscript::default();
        let prover_challenge =
            challenge_round_poly(&params, &committed.commitments, &mut prover_transcript);

        let mut verifier_transcript = Blake2bTranscript::default();
        let verifier_challenge =
            challenge_round_poly(&params, &committed.commitments, &mut verifier_transcript);

        assert_eq!(prover_challenge, verifier_challenge);

        let mut changed_commitments = committed.commitments.clone();
        changed_commitments[0].0 += params.blinding_generator;
        let mut changed_transcript = Blake2bTranscript::default();
        let changed_challenge =
            challenge_round_poly(&params, &changed_commitments, &mut changed_transcript);

        assert_ne!(prover_challenge, changed_challenge);
    }

    #[test]
    fn round_consistency_proof_rejects_wrong_next_commitments() {
        let mut rng = ChaCha20Rng::from_seed([13; 32]);
        let params = PedersenParams::random(&mut rng);
        let r = Fr::from(5_u64);
        let left = RoundPoly {
            coeffs: vec![Fr::from(3_u64), Fr::from(4_u64), Fr::from(2_u64)],
        };
        let left_at_r = left.evaluate(r);
        let right = RoundPoly {
            coeffs: vec![Fr::from(7_u64), left_at_r - Fr::from(14_u64)],
        };
        let bad_right = RoundPoly {
            coeffs: vec![Fr::from(8_u64), left_at_r - Fr::from(14_u64)],
        };

        let left = commit_round_poly(&params, &left, &mut rng);
        let right = commit_round_poly(&params, &right, &mut rng);
        let bad_right = commit_round_poly(&params, &bad_right, &mut rng);

        let mut prover_transcript = Blake2bTranscript::default();
        let proof =
            prove_round_consistency(&params, &left, &right, r, &mut prover_transcript, &mut rng)
                .expect("consistent round link");

        let mut verifier_transcript = Blake2bTranscript::default();
        assert!(!verify_round_consistency(
            &params,
            &left.commitments,
            &bad_right.commitments,
            r,
            &proof,
            &mut verifier_transcript,
        ));
    }
}
