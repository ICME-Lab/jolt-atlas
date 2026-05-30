//! Schnorr-style proofs for scalar Pedersen commitments.
//!
//! `OpeningProof` proves knowledge of the blinding for a public value:
//!
//! ```text
//! C - value * G = blinding * H
//! ```
//!
//! `EqualityProof` proves that two Pedersen commitments under the same `G,H`
//! hide the same value without revealing that value:
//!
//! ```text
//! C1 = value * G + r1 * H
//! C2 = value * G + r2 * H
//! C1 - C2 = (r1 - r2) * H
//! ```

use ark_bn254::{Fr, G1Projective};
use ark_std::UniformRand;
use joltworks::transcripts::Transcript;
use rand_core::CryptoRngCore;

use crate::pedersen::{commitment_without_value, Commitment, Opening, PedersenParams};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpeningProof {
    pub nonce_commitment: G1Projective,
    pub response: Fr,
}

pub fn prove_opening<T, R>(
    params: &PedersenParams,
    commitment: &Commitment,
    opening: &Opening,
    transcript: &mut T,
    rng: &mut R,
) -> OpeningProof
where
    T: Transcript,
    R: CryptoRngCore,
{
    let nonce = Fr::rand(rng);
    let nonce_commitment = params.blinding_generator * nonce;
    absorb_opening_statement(
        params,
        commitment,
        opening.value,
        &nonce_commitment,
        transcript,
    );
    let challenge = transcript.challenge_scalar::<Fr>();
    let response = nonce + challenge * opening.blinding;
    OpeningProof {
        nonce_commitment,
        response,
    }
}

pub fn verify_opening<T>(
    params: &PedersenParams,
    commitment: &Commitment,
    value: Fr,
    proof: &OpeningProof,
    transcript: &mut T,
) -> bool
where
    T: Transcript,
{
    absorb_opening_statement(
        params,
        commitment,
        value,
        &proof.nonce_commitment,
        transcript,
    );
    let challenge = transcript.challenge_scalar::<Fr>();
    let hidden_part = commitment_without_value(params, commitment, value);
    params.blinding_generator * proof.response == proof.nonce_commitment + hidden_part * challenge
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EqualityProof {
    pub nonce_commitment: G1Projective,
    pub blinding_difference_response: Fr,
}

pub fn prove_equality<T, R>(
    params: &PedersenParams,
    left_commitment: &Commitment,
    right_commitment: &Commitment,
    left_opening: &Opening,
    right_opening: &Opening,
    transcript: &mut T,
    rng: &mut R,
) -> Option<EqualityProof>
where
    T: Transcript,
    R: CryptoRngCore,
{
    if left_opening.value != right_opening.value {
        return None;
    }

    let blinding_difference = left_opening.blinding - right_opening.blinding;
    let nonce = Fr::rand(rng);
    let nonce_commitment = params.blinding_generator * nonce;

    absorb_equality_statement(
        params,
        left_commitment,
        right_commitment,
        &nonce_commitment,
        transcript,
    );
    let challenge = transcript.challenge_scalar::<Fr>();

    Some(EqualityProof {
        nonce_commitment,
        blinding_difference_response: nonce + challenge * blinding_difference,
    })
}

pub fn verify_equality<T>(
    params: &PedersenParams,
    left_commitment: &Commitment,
    right_commitment: &Commitment,
    proof: &EqualityProof,
    transcript: &mut T,
) -> bool
where
    T: Transcript,
{
    absorb_equality_statement(
        params,
        left_commitment,
        right_commitment,
        &proof.nonce_commitment,
        transcript,
    );
    let challenge = transcript.challenge_scalar::<Fr>();

    let difference = left_commitment.0 - right_commitment.0;
    params.blinding_generator * proof.blinding_difference_response
        == proof.nonce_commitment + difference * challenge
}

fn absorb_opening_statement<T: Transcript>(
    params: &PedersenParams,
    commitment: &Commitment,
    value: Fr,
    nonce_commitment: &G1Projective,
    transcript: &mut T,
) {
    transcript.append_message(b"cs/schnorr-open/v1");
    transcript.append_point(&params.value_generator);
    transcript.append_point(&params.blinding_generator);
    transcript.append_point(&commitment.0);
    transcript.append_scalar(&value);
    transcript.append_point(nonce_commitment);
}

fn absorb_equality_statement<T: Transcript>(
    params: &PedersenParams,
    left_commitment: &Commitment,
    right_commitment: &Commitment,
    nonce_commitment: &G1Projective,
    transcript: &mut T,
) {
    transcript.append_message(b"cs/schnorr-eq/v1");
    transcript.append_point(&params.value_generator);
    transcript.append_point(&params.blinding_generator);
    transcript.append_point(&left_commitment.0);
    transcript.append_point(&right_commitment.0);
    transcript.append_point(nonce_commitment);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pedersen::{commit, PedersenParams};
    use joltworks::transcripts::Blake2bTranscript;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn opening_proof_accepts_valid_opening() {
        let mut rng = ChaCha20Rng::from_seed([7; 32]);
        let params = PedersenParams::random(&mut rng);
        let opening = Opening::random(Fr::from(42_u64), &mut rng);
        let commitment = commit(&params, &opening);

        let mut prover_transcript = Blake2bTranscript::default();
        let proof = prove_opening(
            &params,
            &commitment,
            &opening,
            &mut prover_transcript,
            &mut rng,
        );

        let mut verifier_transcript = Blake2bTranscript::default();
        assert!(verify_opening(
            &params,
            &commitment,
            opening.value,
            &proof,
            &mut verifier_transcript,
        ));
    }

    #[test]
    fn opening_proof_rejects_wrong_value() {
        let mut rng = ChaCha20Rng::from_seed([8; 32]);
        let params = PedersenParams::random(&mut rng);
        let opening = Opening::random(Fr::from(42_u64), &mut rng);
        let commitment = commit(&params, &opening);
        let mut prover_transcript = Blake2bTranscript::default();
        let proof = prove_opening(
            &params,
            &commitment,
            &opening,
            &mut prover_transcript,
            &mut rng,
        );

        let mut verifier_transcript = Blake2bTranscript::default();
        assert!(!verify_opening(
            &params,
            &commitment,
            Fr::from(43_u64),
            &proof,
            &mut verifier_transcript,
        ));
    }

    #[test]
    fn equality_proof_accepts_same_value() {
        let mut rng = ChaCha20Rng::from_seed([9; 32]);
        let params = PedersenParams::random(&mut rng);
        let value = Fr::from(11_u64);
        let left_opening = Opening::random(value, &mut rng);
        let right_opening = Opening::random(value, &mut rng);
        let left_commitment = commit(&params, &left_opening);
        let right_commitment = commit(&params, &right_opening);

        let mut prover_transcript = Blake2bTranscript::default();
        let proof = prove_equality(
            &params,
            &left_commitment,
            &right_commitment,
            &left_opening,
            &right_opening,
            &mut prover_transcript,
            &mut rng,
        )
        .expect("same-value equality proof");

        let mut verifier_transcript = Blake2bTranscript::default();
        assert!(verify_equality(
            &params,
            &left_commitment,
            &right_commitment,
            &proof,
            &mut verifier_transcript,
        ));
    }

    #[test]
    fn equality_proof_rejects_wrong_commitment() {
        let mut rng = ChaCha20Rng::from_seed([10; 32]);
        let params = PedersenParams::random(&mut rng);
        let value = Fr::from(11_u64);
        let left_opening = Opening::random(value, &mut rng);
        let right_opening = Opening::random(value, &mut rng);
        let bad_right_opening = Opening::random(Fr::from(12_u64), &mut rng);
        let left_commitment = commit(&params, &left_opening);
        let right_commitment = commit(&params, &right_opening);
        let bad_right_commitment = commit(&params, &bad_right_opening);

        let mut prover_transcript = Blake2bTranscript::default();
        let proof = prove_equality(
            &params,
            &left_commitment,
            &right_commitment,
            &left_opening,
            &right_opening,
            &mut prover_transcript,
            &mut rng,
        )
        .expect("same-value equality proof");

        let mut verifier_transcript = Blake2bTranscript::default();
        assert!(!verify_equality(
            &params,
            &left_commitment,
            &bad_right_commitment,
            &proof,
            &mut verifier_transcript,
        ));
    }
}
