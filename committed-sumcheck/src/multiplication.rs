//! Sigma protocol for committed multiplication under scalar Pedersen commitments.
//!
//! Proves knowledge of openings for
//!
//! ```text
//! C_a = aG + rho_a H
//! C_b = bG + rho_b H
//! C_c = cG + rho_c H
//! c = a b
//! ```
//!
//! Protocol:
//!
//! ```text
//! Public statement:
//!   G, H, C_a, C_b, C_c
//!
//! Prover witness:
//!   a, b, c, rho_a, rho_b, rho_c
//!   where C_a = aG + rho_a H
//!         C_b = bG + rho_b H
//!         C_c = cG + rho_c H
//!         c = a b
//!
//! Prover samples:
//!   x, rho_x, y, rho_y, tau
//!
//! Prover sends:
//!   T_a   = xG + rho_x H
//!   T_b   = yG + rho_y H
//!   T_mul = xC_b + tau H
//!
//! Verifier challenge:
//!   e <- transcript(G, H, C_a, C_b, C_c, T_a, T_b, T_mul)
//!
//! Prover responds:
//!   z_a       = x + a e
//!   z_rho_a   = rho_x + rho_a e
//!   z_b       = y + b e
//!   z_rho_b   = rho_y + rho_b e
//!   z_rho_mul = tau + (rho_c - a rho_b) e
//!
//! Verifier checks:
//!   z_a G + z_rho_a H       = T_a   + e C_a
//!   z_b G + z_rho_b H       = T_b   + e C_b
//!   z_a C_b + z_rho_mul H   = T_mul + e C_c
//!
//! The final check works because:
//!   z_a C_b + z_rho_mul H
//!     = (x + ae)(bG + rho_b H) + (tau + (rho_c - a rho_b)e)H
//!     = xC_b + tau H + e(abG + rho_c H)
//!     = T_mul + e C_c
//! ```

use ark_bn254::{Fr, G1Projective};
use ark_std::UniformRand;
use joltworks::transcripts::Transcript;
use rand_core::CryptoRngCore;

use crate::pedersen::{commit, Commitment, Opening, PedersenParams};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiplicationProof {
    pub t_a: G1Projective,
    pub t_b: G1Projective,
    pub t_mul: G1Projective,
    pub z_a: Fr,
    pub z_a_blinding: Fr,
    pub z_b: Fr,
    pub z_b_blinding: Fr,
    pub z_mul_blinding: Fr,
}

pub fn prove_multiplication<T, R>(
    params: &PedersenParams,
    a_commitment: &Commitment,
    b_commitment: &Commitment,
    c_commitment: &Commitment,
    a_opening: &Opening,
    b_opening: &Opening,
    c_opening: &Opening,
    transcript: &mut T,
    rng: &mut R,
) -> Option<MultiplicationProof>
where
    T: Transcript,
    R: CryptoRngCore,
{
    if a_opening.value * b_opening.value != c_opening.value {
        return None;
    }
    if commit(params, a_opening) != *a_commitment
        || commit(params, b_opening) != *b_commitment
        || commit(params, c_opening) != *c_commitment
    {
        return None;
    }

    let x = Fr::rand(rng);
    let x_blinding = Fr::rand(rng);
    let y = Fr::rand(rng);
    let y_blinding = Fr::rand(rng);
    let tau = Fr::rand(rng);

    let t_a = params.value_generator * x + params.blinding_generator * x_blinding;
    let t_b = params.value_generator * y + params.blinding_generator * y_blinding;
    let t_mul = b_commitment.0 * x + params.blinding_generator * tau;

    absorb_multiplication_statement(
        params,
        a_commitment,
        b_commitment,
        c_commitment,
        &t_a,
        &t_b,
        &t_mul,
        transcript,
    );
    let challenge = transcript.challenge_scalar::<Fr>();

    let mul_blinding = c_opening.blinding - a_opening.value * b_opening.blinding;
    let proof = MultiplicationProof {
        t_a,
        t_b,
        t_mul,
        z_a: x + a_opening.value * challenge,
        z_a_blinding: x_blinding + a_opening.blinding * challenge,
        z_b: y + b_opening.value * challenge,
        z_b_blinding: y_blinding + b_opening.blinding * challenge,
        z_mul_blinding: tau + mul_blinding * challenge,
    };
    absorb_multiplication_response(&proof, transcript);
    Some(proof)
}

pub fn verify_multiplication<T>(
    params: &PedersenParams,
    a_commitment: &Commitment,
    b_commitment: &Commitment,
    c_commitment: &Commitment,
    proof: &MultiplicationProof,
    transcript: &mut T,
) -> bool
where
    T: Transcript,
{
    absorb_multiplication_statement(
        params,
        a_commitment,
        b_commitment,
        c_commitment,
        &proof.t_a,
        &proof.t_b,
        &proof.t_mul,
        transcript,
    );
    let challenge = transcript.challenge_scalar::<Fr>();

    let check_a = params.value_generator * proof.z_a
        + params.blinding_generator * proof.z_a_blinding
        == proof.t_a + a_commitment.0 * challenge;
    let check_b = params.value_generator * proof.z_b
        + params.blinding_generator * proof.z_b_blinding
        == proof.t_b + b_commitment.0 * challenge;
    let check_mul = b_commitment.0 * proof.z_a + params.blinding_generator * proof.z_mul_blinding
        == proof.t_mul + c_commitment.0 * challenge;

    let accepted = check_a && check_b && check_mul;
    if accepted {
        absorb_multiplication_response(proof, transcript);
    }
    accepted
}

fn absorb_multiplication_statement<T: Transcript>(
    params: &PedersenParams,
    a_commitment: &Commitment,
    b_commitment: &Commitment,
    c_commitment: &Commitment,
    t_a: &G1Projective,
    t_b: &G1Projective,
    t_mul: &G1Projective,
    transcript: &mut T,
) {
    transcript.append_message(b"cs/multiplication/v1");
    transcript.append_point(&params.value_generator);
    transcript.append_point(&params.blinding_generator);
    transcript.append_point(&a_commitment.0);
    transcript.append_point(&b_commitment.0);
    transcript.append_point(&c_commitment.0);
    transcript.append_point(t_a);
    transcript.append_point(t_b);
    transcript.append_point(t_mul);
}

pub fn absorb_multiplication_response<T: Transcript>(
    proof: &MultiplicationProof,
    transcript: &mut T,
) {
    transcript.append_message(b"cs/multiplication-response/v1");
    transcript.append_scalar(&proof.z_a);
    transcript.append_scalar(&proof.z_a_blinding);
    transcript.append_scalar(&proof.z_b);
    transcript.append_scalar(&proof.z_b_blinding);
    transcript.append_scalar(&proof.z_mul_blinding);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pedersen::PedersenParams;
    use joltworks::transcripts::Blake2bTranscript;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn multiplication_proof_accepts_valid_product() {
        let mut rng = ChaCha20Rng::from_seed([18; 32]);
        let params = PedersenParams::random(&mut rng);
        let a_opening = Opening::random(Fr::from(7_u64), &mut rng);
        let b_opening = Opening::random(Fr::from(11_u64), &mut rng);
        let c_opening = Opening::random(a_opening.value * b_opening.value, &mut rng);
        let a_commitment = commit(&params, &a_opening);
        let b_commitment = commit(&params, &b_opening);
        let c_commitment = commit(&params, &c_opening);

        let mut prover_transcript = Blake2bTranscript::default();
        let proof = prove_multiplication(
            &params,
            &a_commitment,
            &b_commitment,
            &c_commitment,
            &a_opening,
            &b_opening,
            &c_opening,
            &mut prover_transcript,
            &mut rng,
        )
        .expect("valid multiplication witness");

        let mut verifier_transcript = Blake2bTranscript::default();
        assert!(verify_multiplication(
            &params,
            &a_commitment,
            &b_commitment,
            &c_commitment,
            &proof,
            &mut verifier_transcript,
        ));
    }

    #[test]
    fn multiplication_proof_rejects_wrong_product_commitment() {
        let mut rng = ChaCha20Rng::from_seed([19; 32]);
        let params = PedersenParams::random(&mut rng);
        let a_opening = Opening::random(Fr::from(7_u64), &mut rng);
        let b_opening = Opening::random(Fr::from(11_u64), &mut rng);
        let c_opening = Opening::random(a_opening.value * b_opening.value, &mut rng);
        let bad_c_opening = Opening::random(c_opening.value + Fr::from(1_u64), &mut rng);
        let a_commitment = commit(&params, &a_opening);
        let b_commitment = commit(&params, &b_opening);
        let c_commitment = commit(&params, &c_opening);
        let bad_c_commitment = commit(&params, &bad_c_opening);

        let mut prover_transcript = Blake2bTranscript::default();
        let proof = prove_multiplication(
            &params,
            &a_commitment,
            &b_commitment,
            &c_commitment,
            &a_opening,
            &b_opening,
            &c_opening,
            &mut prover_transcript,
            &mut rng,
        )
        .expect("valid multiplication witness");

        let mut verifier_transcript = Blake2bTranscript::default();
        assert!(!verify_multiplication(
            &params,
            &a_commitment,
            &b_commitment,
            &bad_c_commitment,
            &proof,
            &mut verifier_transcript,
        ));
    }

    #[test]
    fn multiplication_prover_rejects_invalid_witness() {
        let mut rng = ChaCha20Rng::from_seed([20; 32]);
        let params = PedersenParams::random(&mut rng);
        let a_opening = Opening::random(Fr::from(7_u64), &mut rng);
        let b_opening = Opening::random(Fr::from(11_u64), &mut rng);
        let c_opening = Opening::random(Fr::from(12_u64), &mut rng);
        let a_commitment = commit(&params, &a_opening);
        let b_commitment = commit(&params, &b_opening);
        let c_commitment = commit(&params, &c_opening);

        let mut transcript = Blake2bTranscript::default();
        assert!(prove_multiplication(
            &params,
            &a_commitment,
            &b_commitment,
            &c_commitment,
            &a_opening,
            &b_opening,
            &c_opening,
            &mut transcript,
            &mut rng,
        )
        .is_none());
    }
}
