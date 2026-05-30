//! A small committed GKR composition for three product gates.
//!
//! The circuit shape is fixed:
//!
//! ```text
//! E = A * B
//! F = C * D
//! G = E * F
//! ```
//!
//! For each product layer, the prover runs a committed sumcheck for
//! `sum_x eq_R(x) lhs(x) rhs(x) = out(R)`. The incoming scalar claim is treated
//! as a degree-0 round polynomial, so the first-round link and all adjacent
//! round links use the same round-consistency proof.

use ark_bn254::Fr;
use ark_std::UniformRand;
use joltworks::{field::JoltField, transcripts::Transcript};
use rand_core::CryptoRngCore;

use crate::{
    committed_round::scalar_round_poly,
    gkr_layer::{prove_product_layer, verify_product_layer, ProductLayerProof},
    gkr_util::{absorb_gkr_statement, evaluate_mle, hadamard_values, validate_tables},
    pedersen::{commit, Commitment, Opening, PedersenParams},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreeProductGkrProof {
    pub output_commitment: Commitment,
    pub top: ProductLayerProof,
    pub left: ProductLayerProof,
    pub right: ProductLayerProof,
}

pub fn prove_three_product_gkr<T, R>(
    params: &PedersenParams,
    a: &[Fr],
    b: &[Fr],
    c: &[Fr],
    d: &[Fr],
    output_point: &[<Fr as JoltField>::Challenge],
    transcript: &mut T,
    rng: &mut R,
) -> Option<ThreeProductGkrProof>
where
    T: Transcript,
    R: CryptoRngCore,
{
    validate_tables(output_point.len(), &[a, b, c, d])?;

    let e = hadamard_values(a, b);
    let f = hadamard_values(c, d);
    let g = hadamard_values(&e, &f);

    let output_opening = Opening {
        value: evaluate_mle(&g, output_point),
        blinding: Fr::rand(rng),
    };
    let output_commitment = commit(params, &output_opening);
    let output_round = scalar_round_poly(output_commitment, output_opening);
    absorb_gkr_statement(params, output_point, &output_commitment, transcript);

    // G = E * F
    let (top_proof, top_claims) = prove_product_layer(
        b"top",
        params,
        &e,
        &f,
        output_point,
        &output_round,
        Fr::from(0_u64),
        transcript,
        rng,
    )?;

    let e_f_point = top_claims.sumcheck_point;

    // E = A * B
    let e_claim = scalar_round_poly(
        top_claims.lhs_at_sumcheck_point.commitment,
        top_claims.lhs_at_sumcheck_point.opening,
    );
    let (left_proof, _left_claims) = prove_product_layer(
        b"left",
        params,
        a,
        b,
        &e_f_point,
        &e_claim,
        Fr::from(0_u64),
        transcript,
        rng,
    )?;

    // F = C * D
    let f_claim = scalar_round_poly(
        top_claims.rhs_at_sumcheck_point.commitment,
        top_claims.rhs_at_sumcheck_point.opening,
    );
    let (right_proof, _right_claims) = prove_product_layer(
        b"right",
        params,
        c,
        d,
        &e_f_point,
        &f_claim,
        Fr::from(0_u64),
        transcript,
        rng,
    )?;

    Some(ThreeProductGkrProof {
        output_commitment,
        top: top_proof,
        left: left_proof,
        right: right_proof,
    })
}

pub fn verify_three_product_gkr<T>(
    params: &PedersenParams,
    proof: &ThreeProductGkrProof,
    num_vars: usize,
    output_point: &[<Fr as JoltField>::Challenge],
    transcript: &mut T,
) -> bool
where
    T: Transcript,
{
    if output_point.len() != num_vars || num_vars == 0 {
        return false;
    }

    absorb_gkr_statement(params, output_point, &proof.output_commitment, transcript);

    // G = E * F
    let Some(top_claims) = verify_product_layer(
        b"top",
        params,
        &[proof.output_commitment],
        Fr::from(0_u64),
        output_point,
        &proof.top,
        num_vars,
        transcript,
    ) else {
        return false;
    };

    let e_f_point = top_claims.sumcheck_point;

    // E = A * B
    if verify_product_layer(
        b"left",
        params,
        &[top_claims.lhs_at_sumcheck_point_commitment],
        Fr::from(0_u64),
        &e_f_point,
        &proof.left,
        e_f_point.len(),
        transcript,
    )
    .is_none()
    {
        return false;
    }

    // F = C * D
    verify_product_layer(
        b"right",
        params,
        &[top_claims.rhs_at_sumcheck_point_commitment],
        Fr::from(0_u64),
        &e_f_point,
        &proof.right,
        e_f_point.len(),
        transcript,
    )
    .is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use joltworks::transcripts::Blake2bTranscript;
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

    #[test]
    fn proves_three_product_gkr() {
        let mut rng = ChaCha20Rng::from_seed([30; 32]);
        let params = PedersenParams::random(&mut rng);
        let num_vars = 3;
        let len = 1 << num_vars;
        let a = random_values(len, &mut rng);
        let b = random_values(len, &mut rng);
        let c = random_values(len, &mut rng);
        let d = random_values(len, &mut rng);
        let output_point = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect::<Vec<_>>();

        let mut prover_transcript = Blake2bTranscript::default();
        let proof = prove_three_product_gkr(
            &params,
            &a,
            &b,
            &c,
            &d,
            &output_point,
            &mut prover_transcript,
            &mut rng,
        )
        .expect("valid GKR witness");

        let mut verifier_transcript = Blake2bTranscript::default();
        assert!(verify_three_product_gkr(
            &params,
            &proof,
            num_vars,
            &output_point,
            &mut verifier_transcript,
        ));
    }

    #[test]
    fn rejects_bad_top_multiplication_proof() {
        let mut rng = ChaCha20Rng::from_seed([31; 32]);
        let params = PedersenParams::random(&mut rng);
        let num_vars = 2;
        let len = 1 << num_vars;
        let a = random_values(len, &mut rng);
        let b = random_values(len, &mut rng);
        let c = random_values(len, &mut rng);
        let d = random_values(len, &mut rng);
        let output_point = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect::<Vec<_>>();

        let mut prover_transcript = Blake2bTranscript::default();
        let mut proof = prove_three_product_gkr(
            &params,
            &a,
            &b,
            &c,
            &d,
            &output_point,
            &mut prover_transcript,
            &mut rng,
        )
        .expect("valid GKR witness");
        proof.top.multiplication_proof.z_a += Fr::from(1_u64);

        let mut verifier_transcript = Blake2bTranscript::default();
        assert!(!verify_three_product_gkr(
            &params,
            &proof,
            num_vars,
            &output_point,
            &mut verifier_transcript,
        ));
    }

    fn random_values<R: CryptoRngCore>(len: usize, rng: &mut R) -> Vec<Fr> {
        (0..len).map(|_| Fr::rand(rng)).collect()
    }
}
