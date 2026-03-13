//! Two-to-one evaluation reduction without sumcheck.
//!
//! This module implements the PAZK-style line-restriction reduction for a
//! *single polynomial* with two opening claims:
//! - `P(r1) = v1`
//! - `P(r2) = v2`
//!
//! The prover sends the univariate restriction `h(t) = P(l(t))`, where
//! `l(0) = r1` and `l(1) = r2`. The verifier checks `h(0) = v1` and
//! `h(1) = v2`, derives a Fiat-Shamir challenge `x'`, and reduces to one claim:
//! `P(r') = v'` with `r' = l(x')` and `v' = h(x')`.

use core::panic;
use std::iter::zip;

use crate::{
    field::JoltField,
    poly::{
        multilinear_polynomial::MultilinearPolynomial, opening_proof::Opening, unipoly::UniPoly,
    },
    transcripts::{AppendToTranscript, Transcript},
    utils::errors::ProofVerifyError,
};
use allocative::Allocative;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// Reduced claim produced by the 2-to-1 evaluation reduction.
/// TODO: Should be used to populate a new opening for given polynomial once eval reduction occured, so that prover proves P(r') = v'
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize, Allocative)]
pub struct ReducedEvaluationClaim<F: JoltField> {
    /// Reduced opening point in the base field.
    pub r_prime: Vec<F>,
    /// Reduced evaluation value.
    pub v_prime: F,
}

/// Proof object for a single 2-to-1 evaluation reduction round.
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize, Allocative)]
pub struct EvaluationReduction2To1Proof<F: JoltField> {
    /// Restriction polynomial h(t) = P(l(t)).
    pub h: UniPoly<F>,
}

/// Prover-side 2-to-1 evaluation reduction.
///
/// Returns the reduction proof and the reduced claim `(r', v')`.
pub fn prove_two_to_one<F: JoltField, T: Transcript>(
    transcript: &mut T,
    mle: MultilinearPolynomial<F>,
    opening1: &Opening<F>,
    opening2: &Opening<F>,
) -> Result<(EvaluationReduction2To1Proof<F>, ReducedEvaluationClaim<F>), ProofVerifyError> {
    let r1: Vec<F> = opening1.0.r.iter().map(|&c| c.into()).collect();
    let r2: Vec<F> = opening2.0.r.iter().map(|&c| c.into()).collect();
    if r1.len() != r2.len() || r1.len() != mle.get_num_vars() {
        return Err(ProofVerifyError::InvalidInputLength(r1.len(), r2.len()));
    }

    let r2_sub_r1: Vec<F> = zip(&r1, &r2).map(|(&x, &y)| y - x).collect();

    let h = compute_h(&mle, &r1, &r2_sub_r1);

    #[cfg(test)]
    {
        // Sanity check: h(0) should match opening1 claim, h(1) should match opening2 claim.
        assert_eq!(h.eval_at_zero(), opening1.1);
        assert_eq!(h.eval_at_one(), opening2.1);
    }

    h.append_to_transcript(transcript);
    let x_prime: F::Challenge = transcript.challenge_scalar_optimized::<F>();

    let reduced_point = compute_l(&r1, &r2_sub_r1, x_prime.into());
    let reduced_value = h.evaluate(&x_prime);

    let proof = EvaluationReduction2To1Proof { h };
    let reduced_claim = ReducedEvaluationClaim {
        r_prime: reduced_point,
        v_prime: reduced_value,
    };

    Ok((proof, reduced_claim))
}

/// Verifier-side 2-to-1 evaluation reduction.
///
/// Validates boundary checks and transcript-consistency, then outputs reduced
/// claim `(r', v')`.
pub fn verify_two_to_one<F: JoltField, T: Transcript>(
    transcript: &mut T,
    opening1: &Opening<F>,
    opening2: &Opening<F>,
    proof: &EvaluationReduction2To1Proof<F>,
    reduced_opening: F,
) -> Result<ReducedEvaluationClaim<F>, ProofVerifyError> {
    let r1: Vec<F> = opening1.0.r.iter().map(|&c| c.into()).collect();
    let r2: Vec<F> = opening2.0.r.iter().map(|&c| c.into()).collect();
    if r1.len() != r2.len() {
        return Err(ProofVerifyError::InvalidInputLength(r1.len(), r2.len()));
    }

    if proof.h.eval_at_zero() != opening1.1 || proof.h.eval_at_one() != opening2.1 {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "2-to-1 evaluation reduction boundary check failed".to_string(),
        ));
    }

    proof.h.append_to_transcript(transcript);
    let x_prime = transcript.challenge_scalar_optimized::<F>();

    let expected_reduced_value = proof.h.evaluate::<F>(&x_prime.into());
    if expected_reduced_value != reduced_opening {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "2-to-1 evaluation reduction reduced-value mismatch".to_string(),
        ));
    }

    let r2_sub_r1: Vec<F> = zip(&r1, &r2).map(|(&x, &y)| y - x).collect();
    let reduced_point = compute_l::<F>(&r1, &r2_sub_r1, x_prime.into());
    Ok(ReducedEvaluationClaim {
        r_prime: reduced_point,
        v_prime: reduced_opening,
    })
}

fn compute_l<F: JoltField>(r1: &[F], r2_sub_r1: &[F], x: F) -> Vec<F> {
    if r1.len() != r2_sub_r1.len() {
        panic!(
            "compute_l input length mismatch: left: {}, right: {}",
            r1.len(),
            r2_sub_r1.len()
        );
    }

    // l(x) = r1 + x * (r2 - r1)
    r1.iter()
        .zip(r2_sub_r1.iter())
        .map(|(r1_i, delta_i)| *r1_i + x * *delta_i)
        .collect()
}

fn compute_h<F: JoltField>(
    mle: &MultilinearPolynomial<F>,
    r1: &[F],
    r2_sub_r1: &[F],
) -> UniPoly<F> {
    let num_vars = r1.len();

    let mut mle_polys: Vec<UniPoly<F>> = mle
        .coeffs()
        .iter()
        .map(|&coeff| {
            UniPoly::from_coeff(vec![coeff]) // Start with constant polynomial for each coeff
        })
        .collect();

    for (i, (r1_i, delta_i)) in zip(r1, r2_sub_r1).enumerate() {
        let var_weight = num_vars - i - 1;
        let half_len = 1 << var_weight;
        let var_poly = UniPoly::from_coeff(vec![*r1_i, *delta_i]); // r1_i + t * delta_i
        for j in 0..half_len {
            let left = std::mem::take(&mut mle_polys[j]);
            let right = std::mem::take(&mut mle_polys[j + half_len]);
            mle_polys[j] = &left + &var_poly * (&right - &left);
        }
    }

    std::mem::take(&mut mle_polys[0])
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use super::*;
    use crate::{
        field::JoltField, poly::multilinear_polynomial::PolynomialEvaluation,
        transcripts::Blake2bTranscript,
    };
    use ark_bn254::Fr;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rand_distr::Uniform;

    fn f(x: u128) -> Fr {
        Fr::from(x)
    }

    fn ch(x: u128) -> <Fr as JoltField>::Challenge {
        <Fr as JoltField>::Challenge::from(x)
    }

    #[test]
    fn eval_reduction_2to1_happy_path() {
        let mle =
            MultilinearPolynomial::<Fr>::from(vec![f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8)]);
        let r1 = vec![ch(1), ch(2), ch(3)];
        let r2 = vec![ch(4), ch(6), ch(8)];

        let v1 = mle.evaluate(&r1);
        let v2 = mle.evaluate(&r2);

        let opening1: Opening<Fr> = (r1.clone().into(), v1);
        let opening2: Opening<Fr> = (r2.clone().into(), v2);

        let mut prover_tr = Blake2bTranscript::new(b"eval-reduction-test");
        let (proof, reduced_prover) =
            prove_two_to_one::<Fr, _>(&mut prover_tr, mle, &opening1, &opening2)
                .expect("prover should succeed");

        let mut verifier_tr = Blake2bTranscript::new(b"eval-reduction-test");
        let reduced_verifier = verify_two_to_one::<Fr, _>(
            &mut verifier_tr,
            &opening1,
            &opening2,
            &proof,
            reduced_prover.v_prime,
        )
        .expect("verifier should accept valid reduction proof");

        assert_eq!(reduced_prover, reduced_verifier);
    }

    #[test]
    fn eval_reduction_2to1_rejects_wrong_boundary() {
        let mle = MultilinearPolynomial::<Fr>::from(vec![f(1), f(2), f(3), f(4)]);
        let r1 = vec![ch(1), ch(2)];
        let r2 = vec![ch(3), ch(4)];

        let v1 = mle.evaluate(&r1);
        let v2 = mle.evaluate(&r2);

        let opening1: Opening<Fr> = (r1.clone().into(), v1);
        let opening2: Opening<Fr> = (r2.clone().into(), v2);

        let mut prover_tr = Blake2bTranscript::new(b"eval-reduction-test");
        let (proof, reduced) =
            prove_two_to_one::<Fr, _>(&mut prover_tr, mle, &opening1, &opening2).unwrap();

        // Create a fake opening with wrong boundary values
        let fake_opening1: Opening<Fr> = (r1.clone().into(), v1 + Fr::from(1u64));
        let mut verifier_tr = Blake2bTranscript::new(b"eval-reduction-test");
        let err = verify_two_to_one(
            &mut verifier_tr,
            &fake_opening1,
            &opening2,
            &proof,
            reduced.v_prime,
        )
        .expect_err("verifier should reject when boundary checks fail");
        assert!(matches!(err, ProofVerifyError::InvalidOpeningProof(_)));
    }

    #[test]
    fn eval_reduction_2to1_rejects_transcript_mismatch() {
        let mle = MultilinearPolynomial::<Fr>::from(vec![f(1), f(2), f(3), f(4)]);
        let r1 = vec![ch(10), ch(11)];
        let r2 = vec![ch(12), ch(13)];
        let v1 = mle.evaluate(&r1);
        let v2 = mle.evaluate(&r2);

        let opening1: Opening<Fr> = (r1.clone().into(), v1);
        let opening2: Opening<Fr> = (r2.clone().into(), v2);

        let mut prover_tr = Blake2bTranscript::new(b"eval-reduction-test");
        let (proof, reduced) =
            prove_two_to_one::<Fr, _>(&mut prover_tr, mle, &opening1, &opening2).unwrap();

        let mut verifier_tr = Blake2bTranscript::new(b"eval-reduction-test");
        // Force transcript divergence before verification starts.
        verifier_tr.append_message(b"desync");

        let err = verify_two_to_one::<Fr, _>(
            &mut verifier_tr,
            &opening1,
            &opening2,
            &proof,
            reduced.v_prime,
        )
        .expect_err("verifier should reject when transcript challenges diverge");
        assert!(matches!(err, ProofVerifyError::InvalidOpeningProof(_)));
    }

    #[test]
    fn eval_reduction_compute_l_matches_expected() {
        let r1 = vec![f(1), f(2), f(3)];
        let r2_sub_r1 = vec![f(4), f(5), f(6)];
        let x = f(2);

        let expected = vec![f(1) + f(2) * f(4), f(2) + f(2) * f(5), f(3) + f(2) * f(6)];

        let result = compute_l::<Fr>(&r1, &r2_sub_r1, x);
        assert_eq!(result, expected);
    }

    #[test]
    // Asserts that the evaluation of l at a random point x matches the expected evaluation of the line,
    // defined by l(0) = r1 and l(1) = r2.
    fn test_l_evaluation() {
        let mut rng = StdRng::seed_from_u64(0x888);

        for _ in 0..10 {
            let num_vars = rng.sample(Uniform::from(1..10));
            let r1 = (0..num_vars)
                .map(|_| Fr::random(&mut rng))
                .collect::<Vec<_>>();
            let r2 = (0..num_vars)
                .map(|_| Fr::random(&mut rng))
                .collect::<Vec<_>>();
            let r2_sub_r1: Vec<Fr> = r2
                .iter()
                .zip(r1.iter())
                .map(|(r2_i, r1_i)| *r2_i - *r1_i)
                .collect();
            let x = Fr::random(&mut rng);

            let l_eval = compute_l::<Fr>(&r1, &r2_sub_r1, x);
            let l_direct: Vec<Fr> = r1
                .iter()
                .zip(r2.iter())
                .map(|(r1_i, r2_i)| *r1_i + x * (*r2_i - *r1_i))
                .collect();

            assert_eq!(l_eval, l_direct);
        }
    }

    #[test]
    // Asserts the correct computation of restriction h of mle to the line l; h = mle ∘ l
    // In this test we assert that h(r1) = v1 and h(r2) = v2,
    // and that degree of h is at most the number of variables in the original MLE.
    fn test_h_computation() {
        let mut rng = StdRng::seed_from_u64(0x888);

        for _ in 0..10 {
            let num_vars = rng.sample(Uniform::from(1..10));
            let mle = MultilinearPolynomial::from(
                (0..1 << num_vars)
                    .map(|_| Fr::random(&mut rng))
                    .collect::<Vec<_>>(),
            );
            let r1 = (0..num_vars)
                .map(|_| Fr::random(&mut rng))
                .collect::<Vec<_>>();
            let r2 = (0..num_vars)
                .map(|_| Fr::random(&mut rng))
                .collect::<Vec<_>>();
            let r2_sub_r1 = zip(&r1, &r2).map(|(x, y)| y - x).collect::<Vec<_>>();
            let h = compute_h(&mle, &r1, &r2_sub_r1);

            assert!(h.degree() <= num_vars);
            assert_eq!(h.eval_at_zero(), mle.evaluate(&r1));
            assert_eq!(h.eval_at_one(), mle.evaluate(&r2));
        }
    }

    #[test]
    // Asserts that with a random generated challenge t_f,
    // the evaluation of h at t_f matches the evaluation of the original MLE at the corresponding l(t_f).
    fn h_restr_l_matches_expected() {
        let mut rng = StdRng::seed_from_u64(0x888);

        for _ in 0..10 {
            let num_vars = rng.sample(Uniform::from(1..10));
            let mle = MultilinearPolynomial::from(
                (0..1 << num_vars)
                    .map(|_| Fr::random(&mut rng))
                    .collect::<Vec<_>>(),
            );
            let r1 = (0..num_vars)
                .map(|_| Fr::random(&mut rng))
                .collect::<Vec<_>>();
            let r2 = (0..num_vars)
                .map(|_| Fr::random(&mut rng))
                .collect::<Vec<_>>();
            let r2_sub_r1 = zip(&r1, &r2).map(|(x, y)| y - x).collect::<Vec<_>>();
            let h = compute_h(&mle, &r1, &r2_sub_r1);

            // Verify random challenges
            for _ in 0..10 {
                let t_f: Fr = rng.gen();
                let l_eval = compute_l::<Fr>(&r1, &r2_sub_r1, t_f);
                let h_eval = h.evaluate(&t_f);
                let mle_eval = mle.evaluate(&l_eval);
                assert_eq!(h_eval, mle_eval);
            }
        }
    }
}
