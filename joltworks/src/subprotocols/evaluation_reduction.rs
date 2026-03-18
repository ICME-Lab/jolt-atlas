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
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{Opening, OpeningId, Openings, SumcheckId},
        unipoly::UniPoly,
    },
    transcripts::{AppendToTranscript, Transcript},
    utils::errors::ProofVerifyError,
};
use allocative::Allocative;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use atlas_onnx_tracer::tensor::Tensor;
use common::VirtualPolynomial;

/// Public instance for one 2-to-1 evaluation reduction round.
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize, Allocative)]
pub struct EvalReductionInstance<F: JoltField> {
    /// First opening point of the same polynomial.
    pub r1: Vec<F>,
    /// First opening claim v1 = P(r1).
    pub v1: F,
    /// Second opening point of the same polynomial.
    pub r2: Vec<F>,
    /// Second opening claim v2 = P(r2).
    pub v2: F,
}

impl<F: JoltField> EvalReductionInstance<F> {
    pub fn new(opening1: &Opening<F>, opening2: &Opening<F>) -> Result<Self, ProofVerifyError> {
        let r1: Vec<F> = opening1.0.r.iter().map(|&c| c.into()).collect();
        let r2: Vec<F> = opening2.0.r.iter().map(|&c| c.into()).collect();
        if r1.len() != r2.len() {
            return Err(ProofVerifyError::InvalidInputLength(r1.len(), r2.len()));
        }

        Ok(Self {
            r1,
            r2,
            v1: opening1.1,
            v2: opening2.1,
        })
    }
}

/// Witness for one 2-to-1 evaluation reduction round.
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize, Allocative)]
pub struct EvalReductionWitness<F: JoltField> {
    pub mle: MultilinearPolynomial<F>,
}

impl<F: JoltField> EvalReductionWitness<F> {
    /// Build a witness directly from an already-materialized MLE.
    pub fn new(mle: MultilinearPolynomial<F>) -> Self {
        Self { mle }
    }

    /// Build a witness from an integer tensor by padding to next power of two,
    /// then interpreting it as the coefficient vector of an MLE.
    pub fn from_tensor(tensor: &Tensor<i32>) -> Self {
        Self {
            mle: MultilinearPolynomial::from(tensor.padded_next_power_of_two()),
        }
    }
}

/// Proof for one 2-to-1 evaluation reduction round.
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize, Allocative)]
pub struct EvalReductionProof<F: JoltField> {
    /// Restriction polynomial h(t) = P(l(t)).
    pub h: UniPoly<F>,
}

/// Reduced instance produced by the 2-to-1 evaluation reduction.
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize, Allocative)]
pub struct ReducedInstance<F: JoltField> {
    pub r: Vec<F>,
    pub claim: F,
}

impl<F: JoltField> EvalReductionInstance<F> {
    pub fn prove<T: Transcript>(
        &self,
        witness: &EvalReductionWitness<F>,
        transcript: &mut T,
    ) -> Result<(EvalReductionProof<F>, ReducedInstance<F>), ProofVerifyError> {
        if self.r1.len() != witness.mle.get_num_vars() {
            return Err(ProofVerifyError::InvalidInputLength(
                self.r1.len(),
                witness.mle.get_num_vars(),
            ));
        }

        let r2_sub_r1: Vec<F> = zip(&self.r1, &self.r2).map(|(&x, &y)| y - x).collect();
        let h = compute_h(&witness.mle, &self.r1, &r2_sub_r1);

        #[cfg(test)]
        {
            // Sanity check: h(0) should match opening1 claim, h(1) should match opening2 claim.
            assert_eq!(h.eval_at_zero(), self.v1);
            assert_eq!(h.eval_at_one(), self.v2);
        }

        h.append_to_transcript(transcript);
        let x_prime: F::Challenge = transcript.challenge_scalar_optimized::<F>();

        let proof = EvalReductionProof { h };
        let reduced_instance = ReducedInstance {
            r: compute_l(&self.r1, &r2_sub_r1, x_prime.into()),
            claim: proof.h.evaluate(&x_prime),
        };

        Ok((proof, reduced_instance))
    }

    pub fn verify<T: Transcript>(
        &self,
        proof: &EvalReductionProof<F>,
        transcript: &mut T,
    ) -> Result<ReducedInstance<F>, ProofVerifyError> {
        if proof.h.eval_at_zero() != self.v1 || proof.h.eval_at_one() != self.v2 {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "2-to-1 evaluation reduction boundary check failed".to_string(),
            ));
        }

        if self.r1.len() != self.r2.len() {
            return Err(ProofVerifyError::InvalidInputLength(
                self.r1.len(),
                self.r2.len(),
            ));
        }

        proof.h.append_to_transcript(transcript);
        let x_prime = transcript.challenge_scalar_optimized::<F>();

        let expected_reduced_value = proof.h.evaluate::<F>(&x_prime.into());
        let r2_sub_r1: Vec<F> = zip(&self.r1, &self.r2).map(|(&x, &y)| y - x).collect();
        let expected_r = compute_l::<F>(&self.r1, &r2_sub_r1, x_prime.into());

        Ok(ReducedInstance {
            r: expected_r,
            claim: expected_reduced_value,
        })
    }
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
    zip(r1, r2_sub_r1)
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

pub struct EvalReductionProtocol;

impl EvalReductionProtocol {
    pub fn prove<F: JoltField, T: Transcript>(
        openings: &Openings<F>,
        output_mle: MultilinearPolynomial<F>,
        idx: usize,
        transcript: &mut T,
    ) -> Result<(EvalReductionProof<F>, ReducedInstance<F>), ProofVerifyError> {
        let lo = OpeningId::Virtual(
            VirtualPolynomial::NodeOutput(idx),
            SumcheckId::NodeExecution(idx + 1),
        );
        let hi = OpeningId::Virtual(
            VirtualPolynomial::NodeOutput(idx),
            SumcheckId::NodeExecution(usize::MAX),
        );

        let entries: Vec<(OpeningId, Opening<F>)> = openings
            .range(lo..=hi)
            .map(|(k, v)| (*k, v.clone()))
            .collect();

        if entries.len() < 2 {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "should be at least 2 openings for eval reduction, found {}",
                entries.len()
            )));
        }

        // Current scope is strictly 2-to-1 reduction.
        if entries.len() != 2 {
            // TODO(#138): Extend to iterative n-to-1 reduction when fanout > 2.
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "more than 2 openings found for eval reduction; n-to-1 reduction not yet implemented, found {}",
                entries.len()
            )));
        }

        let opening1 = entries[0].clone().1;
        let opening2 = entries[1].clone().1;

        let witness = EvalReductionWitness::new(output_mle);

        let instance = EvalReductionInstance::new(&opening1, &opening2)
            .expect("evaluation reduction instance should be constructible for matching openings");

        instance.prove(&witness, transcript)
    }

    pub fn verify<F: JoltField, T: Transcript>(
        openings: &Openings<F>,
        proof: &EvalReductionProof<F>,
        idx: usize,
        transcript: &mut T,
    ) -> Result<ReducedInstance<F>, ProofVerifyError> {
        let lo = OpeningId::Virtual(
            VirtualPolynomial::NodeOutput(idx),
            SumcheckId::NodeExecution(idx + 1),
        );
        let hi = OpeningId::Virtual(
            VirtualPolynomial::NodeOutput(idx),
            SumcheckId::NodeExecution(usize::MAX),
        );

        let entries: Vec<(OpeningId, Opening<F>)> = openings
            .range(lo..=hi)
            .map(|(k, v)| (*k, v.clone()))
            .collect();

        if entries.len() < 2 {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "should be at least 2 openings for eval reduction, found {}",
                entries.len()
            )));
        }

        // Current scope is strictly 2-to-1 reduction.
        if entries.len() != 2 {
            // TODO(#138): Extend to iterative n-to-1 reduction when fanout > 2.
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "more than 2 openings found for eval reduction; n-to-1 reduction not yet implemented, found {}",
                entries.len()
            )));
        }

        let opening1 = entries[0].clone().1;
        let opening2 = entries[1].clone().1;

        let instance = EvalReductionInstance::new(&opening1, &opening2)
            .expect("evaluation reduction instance should be constructible for matching openings");

        instance.verify(proof, transcript)
    }
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

        let witness = EvalReductionWitness { mle };

        let mut prover_tr = Blake2bTranscript::new(b"eval-reduction-test");
        let instance = EvalReductionInstance::new(&opening1, &opening2)
            .expect("instance should be valid for matching witness/openings");
        let (proof, reduced_prover) = instance
            .prove(&witness, &mut prover_tr)
            .expect("prover should succeed");

        let mut verifier_tr = Blake2bTranscript::new(b"eval-reduction-test");
        let reduced_verifier = instance
            .verify(&proof, &mut verifier_tr)
            .expect("verifier should accept valid reduction proof");
        assert_eq!(reduced_verifier, reduced_prover);
    }

    #[test]
    fn eval_reduction_2to1_rejects_tampered_h() {
        let mle = MultilinearPolynomial::<Fr>::from(vec![f(1), f(2), f(3), f(4)]);
        let r1 = vec![ch(1), ch(2)];
        let r2 = vec![ch(3), ch(4)];

        let v1 = mle.evaluate(&r1);
        let v2 = mle.evaluate(&r2);

        let opening1: Opening<Fr> = (r1.clone().into(), v1);
        let opening2: Opening<Fr> = (r2.clone().into(), v2);

        let witness = EvalReductionWitness { mle };

        let mut prover_tr = Blake2bTranscript::new(b"eval-reduction-test");
        let instance = EvalReductionInstance::new(&opening1, &opening2).unwrap();
        let (mut proof, _reduced) = instance.prove(&witness, &mut prover_tr).unwrap();

        // Tamper h so boundary checks fail.
        proof.h.coeffs[0] += Fr::from(1u64);
        let mut verifier_tr = Blake2bTranscript::new(b"eval-reduction-test");
        let err = instance
            .verify(&proof, &mut verifier_tr)
            .expect_err("verifier should reject when h is tampered");
        assert!(matches!(err, ProofVerifyError::InvalidOpeningProof(_)));
    }

    #[test]
    fn eval_reduction_2to1_transcript_mismatch_changes_reduced_instance() {
        let mle = MultilinearPolynomial::<Fr>::from(vec![f(1), f(2), f(3), f(4)]);
        let r1 = vec![ch(10), ch(11)];
        let r2 = vec![ch(12), ch(13)];
        let v1 = mle.evaluate(&r1);
        let v2 = mle.evaluate(&r2);

        let opening1: Opening<Fr> = (r1.clone().into(), v1);
        let opening2: Opening<Fr> = (r2.clone().into(), v2);

        let witness = EvalReductionWitness { mle };

        let mut prover_tr = Blake2bTranscript::new(b"eval-reduction-test");
        let instance = EvalReductionInstance::new(&opening1, &opening2).unwrap();
        let (proof, reduced) = instance.prove(&witness, &mut prover_tr).unwrap();

        let mut verifier_tr = Blake2bTranscript::new(b"eval-reduction-test");
        // Force transcript divergence before verification starts.
        verifier_tr.append_message(b"desync");

        let reduced_verifier = instance
            .verify(&proof, &mut verifier_tr)
            .expect("reduction verify still succeeds but yields a different reduced claim");
        assert_ne!(reduced_verifier, reduced);
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
