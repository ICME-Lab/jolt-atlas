//! Two-to-one evaluation reduction without sumcheck.
//!
//! This module implements the PAZK-style line-restriction reduction for a
//! *single polynomial* with N opening claims:
//! - `P(r1) = v1`
//! - `P(r2) = v2`
//! - ...
//! - `P(rN) = vN`
//!
//! The prover sends the univariate restriction `h(t) = P(l(t))`, where
//! for all i in 0..N, `l(i) = ri`. The verifier checks for all i in 0..N, `h(i) = vi`,
//! derives a Fiat-Shamir challenge `x'`, and reduces to one claim:
//! `P(r') = v'` with `r' = l(x')` and `v' = h(x')`.

use std::fmt::Debug;

use crate::{
    field::JoltField,
    poly::{
        multilinear_polynomial::MultilinearPolynomial, opening_proof::Opening, unipoly::UniPoly,
    },
    transcripts::{AppendToTranscript, Transcript},
    utils::errors::{ProofVerifyError, ProvingError},
};
use allocative::Allocative;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use atlas_onnx_tracer::tensor::Tensor;

/// Public instance for one N-to-1 evaluation reduction round.
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize, Allocative)]
pub struct EvalReductionInstance<F: JoltField> {
    openings: Vec<(Vec<F>, F)>, // Vec of (r, v) pairs for each opening claim
}

/// Witness for one N-to-1 evaluation reduction round.
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize, Allocative)]
pub struct EvalReductionWitness<F: JoltField> {
    pub mle: MultilinearPolynomial<F>,
}

/// Proof for one N-to-1 evaluation reduction round.
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize, Allocative)]
pub struct EvalReductionProof<F: JoltField> {
    /// Restriction polynomial h(t) = P(l(t)).
    pub h: UniPoly<F>,
}

/// Reduced instance produced by the N-to-1 evaluation reduction.
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize, Allocative)]
pub struct ReducedInstance<F: JoltField> {
    pub r: Vec<F>,
    pub claim: F,
}

impl<F: JoltField> EvalReductionInstance<F> {
    pub fn new(openings: &[&Opening<F>]) -> Self {
        let openings: Vec<(Vec<F>, F)> = openings
            .iter()
            .map(|&opening| (opening.0.clone().into(), opening.1))
            .collect::<Vec<_>>();

        Self { openings }
    }
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

impl<F: JoltField> From<Opening<F>> for ReducedInstance<F> {
    fn from(opening: Opening<F>) -> Self {
        Self {
            r: opening.0.into(),
            claim: opening.1,
        }
    }
}

impl<F: JoltField> EvalReductionInstance<F> {
    pub fn prove<T: Transcript>(
        &self,
        witness: &EvalReductionWitness<F>,
        transcript: &mut T,
    ) -> Result<(EvalReductionProof<F>, ReducedInstance<F>), ProvingError> {
        if self.openings.is_empty() {
            return Err(ProvingError::EmptyInput);
        }

        let opening_points = self.openings.iter().map(|(r, _)| r).collect::<Vec<_>>();
        let num_vars = witness.mle.get_num_vars();

        if let Some(opening) = opening_points.iter().find(|r| r.len() != num_vars) {
            return Err(ProvingError::InvalidInputLength(
                witness.mle.get_num_vars(),
                opening.len(),
            ));
        }

        if opening_points.len() == 1 {
            // Short path:
            // If there's only one opening, we can skip the reduction and directly return the claim as the reduced instance.
            let (r, claim) = &self.openings[0];
            let reduced_instance = ReducedInstance {
                r: r.clone(),
                claim: *claim,
            };
            let proof = EvalReductionProof {
                h: UniPoly::from_coeff(vec![*claim]), // h is just the constant polynomial equal to the claim
            };

            return Ok((proof, reduced_instance));
        }

        // i'th vector of this Vec<Vec<F>> is the vector of i'th variables across all evaluation points.
        let ri_vec = group_by_variable(&opening_points);
        // h = mle ∘ l, where l is the unique n-1 degree polynomial defined by the n opening points.
        let h = compute_h(&witness.mle, &ri_vec);

        #[cfg(test)]
        {
            for (i, (_, claim)) in self.openings.iter().enumerate() {
                let eval_at_i = h.evaluate(&F::from_u32(i as u32));
                assert_eq!(eval_at_i, *claim, "h does not match opening claim at t={i}");
            }
        }

        h.append_to_transcript(transcript);
        let x_prime: F::Challenge = transcript.challenge_scalar_optimized::<F>();

        let proof = EvalReductionProof { h };
        let reduced_instance = ReducedInstance {
            r: eval_on_l(&ri_vec, x_prime.into()),
            claim: proof.h.evaluate(&x_prime),
        };

        Ok((proof, reduced_instance))
    }

    pub fn verify<T: Transcript>(
        &self,
        proof: &EvalReductionProof<F>,
        transcript: &mut T,
    ) -> Result<ReducedInstance<F>, ProofVerifyError> {
        if self.openings.is_empty() {
            return Err(ProofVerifyError::EmptyInput);
        }

        let opening_points = self.openings.iter().map(|(r, _)| r).collect::<Vec<_>>();
        let n_vars = opening_points[0].len();

        if let Some(opening) = opening_points.iter().find(|r| r.len() != n_vars) {
            return Err(ProofVerifyError::InvalidInputLength(n_vars, opening.len()));
        }

        if opening_points.len() == 1 {
            // Short path:
            // If there's only one opening, we can skip the reduction and directly return the claim as the reduced instance.
            let (r, claim) = &self.openings[0];
            let reduced_instance = ReducedInstance {
                r: r.clone(),
                claim: *claim,
            };

            return Ok(reduced_instance);
        }

        let ri_vec = group_by_variable(&opening_points);

        let n_openings = self.openings.len();

        if proof.h.degree() > n_vars * (n_openings - 1) {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "Degree of h should be bounded by number of variables in original MLE and number of openings; \
                expected degree <= num_vars * (num_openings - 1), got {} > {}",
                proof.h.degree(),
                n_vars * (n_openings - 1)
            )));
        }

        for (i, (_, claim)) in self.openings.iter().enumerate() {
            let eval_at_i = proof.h.evaluate(&F::from_u32(i as u32));
            if eval_at_i != *claim {
                return Err(ProofVerifyError::InvalidOpeningProof(format!(
                    "h does not match opening claim at t={i}: expected h({i}) = {claim}, got {eval_at_i}"
                )));
            }
        }

        proof.h.append_to_transcript(transcript);
        let x_prime = transcript.challenge_scalar_optimized::<F>();

        let expected_reduced_value = proof.h.evaluate::<F>(&x_prime.into());
        let expected_r = eval_on_l(&ri_vec, x_prime.into());

        Ok(ReducedInstance {
            r: expected_r,
            claim: expected_reduced_value,
        })
    }
}

fn eval_on_l<F: JoltField, Ref: AsRef<[F]>>(ri_vec: &[Ref], x: F) -> Vec<F> {
    ri_vec
        .iter()
        .map(|r_i| {
            let variable_poly = UniPoly::from_evals(r_i.as_ref());
            variable_poly.evaluate(&x)
        })
        .collect()
}

fn compute_h<F: JoltField, Ref: AsRef<[F]>>(
    mle: &MultilinearPolynomial<F>,
    ri_vec: &[Ref],
) -> UniPoly<F> {
    let num_vars = ri_vec.len();

    let mut mle_polys: Vec<UniPoly<F>> = mle
        .coeffs()
        .into_iter()
        .map(|coeff| {
            UniPoly::from_coeff(vec![coeff]) // Start with constant polynomial for each coeff
        })
        .collect();

    for (i, r_i) in ri_vec.iter().enumerate() {
        let var_weight = num_vars - i - 1;
        let half_len = 1 << var_weight;
        let var_poly = UniPoly::from_evals(r_i.as_ref()); // r1_i + t * delta_i
        for j in 0..half_len {
            let left = std::mem::take(&mut mle_polys[j]);
            let right = std::mem::take(&mut mle_polys[j + half_len]);
            mle_polys[j] = &left + &var_poly * (&right - &left);
        }
    }

    std::mem::take(&mut mle_polys[0])
}

fn group_by_variable<F: JoltField, Ref: AsRef<[F]>>(evaluation_points: &[Ref]) -> Vec<Vec<F>> {
    let num_vars = evaluation_points[0].as_ref().len();
    debug_assert!(
        evaluation_points
            .iter()
            .all(|p| p.as_ref().len() == num_vars),
        "All evaluation points should have the same number of variables"
    );

    // Create a vector of vector where ri_vec[k][j] = evaluation_points[j][k],
    // i.e. the vector of i'th variables across all evaluation points.
    let mut r_i_vec: Vec<Vec<F>> = vec![vec![]; num_vars];
    for p_i in evaluation_points {
        for (i, &r_i) in p_i.as_ref().iter().enumerate() {
            r_i_vec[i].push(r_i);
        }
    }
    r_i_vec
}

pub struct EvalReductionProtocol;

impl EvalReductionProtocol {
    pub fn prove<F: JoltField, T: Transcript>(
        openings: &[&Opening<F>],
        output_mle: MultilinearPolynomial<F>,
        transcript: &mut T,
    ) -> Result<(EvalReductionProof<F>, ReducedInstance<F>), ProvingError> {
        let witness = EvalReductionWitness::new(output_mle);
        let instance = EvalReductionInstance::new(openings);

        instance.prove(&witness, transcript)
    }

    pub fn verify<F: JoltField, T: Transcript>(
        openings: &[&Opening<F>],
        proof: &EvalReductionProof<F>,
        transcript: &mut T,
    ) -> Result<ReducedInstance<F>, ProofVerifyError> {
        let instance = EvalReductionInstance::new(openings);

        instance.verify(proof, transcript)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        field::{IntoOpening, JoltField},
        poly::multilinear_polynomial::PolynomialEvaluation,
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
    // Asserts that the helper performing the grouping of evaluation points by variable works correctly.
    fn correctly_groups_by_variable() {
        let evaluation_points = [
            vec![f(1), f(2), f(3)],
            vec![f(4), f(5), f(6)],
            vec![f(7), f(8), f(9)],
        ];
        let grouped = group_by_variable(&evaluation_points);
        assert_eq!(
            grouped,
            vec![
                vec![f(1), f(4), f(7)],
                vec![f(2), f(5), f(8)],
                vec![f(3), f(6), f(9)],
            ]
        );
    }

    #[test]
    // Use different openings on which we know the univariate relation on each variable,
    // and assert the `l` evaluation on randomly generated points, for 2, and 3 openings.
    fn eval_reduction_compute_l_matches_expected() {
        let mut rng = StdRng::seed_from_u64(0x888);

        for _ in 0..100 {
            // 2-eval points
            let r1 = vec![f(1), f(2), f(3)];
            let r2 = vec![f(3), f(6), f(9)];
            let x = Fr::random(&mut rng);

            // each var equals r1_i + x * (r2_i - r1_i)
            let expected = vec![f(1) + x * f(2), f(2) + x * f(4), f(3) + x * f(6)];

            let r_i_vec = group_by_variable(&[&r1, &r2]);
            let result = eval_on_l(&r_i_vec, x);
            assert_eq!(result, expected);

            // 3-eval points
            // r3 = [
            //  1 + 2 * 1 + 2² * 1 = 7,
            //  2 + 2 * 2 + 2² * 2 = 14,
            //  3 + 2 * 3 + 2² * 3 = 21,
            //]
            let r3 = vec![f(7), f(14), f(21)];
            let r_i_vec = group_by_variable(&[&r1, &r2, &r3]);
            let result = eval_on_l(&r_i_vec, x);
            let expected = vec![
                f(1) + x * f(1) + x * x * f(1),
                f(2) + x * f(2) + x * x * f(2),
                f(3) + x * f(3) + x * x * f(3),
            ];
            assert_eq!(result, expected);
        }
    }

    #[test]
    // Asserts that the evaluation of l defined from randomly generated openings,
    // evaluated at the index of one of those openings.
    fn test_l_evaluation() {
        let mut rng = StdRng::seed_from_u64(0x888);

        for _ in 0..100 {
            let num_vars = rng.sample(Uniform::from(2..10));
            let num_openings = rng.sample(Uniform::from(1..10));
            let openings_vec: Vec<Vec<Fr>> = (0..num_openings)
                .map(|_| {
                    (0..num_vars)
                        .map(|_| Fr::random(&mut rng))
                        .collect::<Vec<_>>()
                })
                .collect();

            // Hard to compute a completely random evaluation of l, so we just test that we
            // indeed recover any of the points that are guaranteed to be on the line.
            let x = rng.sample(Uniform::from(0..num_openings));
            let ri_vec = group_by_variable(&openings_vec);

            let l_eval = eval_on_l(&ri_vec, Fr::from(x));
            let l_direct: Vec<Fr> = openings_vec[x as usize].clone();

            assert_eq!(l_eval, l_direct);
        }
    }

    #[test]
    // Asserts the correct computation of restriction h of mle to the line l; h = mle ∘ l
    // In this test we assert that h(r1) = v1 and h(r2) = v2,
    // and that degree of h is at most the number of variables in the original MLE.
    fn test_h_computation() {
        let mut rng = StdRng::seed_from_u64(0x888);

        for _ in 0..100 {
            let num_vars = rng.sample(Uniform::from(2..10));
            let num_openings = rng.sample(Uniform::from(1..10));
            let mle = MultilinearPolynomial::from(
                (0..1 << num_vars)
                    .map(|_| Fr::random(&mut rng))
                    .collect::<Vec<_>>(),
            );

            let openings_vec: Vec<Vec<Fr>> = (0..num_openings)
                .map(|_| {
                    (0..num_vars)
                        .map(|_| Fr::random(&mut rng))
                        .collect::<Vec<_>>()
                })
                .collect();

            let ri_vec = group_by_variable(&openings_vec);
            let h = compute_h(&mle, &ri_vec);

            assert!(h.degree() <= num_vars * (num_openings - 1));
            for (i, opening) in openings_vec.iter().enumerate() {
                let v = mle.evaluate(opening);
                assert_eq!(h.evaluate(&Fr::from(i as u64)), v);
            }
        }
    }

    #[test]
    // Asserts that with a random generated challenge t_f,
    // the evaluation of h at t_f matches the evaluation of the original MLE at the corresponding l(t_f).
    fn h_restr_l_matches_expected() {
        let mut rng = StdRng::seed_from_u64(0x888);

        for _ in 0..100 {
            let num_vars = rng.sample(Uniform::from(2..10));
            let num_openings = rng.sample(Uniform::from(1..10));

            let mle = MultilinearPolynomial::from(
                (0..1 << num_vars)
                    .map(|_| Fr::random(&mut rng))
                    .collect::<Vec<_>>(),
            );
            let openings_vec: Vec<Vec<Fr>> = (0..num_openings)
                .map(|_| {
                    (0..num_vars)
                        .map(|_| Fr::random(&mut rng))
                        .collect::<Vec<_>>()
                })
                .collect();
            let r_i_vec = group_by_variable(&openings_vec);
            let h = compute_h(&mle, &r_i_vec);

            // Verify random challenges
            for _ in 0..10 {
                let t_f: Fr = rng.gen();
                let l_eval = eval_on_l(&r_i_vec, t_f);
                let h_eval = h.evaluate(&t_f);
                let mle_eval = mle.evaluate(&l_eval);
                assert_eq!(h_eval, mle_eval);
            }
        }
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
        let instance = EvalReductionInstance::new(&[&opening1, &opening2]);
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
    fn eval_reduction_single_opening_is_propagated() {
        let mle = MultilinearPolynomial::<Fr>::from(vec![f(1), f(2), f(3), f(4)]);
        let r = vec![ch(7), ch(9)];
        let v = mle.evaluate(&r);
        let opening: Opening<Fr> = (r.clone().into(), v);

        let mut prover_tr = Blake2bTranscript::new(b"eval-red-single");
        let (proof, reduced_prover) =
            EvalReductionProtocol::prove::<Fr, _>(&[&opening], mle, &mut prover_tr)
                .expect("single opening should be propagated by prover");

        assert_eq!(reduced_prover.r, r.into_opening());
        assert_eq!(reduced_prover.claim, v);
        assert!(
            proof.h.degree() == 0,
            "h should be constant for single opening case"
        );

        let mut verifier_tr = Blake2bTranscript::new(b"eval-red-single");
        let reduced_verifier =
            EvalReductionProtocol::verify::<Fr, _>(&[&opening], &proof, &mut verifier_tr)
                .expect("single opening should be propagated by verifier");

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
        let instance = EvalReductionInstance::new(&[&opening1, &opening2]);
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
        let instance = EvalReductionInstance::new(&[&opening1, &opening2]);
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
    fn eval_reduction_nto1() {
        let mle =
            MultilinearPolynomial::<Fr>::from(vec![f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8)]);
        let r1 = vec![ch(1), ch(2), ch(3)];
        let r2 = vec![ch(4), ch(6), ch(8)];
        let r3 = vec![ch(7), ch(14), ch(21)];

        let v1 = mle.evaluate(&r1);
        let v2 = mle.evaluate(&r2);
        let v3 = mle.evaluate(&r3);

        let opening1: Opening<Fr> = (r1.clone().into(), v1);
        let opening2: Opening<Fr> = (r2.clone().into(), v2);
        let opening3: Opening<Fr> = (r3.clone().into(), v3);

        let witness = EvalReductionWitness { mle };

        let mut prover_tr = Blake2bTranscript::new(b"eval-reduction-test");
        let instance = EvalReductionInstance::new(&[&opening1, &opening2, &opening3]);
        let (proof, reduced) = instance.prove(&witness, &mut prover_tr).unwrap();
        let mut verifier_tr = Blake2bTranscript::new(b"eval-reduction-test");
        let reduced_verifier = instance.verify(&proof, &mut verifier_tr).unwrap();
        assert_eq!(reduced_verifier, reduced);
    }
}
