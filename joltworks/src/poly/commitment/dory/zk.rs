//! Native hiding openings. The returned evaluation commitment must also be
//! constrained by BlindFold; accepting the PCS proof alone does not bind a
//! sumcheck claim to its polynomial.

use super::*;
use crate::{
    curve::{Bn254Curve, Bn254G1},
    poly::commitment::pedersen::PedersenGenerators,
};
use std::collections::BTreeMap;

impl DoryScheme {
    /// Keep sparse row commitments and mask their pairing with fresh randomness.
    /// The hint, including its blind, is private prover state.
    pub fn commit_zk(
        poly: &MultilinearPolynomial<Fr>,
        setup: &DoryProverSetup,
    ) -> (DoryCommitment, DoryHint) {
        let (mut commitment, mut hint) = Self::commit(poly, setup);
        hint.commit_blind = <ArkFr as DoryField>::random();
        commitment.0 = commitment.0 + setup.prover.ht.scale(&hint.commit_blind);
        (commitment, hint)
    }

    /// Use Dory's evaluation bases in BlindFold as well. In particular the first
    /// message generator and blinding generator must agree with `proof.y_com`.
    pub fn pedersen_generators(
        setup: &DoryProverSetup,
        count: usize,
    ) -> PedersenGenerators<Bn254Curve> {
        assert!(count > 0 && count <= setup.prover.g1_vec.len());
        PedersenGenerators::new(
            setup.prover.g1_vec[..count]
                .iter()
                .map(|g| Bn254G1(g.0))
                .collect(),
            Bn254G1(setup.prover.h1.0),
        )
    }

    /// Prove a sparse linear combination without disclosing its evaluation.
    /// Every hint must come from the corresponding prior hiding commitment.
    /// Returns the proof, its evaluation commitment, and the private evaluation
    /// blind needed for the BlindFold witness.
    pub fn prove_rlc_zk<T: Transcript>(
        setup: &DoryProverSetup,
        polynomials: &BTreeMap<common::CommittedPoly, MultilinearPolynomial<Fr>>,
        coeffs: &[Fr],
        hints: Vec<DoryHint>,
        opening_point: &[<Fr as JoltField>::Challenge],
        transcript: &mut T,
    ) -> Result<(DoryProof, Bn254G1, Fr), ProofVerifyError> {
        if polynomials.is_empty()
            || coeffs.len() != polynomials.len()
            || hints.len() != polynomials.len()
        {
            return Err(invalid("Missing hiding commitment hints"));
        }
        let joint = SparseRlc::new(coeffs, polynomials, opening_point.len());
        Self::prove_joint_zk(setup, joint, coeffs, hints, opening_point, transcript)
    }

    /// Consume polynomial inputs once their dense joint has been constructed.
    /// Callers must finish every individual opening before handing over ownership.
    pub fn prove_rlc_zk_owned<T: Transcript>(
        setup: &DoryProverSetup,
        polynomials: BTreeMap<common::CommittedPoly, MultilinearPolynomial<Fr>>,
        coeffs: &[Fr],
        hints: Vec<DoryHint>,
        opening_point: &[<Fr as JoltField>::Challenge],
        transcript: &mut T,
    ) -> Result<(DoryProof, Bn254G1, Fr), ProofVerifyError> {
        if polynomials.is_empty()
            || coeffs.len() != polynomials.len()
            || hints.len() != polynomials.len()
        {
            return Err(invalid("Missing hiding commitment hints"));
        }
        let joint = SparseRlc::new_owned(coeffs, polynomials, opening_point.len());
        Self::prove_joint_zk(setup, joint, coeffs, hints, opening_point, transcript)
    }

    fn prove_joint_zk<T: Transcript>(
        setup: &DoryProverSetup,
        joint: SparseRlc<'_>,
        coeffs: &[Fr],
        hints: Vec<DoryHint>,
        opening_point: &[<Fr as JoltField>::Challenge],
        transcript: &mut T,
    ) -> Result<(DoryProof, Bn254G1, Fr), ProofVerifyError> {
        let num_vars = opening_point.len();
        let (nu, sigma) = Self::split(num_vars, Self::column_log(setup));
        let point = Self::dory_point(opening_point);
        let (mut rows, commit_blind) = Self::combine_hints(hints, coeffs).into_parts();
        rows.resize(1 << nu, <ArkG1 as DoryGroup>::identity());
        let (proof, blind) = dory_prove::<_, BN254, ParG1Routines, ParG2Routines, _, _, dory::ZK>(
            &joint,
            &point,
            rows,
            commit_blind,
            nu,
            sigma,
            &setup.prover,
            &mut LocalToDoryTranscript::new(transcript),
        )
        .map_err(|e| invalid(&format!("Dory hiding opening failed: {e:?}")))?;
        let y_com = proof
            .y_com
            .ok_or_else(|| invalid("Missing evaluation commitment"))?;
        let blind = blind.ok_or_else(|| invalid("Missing evaluation blind"))?;
        Ok((DoryProof(proof), Bn254G1(y_com.0), blind.0))
    }

    /// Verify a hiding opening and require the exact commitment checked by
    /// BlindFold. Transparent and mixed proof modes are rejected by this API.
    pub fn verify_zk<T: Transcript>(
        proof: &DoryProof,
        setup: &DoryVerifierSetup,
        transcript: &mut T,
        opening_point: &[<Fr as JoltField>::Challenge],
        evaluation_commitment: &Bn254G1,
        commitment: &DoryCommitment,
    ) -> Result<(), ProofVerifyError> {
        let (nu, sigma) = Self::split(opening_point.len(), setup.0.max_log_n / 2);
        if opening_point.len() > setup.0.max_log_n || proof.0.nu != nu || proof.0.sigma != sigma {
            return Err(invalid(
                "Hiding opening layout does not match the registered setup",
            ));
        }
        if proof.0.y_com.map(|g| Bn254G1(g.0)).as_ref() != Some(evaluation_commitment) {
            return Err(invalid(
                "Missing or mismatched hiding evaluation commitment",
            ));
        }
        // Dory's ZK verifier uses y_com and ignores the clear evaluation argument.
        // Its mode check rejects all partially populated ZK proof shapes.
        dory_verify::<_, BN254, G1Routines, G2Routines, _>(
            commitment.0,
            <ArkFr as DoryField>::zero(),
            &Self::dory_point(opening_point),
            &proof.0,
            setup.0.clone(),
            &mut LocalToDoryTranscript::new(transcript),
        )
        .map_err(|e| invalid(&format!("Dory hiding verification failed: {e:?}")))
    }
}

fn invalid(message: &str) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        curve::JoltGroupElement, poly::multilinear_polynomial::PolynomialEvaluation,
        transcripts::Blake2bTranscript,
    };
    use ark_ff::{One, Zero};
    use common::CommittedPoly;

    #[test]
    fn hidden_opening_binds_blindfold_value_and_rejects_substitution() {
        let setup = DoryScheme::setup_prover(8);
        let verifier_setup = DoryScheme::setup_verifier(&setup);
        let polynomials = BTreeMap::from([
            (
                CommittedPoly::DivNodeQuotient(0),
                MultilinearPolynomial::from(vec![1i64, 4, -2, 9]),
            ),
            (
                CommittedPoly::DivNodeQuotient(1),
                MultilinearPolynomial::from(vec![3i64, 7, 0, -5, 12, 8, 6, 2]),
            ),
        ]);
        let coefficients = vec![Fr::from(5u64), Fr::from(13u64)];
        let (commitments, hints): (Vec<_>, Vec<_>) = polynomials
            .values()
            .map(|poly| DoryScheme::commit_zk(poly, &setup))
            .unzip();
        let commitment = DoryScheme::combine_commitments(&commitments, &coefficients);
        let mut challenge_transcript = Blake2bTranscript::new(b"hidden-opening-point");
        let point: Vec<<Fr as JoltField>::Challenge> = (0..3)
            .map(|_| challenge_transcript.challenge_scalar_optimized::<Fr>())
            .collect();
        let (proof, y_com, blind) = DoryScheme::prove_rlc_zk(
            &setup,
            &polynomials,
            &coefficients,
            hints,
            &point,
            &mut Blake2bTranscript::new(b"native-hiding-opening"),
        )
        .unwrap();

        // Independent direct MLE, including zero padding of the shorter input.
        let combined = MultilinearPolynomial::from(vec![44i64, 111, -10, -20, 156, 104, 78, 26]);
        let expected = combined.evaluate(&point);
        let generators = DoryScheme::pedersen_generators(&setup, 4);
        assert_eq!(y_com, generators.commit(&[expected], &blind));
        let verify = |p: &DoryProof, c: &DoryCommitment, r: &[_], y: &Bn254G1| {
            DoryScheme::verify_zk(
                p,
                &verifier_setup,
                &mut Blake2bTranscript::new(b"native-hiding-opening"),
                r,
                y,
                c,
            )
        };
        assert!(verify(&proof, &commitment, &point, &y_com).is_ok());
        let wrong_y = y_com + generators.message_generators[0];
        assert!(verify(&proof, &commitment, &point, &wrong_y).is_err());
        assert!(verify(&proof, &commitments[0], &point, &y_com).is_err());
        let mut wrong_point = point.clone();
        wrong_point.reverse();
        assert_ne!(wrong_point, point);
        assert!(verify(&proof, &commitment, &wrong_point, &y_com).is_err());
        let mut missing = proof.clone();
        missing.0.y_com = None;
        assert!(verify(&missing, &commitment, &point, &y_com).is_err());
        let mut replaced = proof.clone();
        replaced.0.y_com = Some(ArkG1(wrong_y.0));
        assert!(verify(&replaced, &commitment, &point, &wrong_y).is_err());

        let (fresh, _) = DoryScheme::commit_zk(polynomials.values().next().unwrap(), &setup);
        assert_ne!(fresh, commitments[0]);
        assert_ne!(blind, Fr::zero());
        assert!(!y_com.is_zero());
        assert_ne!(expected, Fr::one());
    }
}

#[cfg(test)]
mod native_batch_tests {
    use super::*;
    use crate::{
        curve::JoltGroupElement,
        poly::{
            multilinear_polynomial::PolynomialEvaluation,
            opening_proof::{
                OpeningId, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            },
        },
        subprotocols::blindfold::{
            assembly::NativeBlindFold, protocol::BlindFoldVerifierInput,
            witness::ExtraConstraintWitness, BlindFoldAccumulator, OutputClaimConstraint,
        },
        transcripts::Blake2bTranscript,
    };
    use ark_ff::{One, Zero};
    use common::CommittedPoly;

    #[test]
    fn native_batch_blindfold_binds_hidden_claims_to_dory() {
        let setup = DoryScheme::setup_prover(8);
        let vp = DoryScheme::setup_verifier(&setup);
        let gens = DoryScheme::pedersen_generators(&setup, 4);
        use crate::poly::one_hot_polynomial::OneHotPolynomial;
        let polynomials = BTreeMap::from([
            (
                CommittedPoly::DivNodeQuotient(0),
                MultilinearPolynomial::from(vec![1i64, 4, -2, 9]),
            ),
            (
                CommittedPoly::DivNodeQuotient(1),
                MultilinearPolynomial::from(vec![3i64, 7, 0, -5, 12, 8, 6, 2]),
            ),
            (
                CommittedPoly::DivNodeQuotient(2),
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    vec![Some(0), Some(1), None, Some(3)],
                    4,
                )),
            ),
            (
                CommittedPoly::DivNodeQuotient(3),
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    vec![Some(2), Some(0), Some(1), Some(2)],
                    4,
                )),
            ),
        ]);
        let (commitments, hints): (Vec<_>, Vec<_>) = polynomials
            .values()
            .map(|p| DoryScheme::commit_zk(p, &setup))
            .unzip();
        let mut pt = Blake2bTranscript::new(b"native-batch-blindfold");
        let mut vt = pt.clone();
        for c in &commitments {
            pt.append_serializable(c);
            vt.append_serializable(c);
        }
        let mut pa = ProverOpeningAccumulator::new();
        pa.zk_mode = true;
        let mut va = VerifierOpeningAccumulator::new_zk();
        let mut constraints = Vec::new();
        let mut extras = Vec::new();
        let mut input_commitments = Vec::new();
        for (node, consumer, coords) in [
            (0, 0, vec![2u64, 3]),
            (0, 1, vec![5, 7]),
            (1, 2, vec![11, 13, 17]),
        ] {
            let poly = CommittedPoly::DivNodeQuotient(node);
            let id = OpeningId::new(poly, SumcheckId::NodeExecution(consumer));
            let point: Vec<Fr> = coords.into_iter().map(Fr::from).collect();
            let value = polynomials[&poly].evaluate(&point);
            let blind = <Fr as JoltField>::random(&mut rand::thread_rng());
            let c = gens.commit(&[value], &blind);
            pt.append_scalars(&point);
            vt.append_scalars(&point);
            pt.append_serializable(&c);
            vt.append_serializable(&c);
            pa.append_dense(&mut pt, id, point.clone(), value);
            va.append_dense(&mut vt, id, point);
            constraints.push(OutputClaimConstraint::direct(id));
            extras.push(ExtraConstraintWitness {
                output_value: value,
                blinding: blind,
                challenge_values: vec![],
                opening_values: vec![value],
            });
            input_commitments.push(c);
        }
        let sparse_polys = vec![
            CommittedPoly::DivNodeQuotient(2),
            CommittedPoly::DivNodeQuotient(3),
        ];
        let sparse_point: Vec<Fr> = [19u64, 23, 29, 31].into_iter().map(Fr::from).collect();
        pt.append_scalars(&sparse_point);
        vt.append_scalars(&sparse_point);
        let sparse_claims: Vec<Fr> = sparse_polys
            .iter()
            .map(|p| polynomials[p].evaluate(&sparse_point))
            .collect();
        for (poly, value) in sparse_polys.iter().zip(&sparse_claims) {
            let id = OpeningId::new(*poly, SumcheckId::NodeExecution(3));
            let blind = <Fr as JoltField>::random(&mut rand::thread_rng());
            let c = gens.commit(&[*value], &blind);
            pt.append_serializable(&c);
            vt.append_serializable(&c);
            constraints.push(OutputClaimConstraint::direct(id));
            extras.push(ExtraConstraintWitness {
                output_value: *value,
                blinding: blind,
                challenge_values: vec![],
                opening_values: vec![*value],
            });
            input_commitments.push(c);
        }
        pa.append_sparse(
            &mut pt,
            sparse_polys.clone(),
            SumcheckId::NodeExecution(3),
            sparse_point[..2].to_vec(),
            sparse_point[2..].to_vec(),
            sparse_claims,
        );
        va.append_sparse(
            &mut vt,
            sparse_polys,
            SumcheckId::NodeExecution(3),
            sparse_point,
        );
        let groups = va.num_sumchecks();
        let mut bf = BlindFoldAccumulator::new();
        pa.prepare_for_sumcheck(&polynomials, &mut pt);
        let (sumcheck, point) = pa.prove_batch_opening_sumcheck_zk::<_, Bn254Curve, _>(
            &mut bf,
            &mut vec![],
            &gens,
            &mut rand::thread_rng(),
            &mut pt,
        );
        let state = pa.finalize_batch_opening_sumcheck(point, &mut pt);
        let verified_point = va
            .verify_batch_opening_sumcheck_zk(&sumcheck, &mut vt, 4)
            .unwrap();
        let vstate =
            va.finalize_batch_opening_sumcheck(verified_point, &vec![Fr::zero(); groups], &mut vt);
        assert_eq!(state.r_sumcheck, vstate.r_sumcheck);
        assert_eq!(state.poly_coeffs, vstate.poly_coeffs);
        let (relation, coefficients) = state.hidden_evaluation_relation();
        let (vrelation, vcoefficients) = vstate.hidden_evaluation_relation();
        assert_eq!(relation.required_openings, vrelation.required_openings);
        assert_eq!(coefficients, vcoefficients);
        // One group contains two sparse polynomials. Both openings of the first
        // polynomial must survive reduction as distinct constrained claims.
        assert_eq!(relation.required_openings.len(), 4);
        assert_ne!(relation.required_openings[0], relation.required_openings[1]);
        let joint_value: Fr = coefficients
            .iter()
            .zip(&state.sumcheck_claims)
            .map(|(a, b)| *a * *b)
            .sum();
        let joint_commitment = DoryScheme::combine_commitments(&commitments, &state.poly_coeffs);
        let (pcs, eval, blind) = DoryScheme::prove_rlc_zk(
            &setup,
            &polynomials,
            &state.poly_coeffs,
            hints,
            &state.r_sumcheck,
            &mut pt,
        )
        .unwrap();
        assert_eq!(eval, gens.commit(&[joint_value], &blind));
        DoryScheme::verify_zk(
            &pcs,
            &vp,
            &mut vt,
            &vstate.r_sumcheck,
            &eval,
            &joint_commitment,
        )
        .unwrap();
        constraints.push(relation);
        extras.push(ExtraConstraintWitness {
            output_value: joint_value,
            blinding: blind,
            challenge_values: coefficients.clone(),
            opening_values: state.sumcheck_claims.clone(),
        });
        input_commitments.push(eval);
        let data = bf.take_stage_data();
        let native_prover = NativeBlindFold::new(
            NativeBlindFold::prover_relations(&data),
            &constraints,
            &coefficients,
            4,
        )
        .unwrap();
        let openings = pa
            .openings
            .iter()
            .map(|(id, (_, value))| (*id, *value))
            .collect();
        let proof = native_prover
            .prove(
                &data,
                &openings,
                extras,
                input_commitments.clone(),
                &gens,
                &mut pt,
            )
            .unwrap();
        let native_verifier =
            NativeBlindFold::new(va.zk_stages.clone(), &constraints, &vcoefficients, 4).unwrap();
        let mut input = BlindFoldVerifierInput {
            round_commitments: sumcheck.round_commitments.clone(),
            output_claims_row_commitments: sumcheck.output_claims_commitments.clone(),
            eval_commitments: input_commitments,
        };
        let verify = |i: &BlindFoldVerifierInput<_>| {
            native_verifier.verify(&proof, i, &gens, &mut vt.clone())
        };
        verify(&input).unwrap();
        input.eval_commitments[0] += gens.message_generators[0];
        assert!(verify(&input).is_err());
        input.eval_commitments[0] -= gens.message_generators[0];
        input.round_commitments[0] += gens.message_generators[0];
        assert!(verify(&input).is_err());
        input.round_commitments[0] -= gens.message_generators[0];
        let oc = input.output_claims_row_commitments.pop().unwrap();
        assert!(verify(&input).is_err());
        input.output_claims_row_commitments.push(oc);
        input.eval_commitments.pop();
        assert!(verify(&input).is_err());
        assert!(!eval.is_zero());
        let mut wrong_coefficients = vcoefficients;
        wrong_coefficients[0] += Fr::one();
        let wrong_relation =
            NativeBlindFold::new(va.zk_stages, &constraints, &wrong_coefficients, 4).unwrap();
        input.eval_commitments.push(eval);
        assert!(wrong_relation
            .verify(&proof, &input, &gens, &mut vt.clone())
            .is_err());
    }
}

#[cfg(test)]
mod owned_proof_tests {
    use super::*;
    use crate::{
        poly::{
            multilinear_polynomial::PolynomialEvaluation, one_hot_polynomial::OneHotPolynomial,
        },
        transcripts::Blake2bTranscript,
    };
    use common::CommittedPoly;

    #[test]
    fn native_owned_joint_proof_binds_dense_sparse_and_hiding_value() {
        let setup = DoryScheme::setup_prover(8);
        let verifier = DoryScheme::setup_verifier(&setup);
        let sparse = OneHotPolynomial::from_indices(vec![Some(0), None, Some(3), Some(1)], 4);
        let polynomials = BTreeMap::from([
            (
                CommittedPoly::DivNodeQuotient(0),
                MultilinearPolynomial::from(vec![-7i32, 3, 0, 1]),
            ),
            (
                CommittedPoly::NodeOutputRaD(0, 0),
                MultilinearPolynomial::OneHot(sparse),
            ),
        ]);
        let coefficients: Vec<Fr> = polynomials
            .keys()
            .map(|id| match id {
                CommittedPoly::DivNodeQuotient(0) => Fr::from(3u64),
                CommittedPoly::NodeOutputRaD(0, 0) => Fr::from(7u64),
                _ => unreachable!(),
            })
            .collect();
        let (commitments, hints): (Vec<_>, Vec<_>) = polynomials
            .values()
            .map(|p| DoryScheme::commit_zk(p, &setup))
            .unzip();
        let commitment = DoryScheme::combine_commitments(&commitments, &coefficients);
        let mut challenges = Blake2bTranscript::new(b"owned joint point");
        let point: Vec<<Fr as JoltField>::Challenge> = (0..6)
            .map(|_| challenges.challenge_scalar_optimized::<Fr>())
            .collect();
        let mut dense = vec![0i32; 64];
        for (i, x) in [-7, 3, 0, 1].iter().enumerate() {
            dense[i] += 3 * x;
        }
        for i in [0, 14, 7] {
            dense[i] += 7;
        }
        let expected = MultilinearPolynomial::from(dense).evaluate(&point);
        let generators = DoryScheme::pedersen_generators(&setup, 4);
        let (borrowed_proof, borrowed_value, borrowed_blind) = DoryScheme::prove_rlc_zk(
            &setup,
            &polynomials,
            &coefficients,
            hints.clone(),
            &point,
            &mut Blake2bTranscript::new(b"owned joint proof"),
        )
        .unwrap();
        assert_eq!(
            borrowed_value,
            generators.commit(&[expected], &borrowed_blind)
        );
        let (proof, value, blind) = DoryScheme::prove_rlc_zk_owned(
            &setup,
            polynomials,
            &coefficients,
            hints,
            &point,
            &mut Blake2bTranscript::new(b"owned joint proof"),
        )
        .unwrap();
        assert_eq!(value, generators.commit(&[expected], &blind));
        let verify = |proof: &DoryProof, value: &Bn254G1, commitment: &DoryCommitment| {
            DoryScheme::verify_zk(
                proof,
                &verifier,
                &mut Blake2bTranscript::new(b"owned joint proof"),
                &point,
                value,
                commitment,
            )
        };
        assert!(verify(&borrowed_proof, &borrowed_value, &commitment).is_ok());
        assert!(verify(&proof, &value, &commitment).is_ok());
        assert!(verify(&proof, &value, &commitments[0]).is_err());
        assert!(verify(
            &proof,
            &(value + generators.message_generators[0]),
            &commitment
        )
        .is_err());
        let mut missing = proof.clone();
        missing.0.y_com = None;
        assert!(verify(&missing, &value, &commitment).is_err());
        assert!(DoryScheme::prove_rlc_zk_owned(
            &setup,
            BTreeMap::new(),
            &[],
            vec![],
            &point,
            &mut Blake2bTranscript::new(b"empty owned joint")
        )
        .is_err());
    }
}
