use super::*;
use atlas_algebra::{
    field::akita::AkitaField,
    poly::{
        multilinear::{MultilinearPolynomial, PolynomialEvaluation},
        one_hot::OneHotPolynomial,
    },
};
use atlas_sumcheck::{
    opening_proof::{
        prover::ProverOpeningAccumulator, OpeningAccumulator, OpeningId, SumcheckId,
        VerifierOpeningAccumulator,
    },
    relation::{
        prover::{prove_relation, RelationWitness},
        verify_relation, RelationSetup,
    },
};
use atlas_transcript::{Blake2bProver, Blake2bVerifier, ProvingTranscript, VerifyingTranscript};
use common::CommittedPoly;
use std::collections::BTreeMap;

#[test]
fn booleanity_checks_sparse_coefficients_and_opens_the_original_domain() {
    let ids = [CommittedPoly::GatherRa(0), CommittedPoly::GatherRa(1)];
    let witnesses = [
        OneHotPolynomial::from_indices(vec![Some(1), None, Some(3), Some(1)], 4),
        OneHotPolynomial::from_indices(vec![None, Some(0), Some(1), Some(2)], 4),
    ];
    let check = CoefficientCheck::one_hot(ids.to_vec(), 2, 2).unwrap();
    let polynomials = ids
        .into_iter()
        .zip(witnesses.iter().cloned().map(MultilinearPolynomial::OneHot))
        .collect();
    let openings = roundtrip(check, &polynomials, |check, setup| {
        BooleanityProver::new(check, setup, &witnesses.each_ref()).unwrap()
    });
    for (id, poly) in ids.into_iter().zip(witnesses) {
        let (point, claim) = openings
            .get_committed_polynomial_opening(OpeningId::new(id, SumcheckId::AkitaBooleanity))
            .unwrap();
        assert_eq!(point.r.len(), 4);
        assert_eq!(claim, poly.evaluate(&point.r));
    }
}

#[test]
fn digit_range_checks_columns_without_a_packed_value_opening() {
    let ids = [
        CommittedPoly::GlobalRemainderDigit(0, 0),
        CommittedPoly::GlobalRemainderDigit(0, 1),
    ];
    let witnesses = [
        MultilinearPolynomial::from(vec![0_u8, 1, 2, 3, 1, 3, 0, 2]),
        MultilinearPolynomial::from(vec![1_u8, 2, 0, 3, 0, 0, 1, 2]),
    ];
    let check = CoefficientCheck::base_four(ids.to_vec(), 3).unwrap();
    let polynomials = ids.into_iter().zip(witnesses.iter().cloned()).collect();
    let openings = roundtrip(check, &polynomials, |check, setup| {
        DigitRangeProver::new(check, setup, witnesses.to_vec()).unwrap()
    });
    for (id, poly) in ids.into_iter().zip(witnesses) {
        let (point, claim) = openings
            .get_committed_polynomial_opening(OpeningId::new(id, SumcheckId::AkitaDigitRange))
            .unwrap();
        assert_eq!(point.r.len(), 3);
        assert_eq!(claim, poly.evaluate(&point.r));
    }
}

fn roundtrip<W: RelationWitness<AkitaField>>(
    check: CoefficientCheck,
    polynomials: &BTreeMap<CommittedPoly, MultilinearPolynomial<AkitaField>>,
    make_witness: impl FnOnce(CoefficientCheck, RelationSetup<AkitaField>) -> W,
) -> VerifierOpeningAccumulator<AkitaField> {
    let mut prover = Blake2bProver::new(b"akita-coefficient-test");
    let mut prover_openings = ProverOpeningAccumulator::new();
    let mut witness = make_witness(check.clone(), check.draw(&mut prover));
    prove_relation(&mut witness, &mut prover_openings, &mut prover);
    let padding = BTreeMap::new();
    prover_openings
        .prepare_for_akita_sumcheck(polynomials, &padding, &mut prover)
        .unwrap();
    let point = prover_openings.prove_batch_opening_sumcheck(&mut prover);
    prover_openings.finalize_batch_opening_sumcheck(point, &mut prover);

    let mut verifier = Blake2bVerifier::new(b"akita-coefficient-test", prover.into_narg());
    let mut verifier_openings = VerifierOpeningAccumulator::new();
    let setup = check.draw(&mut verifier);
    verify_relation(
        &check.into_verifier(setup),
        &mut verifier_openings,
        &mut verifier,
    )
    .unwrap();
    assert_eq!(prover_openings.openings, *verifier_openings.openings());
    verifier_openings
        .prepare_for_akita_sumcheck(&padding, &mut verifier)
        .unwrap();
    let point = verifier_openings
        .verify_batch_opening_sumcheck(&mut verifier)
        .unwrap();
    let state = verifier_openings
        .finalize_batch_opening_sumcheck(point, &mut verifier)
        .unwrap();
    let expected: AkitaField = state
        .polynomials
        .iter()
        .zip(&state.poly_coeffs)
        .map(|(id, &coeff)| coeff * polynomials[id].evaluate(&state.r_sumcheck))
        .sum();
    assert_eq!(verifier_openings.compute_joint_claim(&state), expected);
    verifier.finish().unwrap();
    verifier_openings
}
