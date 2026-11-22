use super::*;
use atlas_algebra::{
    field::Ring,
    poly::{multilinear::PolynomialEvaluation, one_hot::OneHotPolynomial as AtlasOneHot},
};
use jolt_akita::{AkitaScheduleArtifacts, AkitaSetupParams};
use jolt_openings::CommitmentScheme;
use jolt_transcript::{protocol_tag, prover_transcript, verifier_transcript, Keccak};
use std::sync::Arc;

fn artifacts() -> Arc<AkitaScheduleArtifacts> {
    schedule_artifacts().unwrap()
}

fn roundtrip(
    params: AkitaSetupParams,
    layout_digest: [u8; 32],
    columns: &[&MultilinearPolynomial<AkitaField>],
) -> AkitaBackendFlavor {
    let (prover_setup, verifier_setup) = AkitaScheme::setup(params).unwrap();
    let group = Group::new(columns).unwrap();
    let point: Vec<_> = (0..columns[0].get_num_vars())
        .map(|index| AkitaField::from_u64(3 * index as u64 + 2))
        .collect();
    let native_point = group.layout().point(&point).unwrap();
    let evaluations: Vec<_> = columns.iter().map(|poly| poly.evaluate(&point)).collect();
    for (poly, &value) in group.polynomials().iter().zip(&evaluations) {
        assert_eq!(poly.evaluate(&native_point), value.into_native());
    }

    let (commitment, hint) = group.commit(&prover_setup, layout_digest).unwrap();
    let session = protocol_tag(b"atlas-akita-polynomial-test");
    let mut prover = prover_transcript::<Keccak>(&session, &layout_digest);
    let proof = group
        .prove_opening(
            &prover_setup,
            &commitment,
            hint,
            &point,
            &evaluations,
            &mut prover,
        )
        .unwrap();
    let mut verifier = verifier_transcript::<Keccak>(&session, &layout_digest, &[]);
    verify_opening(
        &verifier_setup,
        group.layout(),
        &commitment,
        &point,
        &evaluations,
        &proof,
        &mut verifier,
    )
    .unwrap();
    verifier.check_eof().unwrap();
    commitment.backend_flavor()
}

#[test]
fn sparse_one_hot_roundtrips_at_atlas_point() {
    let digest = [3; 32];
    let make_column = |offset: u16| {
        MultilinearPolynomial::OneHot(AtlasOneHot::from_indices(
            (0..256)
                .map(|row| (row % 7 != 0).then_some((row * 5 + offset) % 16))
                .collect(),
            16,
        ))
    };
    assert_eq!(
        roundtrip(
            AkitaSetupParams::one_hot_only(12, 2, digest, 16, artifacts()),
            digest,
            &[&make_column(0), &make_column(3)],
        ),
        AkitaBackendFlavor::OneHot
    );
}

#[test]
fn dense_roundtrips_signed_coefficients_and_wide_gather() {
    let digest = [4; 32];
    let gather = MultilinearPolynomial::OneHot(AtlasOneHot::from_indices(
        (0..16).map(|i| (i % 5 != 0).then_some(i * 61)).collect(),
        1024,
    ));
    let dense =
        MultilinearPolynomial::from((0..1 << 14).map(|i| i % 13 - 6_i32).collect::<Vec<_>>());
    assert_eq!(
        roundtrip(
            AkitaSetupParams::dense_only(14, 2, digest, artifacts()),
            digest,
            &[&gather, &dense],
        ),
        AkitaBackendFlavor::Dense
    );
}

#[test]
fn atlas_dense_group_schedules_roundtrip() {
    for (num_vars, count) in [
        (10, 4),
        (10, 8),
        (11, 1),
        (11, 2),
        (11, 4),
        (11, 8),
        (13, 8),
        (14, 8),
        (15, 16),
        (16, 8),
        (18, 8),
    ] {
        let digest = [num_vars as u8; 32];
        let polys: Vec<_> = (0..count)
            .map(|offset| {
                MultilinearPolynomial::from(
                    (0..1_usize << num_vars)
                        .map(|index| ((index + offset) % 29) as i32 - 14)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        assert_eq!(
            roundtrip(
                AkitaSetupParams::dense_only(num_vars, count, digest, artifacts()),
                digest,
                &polys.iter().collect::<Vec<_>>(),
            ),
            AkitaBackendFlavor::Dense,
            "dense group with {num_vars} variables and {count} polynomials",
        );
    }
}

#[test]
fn atlas_one_hot_group_schedules_roundtrip() {
    for (k, rows, count) in [(16_usize, 1024_usize, 16), (16, 4096, 32), (256, 256, 8)] {
        let num_vars = k.ilog2() as usize + rows.ilog2() as usize;
        let digest = [count as u8; 32];
        let polys: Vec<_> = (0..count)
            .map(|offset| {
                MultilinearPolynomial::OneHot(AtlasOneHot::from_indices(
                    (0..rows)
                        .map(|row| (row % 7 != 0).then_some(((row * 5 + offset) % k) as u16))
                        .collect(),
                    k,
                ))
            })
            .collect();
        assert_eq!(
            roundtrip(
                AkitaSetupParams::one_hot_only(num_vars, count, digest, k, artifacts()),
                digest,
                &polys.iter().collect::<Vec<_>>(),
            ),
            AkitaBackendFlavor::OneHot,
            "one-hot group with K={k}, {rows} rows, and {count} polynomials",
        );
    }
}

#[test]
fn atlas_field_accumulation_and_encoding_use_the_akita_modulus() {
    use atlas_algebra::{
        field::{Accumulator, JoltField, WithAccumulator},
        serialize::Canonical,
    };

    let a = AkitaField::from_u128(u128::MAX);
    assert_eq!(a.into_native().to_canonical_u128(), Native::C - 1);
    let b = AkitaField::from_i64(-7);
    let expected = -a.into_native() * AkitaField::from_u64(7).into_native();
    assert_eq!(
        AkitaField::reduce_product(a.mul_to_product(b)).into_native(),
        expected
    );
    let mut acc = <AkitaField as WithAccumulator>::Accumulator::default();
    acc.fmadd(a, b);
    assert_eq!(acc.reduce().into_native(), expected);

    let mut bytes = Vec::new();
    a.encode(&mut bytes).unwrap();
    assert_eq!(AkitaField::decode(bytes.as_slice()).unwrap(), a);
    assert!(AkitaField::decode(u128::MAX.to_le_bytes().as_slice()).is_err());
}

#[test]
fn padded_one_hot_uses_one_point_after_the_atlas_batch_sumcheck() {
    use atlas_sumcheck::opening_proof::{
        prover::ProverOpeningAccumulator, AkitaOpeningPadding, OpeningAccumulator, OpeningId,
        SumcheckId, VerifierOpeningAccumulator,
    };
    use atlas_transcript::{
        Blake2bProver, Blake2bVerifier, ProvingTranscript, VerifyingTranscript,
    };
    use common::CommittedPoly;
    use std::collections::BTreeMap;

    let digest = [9; 32];
    let ids = [CommittedPoly::GatherRa(0), CommittedPoly::GatherRa(1)];
    let original = [512, 1024].map(|rows| {
        MultilinearPolynomial::OneHot(AtlasOneHot::from_indices(
            (0..rows).map(|row| Some((row * 3) % 16)).collect(),
            16,
        ))
    });
    let layouts = original
        .each_ref()
        .map(|poly| GroupLayout::for_polynomial(poly).unwrap());
    let catalog = artifacts();
    let plan = plan_groups_with_padding(&layouts, &catalog).unwrap();
    let [planned] = plan.as_slice() else {
        panic!("expected one mixed-length group");
    };
    assert_eq!(planned.polynomial_indices, [0, 1]);
    let padded = original
        .each_ref()
        .map(|poly| planned.layout.pad_polynomial(poly).unwrap());
    let witness: BTreeMap<_, _> = ids
        .into_iter()
        .zip(padded.map(|poly| poly.into_owned()))
        .collect();
    let group = Group::new(&witness.values().collect::<Vec<_>>()).unwrap();
    let params = AkitaSetupParams::one_hot_only(14, 2, digest, 16, catalog);
    let (pk, vk) = AkitaScheme::setup(params).unwrap();
    let (commitment, hint) = group.commit(&pk, digest).unwrap();

    let padding = BTreeMap::from([(ids[0], AkitaOpeningPadding::new(Some(4), 13, 14).unwrap())]);
    let mut prover = ProverOpeningAccumulator::new();
    let mut proof_transcript = Blake2bProver::new(b"padded-akita-reduction");
    let points = layouts.map(|layout| {
        (0..layout.num_vars())
            .map(|i| AkitaField::from_u64(i as u64 + 3))
            .collect::<Vec<_>>()
    });
    for ((id, poly), point) in ids.into_iter().zip(&original).zip(&points) {
        prover.append_sparse(
            &mut proof_transcript,
            vec![id],
            SumcheckId::NodeExecution(0),
            point[..4].to_vec(),
            point[4..].to_vec(),
            vec![poly.evaluate(point)],
        );
    }
    prover
        .prepare_for_akita_sumcheck(&witness, &padding, &mut proof_transcript)
        .unwrap();
    let challenge = prover.prove_batch_opening_sumcheck(&mut proof_transcript);
    prover.finalize_batch_opening_sumcheck(challenge, &mut proof_transcript);

    let mut verifier = VerifierOpeningAccumulator::new();
    let mut transcript =
        Blake2bVerifier::new(b"padded-akita-reduction", proof_transcript.into_narg());
    for (id, point) in ids.into_iter().zip(points) {
        verifier
            .append_group(
                &mut transcript,
                vec![id],
                SumcheckId::NodeExecution(0),
                point.clone(),
            )
            .unwrap();
        let (registered, _) = verifier
            .get_committed_polynomial_opening(OpeningId::new(id, SumcheckId::NodeExecution(0)))
            .unwrap();
        assert_eq!(registered.r, point);
    }
    verifier
        .prepare_for_akita_sumcheck(&padding, &mut transcript)
        .unwrap();
    let challenge = verifier
        .verify_batch_opening_sumcheck(&mut transcript)
        .unwrap();
    let state = verifier
        .finalize_batch_opening_sumcheck(challenge, &mut transcript)
        .unwrap();
    assert_eq!(verifier.opened_num_vars(&state), [14, 14]);
    let point = state.r_sumcheck.as_slice();
    let evaluations: Vec<_> = witness.values().map(|poly| poly.evaluate(point)).collect();
    let joint: AkitaField = evaluations
        .iter()
        .zip(&state.poly_coeffs)
        .map(|(&eval, &coeff)| eval * coeff)
        .sum();
    assert_eq!(joint, verifier.compute_joint_claim(&state));

    let session = protocol_tag(b"padded-akita-opening");
    let mut opening_prover = prover_transcript::<Keccak>(&session, &digest);
    let proof = group
        .prove_opening(
            &pk,
            &commitment,
            hint,
            point,
            &evaluations,
            &mut opening_prover,
        )
        .unwrap();
    let mut opening_verifier = verifier_transcript::<Keccak>(&session, &digest, &[]);
    verify_opening(
        &vk,
        group.layout(),
        &commitment,
        point,
        &evaluations,
        &proof,
        &mut opening_verifier,
    )
    .unwrap();
    opening_verifier.check_eof().unwrap();
}
