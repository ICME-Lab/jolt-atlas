use super::*;
use ark_bn254::{Bn254, Fq, Fr, G1Affine, G2Affine};
use ark_ec::{pairing::Pairing, AffineRepr};
use ark_ff::{BigInteger, PrimeField, UniformRand};

fn wrapped(x: Fq12) -> CompactDoryCommitment {
    CompactDoryCommitment(DoryCommitment(ArkGT(PairingOutput(x))))
}

fn original_check(x: Fq12) -> bool {
    wrapped(x).0.check().is_ok()
}

fn parity(x: Fq12) {
    assert_eq!(wrapped(x).check().is_ok(), original_check(x));
    assert_eq!(original_check(x), x.pow(Fr::MODULUS) == Fq12::ONE);
}

#[test]
fn shorter_exponent_matches_bn254_parameters() {
    let mut difference = Fq::MODULUS;
    assert!(!difference.sub_with_borrow(&Fr::MODULUS));
    assert_eq!(difference.0, [P_MINUS_R[0], P_MINUS_R[1], 0, 0]);
    assert_eq!(difference.num_bits(), 127);
    let mut rng = ark_std::test_rng();
    for _ in 0..16 {
        let x = Fq12::rand(&mut rng);
        assert_eq!(x.frobenius_map(1), x.pow(Fq::MODULUS));
    }
}

#[test]
fn shorter_subgroup_check_matches_original_on_extension_field_and_gt() {
    let mut rng = ark_std::test_rng();
    let base = Bn254::pairing(G1Affine::generator(), G2Affine::generator()).0;
    for x in [Fq12::ZERO, Fq12::ONE, -Fq12::ONE] {
        parity(x);
    }
    assert!(wrapped(Fq12::ZERO).check().is_err());
    assert!(wrapped(-Fq12::ONE).check().is_err());
    assert!(wrapped(Fq12::ONE).check().is_ok());
    for _ in 0..128 {
        parity(Fq12::rand(&mut rng));
        let gt = base.pow(Fr::rand(&mut rng).into_bigint());
        assert!(wrapped(gt).check().is_ok());
        parity(gt);
        parity(gt.inverse().unwrap());
        // Multiplication by the order-two point leaves the GT subgroup.
        assert!(wrapped(-gt).check().is_err());
        parity(-gt);
    }
}

#[test]
fn shorter_subgroup_check_retains_checked_torus_decoding() {
    let mut rng = ark_std::test_rng();
    let mut rejected = 0;
    for t in std::iter::once(Fq6::ZERO).chain((0..128).map(|_| Fq6::rand(&mut rng))) {
        let mut encoded = Vec::new();
        t.serialize_compressed(&mut encoded).unwrap();
        let unchecked =
            CompactDoryCommitment::deserialize_compressed_unchecked(encoded.as_slice()).unwrap();
        let x = unchecked.0 .0 .0 .0;
        assert_eq!(x.c0.square() - times_nonresidue(x.c1.square()), Fq6::ONE);
        let accepted = original_check(x);
        assert_eq!(
            CompactDoryCommitment::deserialize_compressed(encoded.as_slice()).is_ok(),
            accepted
        );
        rejected += usize::from(!accepted);
        parity(x);
    }
    assert!(rejected > 100);
    // Field canonicality is required before the subgroup predicate is evaluated.
    let mut noncanonical = vec![0u8; 192];
    let modulus = Fq::MODULUS.to_bytes_le();
    noncanonical[..modulus.len()].copy_from_slice(&modulus);
    assert!(CompactDoryCommitment::deserialize_compressed(noncanonical.as_slice()).is_err());
    assert!(CompactDoryCommitment::deserialize_compressed([0u8; 191].as_slice()).is_err());
}

#[test]
fn shorter_subgroup_check_preserves_canonical_bytes_and_batch_validation() {
    let mut rng = ark_std::test_rng();
    let base = Bn254::pairing(G1Affine::generator(), G2Affine::generator()).0;
    let points: Vec<_> = (0..64)
        .map(|_| wrapped(base.pow(Fr::rand(&mut rng).into_bigint())))
        .collect();
    CompactDoryCommitment::batch_check(points.iter()).unwrap();
    for point in &points {
        let mut compact = Vec::new();
        point.serialize_compressed(&mut compact).unwrap();
        let restored = CompactDoryCommitment::deserialize_compressed(compact.as_slice()).unwrap();
        assert_eq!(*point, restored);
        let mut before = Vec::new();
        let mut after = Vec::new();
        point.0.serialize_compressed(&mut before).unwrap();
        restored.0.serialize_compressed(&mut after).unwrap();
        assert_eq!(before, after);
        after.clear();
        restored.serialize_compressed(&mut after).unwrap();
        assert_eq!(compact, after);
    }
    for invalid in [Fq12::ZERO, -Fq12::ONE] {
        let mut bad = points.clone();
        bad[points.len() / 2] = wrapped(invalid);
        assert!(CompactDoryCommitment::batch_check(bad.iter()).is_err());
    }
}

#[test]
#[ignore = "isolated subgroup validation comparison"]
fn benchmark_compact_subgroup_check() {
    use std::{hint::black_box, time::Instant};
    let base = Bn254::pairing(G1Affine::generator(), G2Affine::generator()).0;
    let points: Vec<_> = (0..128u64).map(|i| wrapped(base.pow([i]))).collect();
    for original in [true, false, false, true] {
        let start = Instant::now();
        let mut count = 0;
        for _ in 0..16 {
            for point in &points {
                let point = black_box(point);
                count += usize::from(
                    black_box(if original {
                        point.0.check()
                    } else {
                        point.check()
                    })
                    .is_ok(),
                );
            }
        }
        let seconds = start.elapsed().as_secs_f64();
        assert_eq!(count, 2048);
        println!("SUBGROUP_BENCH original={original} checks={count} seconds={seconds:.9}");
    }
}
