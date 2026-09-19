use super::*;
use ark_bn254::{Bn254, Fr, G1Affine, G2Affine};
use ark_ec::{pairing::Pairing, AffineRepr};
use ark_ff::{PrimeField, UniformRand};

// Exact Phi_12(p) = p^4 - p^2 + 1 for the pinned BN254 base field.
const CYCLOTOMIC_ORDER: [u64; 16] = [
    0xc7c4b3abbcdf42b1,
    0xa0a28484ceffba2b,
    0x8ecc662fdf333aef,
    0xe1786f6a4a984cc6,
    0x07acd56bc7bac087,
    0x15dd11b9e27aad6e,
    0xc2c78b0e23f7f3cd,
    0x18f249b337319fea,
    0x7e1b009439ceba33,
    0xca425189b6172413,
    0x4f97cc2276924233,
    0xea401bebaf1b1332,
    0x06f6feb7b4e30336,
    0x562e001117c18136,
    0x94d5ab7ebe19457b,
    0x0053ad676ccd6cff,
];

fn wrapped(x: Fq12) -> CompactDoryCommitment {
    CompactDoryCommitment(DoryCommitment(ArkGT(PairingOutput(x))))
}

fn gate(x: Fq12) -> bool {
    x != Fq12::ZERO && x.frobenius_map(4) * x == x.frobenius_map(2)
}

fn ordinary_check(x: Fq12) -> bool {
    x != Fq12::ZERO && x.frobenius_map(1) == x.pow(P_MINUS_R)
}

fn parity(x: Fq12) {
    let expected = wrapped(x).0.check().is_ok();
    assert_eq!(ordinary_check(x), expected);
    assert_eq!(wrapped(x).check().is_ok(), expected);
}

fn project(x: Fq12) -> Fq12 {
    // x^((p^6 - 1)(p^2 + 1)), using only ordinary field operations.
    let norm_one = x.frobenius_map(6) * x.inverse().unwrap();
    norm_one.frobenius_map(2) * norm_one
}

#[test]
fn cyclotomic_gate_matches_full_order_relation() {
    let mut rng = ark_std::test_rng();
    for x in [Fq12::ZERO, Fq12::ONE, -Fq12::ONE] {
        assert_eq!(gate(x), x.pow(CYCLOTOMIC_ORDER) == Fq12::ONE);
        parity(x);
    }
    for _ in 0..32 {
        let x = Fq12::rand(&mut rng);
        assert_ne!(x, Fq12::ZERO);
        let norm_one = x.frobenius_map(6) * x.inverse().unwrap();
        let cyclotomic = project(x);
        for value in [x, norm_one, cyclotomic, -cyclotomic] {
            assert_eq!(gate(value), value.pow(CYCLOTOMIC_ORDER) == Fq12::ONE);
            parity(value);
        }
        assert!(gate(cyclotomic));
        assert!(!gate(-cyclotomic));
        assert_eq!(cyclotomic.cyclotomic_square(), cyclotomic.square());
        assert_eq!(
            cyclotomic.cyclotomic_exp(P_MINUS_R),
            cyclotomic.pow(P_MINUS_R)
        );
    }
}

#[test]
fn cyclotomic_membership_alone_does_not_accept_points_outside_gt() {
    let mut rng = ark_std::test_rng();
    let mut outside_gt = 0;
    let base = Bn254::pairing(G1Affine::generator(), G2Affine::generator()).0;
    for _ in 0..64 {
        let cyclotomic = project(Fq12::rand(&mut rng));
        assert!(gate(cyclotomic));
        parity(cyclotomic);
        if wrapped(cyclotomic).0.check().is_err() {
            outside_gt += 1;
            assert!(wrapped(cyclotomic).check().is_err());
            assert!(CompactDoryCommitment::batch_check([wrapped(cyclotomic)].iter()).is_err());
        }
        let gt = base.pow(Fr::rand(&mut rng).into_bigint());
        assert!(gate(gt));
        assert!(wrapped(gt).check().is_ok());
        assert!(wrapped(gt.inverse().unwrap()).check().is_ok());
        parity(gt);
        parity(gt.inverse().unwrap());
        parity(-gt);
    }
    assert_eq!(outside_gt, 64);
}

#[test]
fn cyclotomic_checked_decoder_matches_both_reference_predicates() {
    let mut rng = ark_std::test_rng();
    let mut rejected = 0;
    for t in std::iter::once(Fq6::ZERO).chain((0..128).map(|_| Fq6::rand(&mut rng))) {
        let mut bytes = Vec::new();
        t.serialize_compressed(&mut bytes).unwrap();
        let point =
            CompactDoryCommitment::deserialize_compressed_unchecked(bytes.as_slice()).unwrap();
        let x = point.0 .0 .0 .0;
        parity(x);
        let expected = point.0.check().is_ok();
        assert_eq!(
            CompactDoryCommitment::deserialize_compressed(bytes.as_slice()).is_ok(),
            expected
        );
        rejected += usize::from(!expected);
    }
    assert_eq!(rejected, 128);
}

#[test]
fn cyclotomic_gate_retains_canonical_roundtrips_and_batch_rejection() {
    let base = Bn254::pairing(G1Affine::generator(), G2Affine::generator()).0;
    let points: Vec<_> = (0..32u64).map(|i| wrapped(base.pow([i]))).collect();
    CompactDoryCommitment::batch_check(points.iter()).unwrap();
    for point in &points {
        let mut bytes = Vec::new();
        point.serialize_compressed(&mut bytes).unwrap();
        assert_eq!(
            CompactDoryCommitment::deserialize_compressed(bytes.as_slice()).unwrap(),
            *point
        );
        let mut original = Vec::new();
        point.0.serialize_compressed(&mut original).unwrap();
        let restored = CompactDoryCommitment::deserialize_compressed(bytes.as_slice()).unwrap();
        let mut after = Vec::new();
        restored.0.serialize_compressed(&mut after).unwrap();
        assert_eq!(original, after);
    }
    let mut rng = ark_std::test_rng();
    let outside = project(Fq12::rand(&mut rng));
    assert!(gate(outside));
    assert!(!ordinary_check(outside));
    for invalid in [Fq12::ZERO, -Fq12::ONE, outside] {
        let mut bad = points.clone();
        bad[16] = wrapped(invalid);
        assert!(CompactDoryCommitment::batch_check(bad.iter()).is_err());
    }
}

#[test]
#[ignore = "isolated gated cyclotomic comparison"]
fn benchmark_gated_cyclotomic_check() {
    use std::{hint::black_box, time::Instant};
    let base = Bn254::pairing(G1Affine::generator(), G2Affine::generator()).0;
    let points: Vec<_> = (0..128u64).map(|i| wrapped(base.pow([i]))).collect();
    for ordinary in [true, false, false, true] {
        let start = Instant::now();
        let mut count = 0;
        for _ in 0..16 {
            for point in &points {
                let point = black_box(point);
                count += usize::from(black_box(if ordinary {
                    ordinary_check(point.0 .0 .0 .0)
                } else {
                    point.check().is_ok()
                }));
            }
        }
        let seconds = start.elapsed().as_secs_f64();
        assert_eq!(count, 2048);
        println!("CYCLOTOMIC_BENCH ordinary={ordinary} checks={count} seconds={seconds:.9}");
    }
}
