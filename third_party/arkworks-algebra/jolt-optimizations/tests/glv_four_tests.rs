use ark_bn254::{Fr, G2Affine, G2Projective};
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use ark_ff::{PrimeField, UniformRand};
use ark_std::{rand::RngCore, test_rng, Zero};

use jolt_optimizations::{
    glv_four_precompute, glv_four_precompute_windowed2_signed, glv_four_scalar_mul,
    glv_four_scalar_mul_online, glv_four_scalar_mul_windowed2_signed,
};

#[test]
fn test_glv_four_consistency() {
    let mut rng = test_rng();

    // Generate test data
    let num_points = 10;
    let points: Vec<G2Projective> = (0..num_points)
        .map(|_| G2Affine::rand(&mut rng).into_group())
        .collect();
    let scalar = Fr::rand(&mut rng);

    // Test online version
    let result_online = glv_four_scalar_mul_online(scalar, &points);

    // Test precomputed full Shamir version
    let precomputed_data = glv_four_precompute(&points);
    let result_precomputed = glv_four_scalar_mul(&precomputed_data, scalar);

    // Test 2-bit windowed signed version
    let windowed2_data = glv_four_precompute_windowed2_signed(&points);
    let result_windowed2 = glv_four_scalar_mul_windowed2_signed(&windowed2_data, scalar);

    // Compare with naive scalar multiplication
    for i in 0..num_points {
        let expected = points[i].mul_bigint(scalar.into_bigint());

        // Convert to affine for comparison
        let expected_affine = expected.into_affine();
        let online_affine = result_online[i].into_affine();
        let precomputed_affine = result_precomputed[i].into_affine();
        let windowed2_affine = result_windowed2[i].into_affine();

        assert_eq!(
            expected_affine, online_affine,
            "Online version mismatch at index {}",
            i
        );
        assert_eq!(
            expected_affine, precomputed_affine,
            "Precomputed version mismatch at index {}",
            i
        );
        assert_eq!(
            expected_affine, windowed2_affine,
            "2-bit windowed version mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_glv_four_edge_cases() {
    let mut rng = test_rng();

    // Test with single point
    let single_point = vec![G2Affine::rand(&mut rng).into_group()];
    let scalar = Fr::rand(&mut rng);

    let result_online = glv_four_scalar_mul_online(scalar, &single_point);
    let expected = single_point[0].mul_bigint(scalar.into_bigint());
    assert_eq!(result_online[0].into_affine(), expected.into_affine());

    // Test with zero scalar
    let points: Vec<G2Projective> = (0..3)
        .map(|_| G2Affine::rand(&mut rng).into_group())
        .collect();
    let zero_scalar = Fr::from(0u64);

    let result_zero = glv_four_scalar_mul_online(zero_scalar, &points);
    for i in 0..points.len() {
        assert_eq!(
            result_zero[i],
            G2Projective::zero(),
            "Zero scalar should produce identity at index {}",
            i
        );
    }

    // Test with scalar = 1
    let one_scalar = Fr::from(1u64);
    let result_one = glv_four_scalar_mul_online(one_scalar, &points);
    for i in 0..points.len() {
        assert_eq!(
            result_one[i].into_affine(),
            points[i].into_affine(),
            "Scalar 1 should return original point at index {}",
            i
        );
    }
}

#[test]
fn test_glv_four_large_scalars() {
    let mut rng = test_rng();

    // Test with maximum scalar values
    let points: Vec<G2Projective> = (0..5)
        .map(|_| G2Affine::rand(&mut rng).into_group())
        .collect();

    // Generate a large scalar close to the field modulus
    let large_scalar = Fr::from(-1i64); // This is Fr::MODULUS - 1

    // Test all methods with large scalar
    let result_online = glv_four_scalar_mul_online(large_scalar, &points);
    let precomputed_data = glv_four_precompute(&points);
    let result_precomputed = glv_four_scalar_mul(&precomputed_data, large_scalar);
    let windowed2_data = glv_four_precompute_windowed2_signed(&points);
    let result_windowed2 = glv_four_scalar_mul_windowed2_signed(&windowed2_data, large_scalar);

    // Compare with naive
    for i in 0..points.len() {
        let expected = points[i].mul_bigint(large_scalar.into_bigint());
        let expected_affine = expected.into_affine();

        assert_eq!(
            result_online[i].into_affine(),
            expected_affine,
            "Online large scalar mismatch at index {}",
            i
        );
        assert_eq!(
            result_precomputed[i].into_affine(),
            expected_affine,
            "Precomputed large scalar mismatch at index {}",
            i
        );
        assert_eq!(
            result_windowed2[i].into_affine(),
            expected_affine,
            "Windowed2 large scalar mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_glv_four_random_extensive() {
    let mut rng = test_rng();

    // Test with many random points and scalars
    for _ in 0..20 {
        let num_points = 1 + (rng.next_u32() % 10) as usize;
        let points: Vec<G2Projective> = (0..num_points)
            .map(|_| G2Affine::rand(&mut rng).into_group())
            .collect();
        let scalar = Fr::rand(&mut rng);

        // Compute using all methods
        let result_online = glv_four_scalar_mul_online(scalar, &points);
        let precomputed_data = glv_four_precompute(&points);
        let result_precomputed = glv_four_scalar_mul(&precomputed_data, scalar);
        let windowed2_data = glv_four_precompute_windowed2_signed(&points);
        let result_windowed2 = glv_four_scalar_mul_windowed2_signed(&windowed2_data, scalar);

        // Verify all match naive computation
        for i in 0..num_points {
            let expected = points[i].mul_bigint(scalar.into_bigint()).into_affine();

            assert_eq!(
                result_online[i].into_affine(),
                expected,
                "Random test: online mismatch"
            );
            assert_eq!(
                result_precomputed[i].into_affine(),
                expected,
                "Random test: precomputed mismatch"
            );
            assert_eq!(
                result_windowed2[i].into_affine(),
                expected,
                "Random test: windowed2 mismatch"
            );
        }
    }
}

#[test]
fn test_glv_four_precomputation_correctness() {
    let mut rng = test_rng();

    // Test that precomputation produces correct results for multiple scalars
    let points: Vec<G2Projective> = (0..5)
        .map(|_| G2Affine::rand(&mut rng).into_group())
        .collect();

    // Precompute once
    let precomputed_data = glv_four_precompute(&points);
    let windowed2_data = glv_four_precompute_windowed2_signed(&points);

    // Test with multiple different scalars
    for _ in 0..10 {
        let scalar = Fr::rand(&mut rng);

        let result_precomputed = glv_four_scalar_mul(&precomputed_data, scalar);
        let result_windowed2 = glv_four_scalar_mul_windowed2_signed(&windowed2_data, scalar);
        let result_online = glv_four_scalar_mul_online(scalar, &points);

        // All should match
        for i in 0..points.len() {
            let online_affine = result_online[i].into_affine();
            assert_eq!(
                result_precomputed[i].into_affine(),
                online_affine,
                "Precomputed doesn't match online"
            );
            assert_eq!(
                result_windowed2[i].into_affine(),
                online_affine,
                "Windowed2 doesn't match online"
            );
        }
    }
}
