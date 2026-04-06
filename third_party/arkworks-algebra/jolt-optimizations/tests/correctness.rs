use ark_bn254::{Fr, G2Affine, G2Projective};
use ark_ec::AffineRepr;
use ark_ec::CurveGroup;
use ark_ec::PrimeGroup;
use ark_ff::{PrimeField, UniformRand};
use ark_std::test_rng;
use jolt_optimizations::{
    glv_four_precompute, glv_four_precompute_windowed2_signed, glv_four_scalar_mul,
    glv_four_scalar_mul_online, glv_four_scalar_mul_windowed2_signed,
};

/// Helper function to perform naive scalar multiplication
fn naive_scalar_mul(points: &[G2Projective], scalar: Fr) -> Vec<G2Projective> {
    points
        .iter()
        .map(|point| point.mul_bigint(scalar.into_bigint()))
        .collect()
}

/// Helper function to check if two G2Projective points are equal
fn points_equal(a: &G2Projective, b: &G2Projective) -> bool {
    // Convert to affine to avoid issues with different projective representations
    let a_affine = a.into_affine();
    let b_affine = b.into_affine();
    a_affine == b_affine
}

/// Test that all GLV methods produce the same result as naive scalar multiplication
#[test]
fn test_all_methods_correctness() {
    let mut rng = test_rng();

    // Test with multiple different test cases
    const NUM_POINTS: usize = 100;
    const NUM_SCALARS: usize = 100;

    for test_case in 0..NUM_SCALARS {
        println!("Testing case {}/{}", test_case + 1, NUM_SCALARS);

        // Generate random test data
        let points: Vec<G2Projective> = (0..NUM_POINTS)
            .map(|_| G2Affine::rand(&mut rng).into_group())
            .collect();

        let scalar = Fr::rand(&mut rng);

        // Compute naive result as ground truth
        let naive_results = naive_scalar_mul(&points, scalar);

        // Test online method (no precomputation)
        let online_results = glv_four_scalar_mul_online(scalar, &points);
        assert_eq!(
            online_results.len(),
            naive_results.len(),
            "Online method: wrong number of results"
        );
        for (i, (naive, glv)) in naive_results.iter().zip(online_results.iter()).enumerate() {
            assert!(
                points_equal(naive, glv),
                "Online method: Point {} mismatch. Naive: {:?}, GLV: {:?}",
                i,
                naive,
                glv
            );
        }
        println!("  ✓ Online method passed");

        // Test precomputed methods
        let precomputed_data = glv_four_precompute(&points);
        let precomputed_results = glv_four_scalar_mul(&precomputed_data, scalar);
        assert_eq!(
            precomputed_results.len(),
            naive_results.len(),
            "Precomputed method: wrong number of results"
        );
        for (i, (naive, glv)) in naive_results
            .iter()
            .zip(precomputed_results.iter())
            .enumerate()
        {
            assert!(
                points_equal(naive, glv),
                "Precomputed method: Point {} mismatch. Naive: {:?}, GLV: {:?}",
                i,
                naive,
                glv
            );
        }
        println!("  ✓ Precomputed method passed");

        // Test 2-bit signed method (memory champion)
        let windowed2_signed_data = glv_four_precompute_windowed2_signed(&points);
        let windowed2_signed_results =
            glv_four_scalar_mul_windowed2_signed(&windowed2_signed_data, scalar);
        assert_eq!(
            windowed2_signed_results.len(),
            naive_results.len(),
            "Windowed2 signed method: wrong number of results"
        );
        for (i, (naive, glv)) in naive_results
            .iter()
            .zip(windowed2_signed_results.iter())
            .enumerate()
        {
            assert!(
                points_equal(naive, glv),
                "Windowed2 signed method: Point {} mismatch. Naive: {:?}, GLV: {:?}",
                i,
                naive,
                glv
            );
        }
        println!("  ✓ Windowed2 signed method passed");
    }

    println!("\n🎉 All methods passed correctness tests!");
}

/// Test edge cases with special scalar values
#[test]
fn test_edge_cases() {
    let mut rng = test_rng();

    // Generate test points
    let points: Vec<G2Projective> = (0..5)
        .map(|_| G2Affine::rand(&mut rng).into_group())
        .collect();

    // Test edge case scalars
    let edge_scalars = vec![
        Fr::from(0u64),     // Zero scalar
        Fr::from(1u64),     // One
        Fr::from(2u64),     // Small scalar
        Fr::from(255u64),   // Byte boundary
        Fr::from(65537u64), // Larger scalar
        -Fr::from(1u64),    // Negative scalar
    ];

    for (i, &scalar) in edge_scalars.iter().enumerate() {
        println!("Testing edge case {}: scalar = {:?}", i, scalar);

        // Compute naive result as ground truth
        let naive_results = naive_scalar_mul(&points, scalar);

        // Test all methods against naive
        let online_results = glv_four_scalar_mul_online(scalar, &points);
        for (naive, glv) in naive_results.iter().zip(online_results.iter()) {
            assert!(
                points_equal(naive, glv),
                "Edge case failed for online method with scalar {:?}",
                scalar
            );
        }

        let precomputed_data = glv_four_precompute(&points);
        let precomputed_results = glv_four_scalar_mul(&precomputed_data, scalar);
        for (naive, glv) in naive_results.iter().zip(precomputed_results.iter()) {
            assert!(
                points_equal(naive, glv),
                "Edge case failed for precomputed method with scalar {:?}",
                scalar
            );
        }

        let windowed2_signed_data = glv_four_precompute_windowed2_signed(&points);
        let windowed2_signed_results =
            glv_four_scalar_mul_windowed2_signed(&windowed2_signed_data, scalar);
        for (naive, glv) in naive_results.iter().zip(windowed2_signed_results.iter()) {
            assert!(
                points_equal(naive, glv),
                "Edge case failed for windowed2 signed method with scalar {:?}",
                scalar
            );
        }
    }

    println!("✓ All edge cases passed!");
}

/// Test with random large scalars to ensure robustness
#[test]
fn test_large_scalars() {
    let mut rng = test_rng();

    // Generate test points
    let points: Vec<G2Projective> = (0..100)
        .map(|_| G2Affine::rand(&mut rng).into_group())
        .collect();

    // Test with 10 random large scalars
    for i in 0..100 {
        let scalar = Fr::rand(&mut rng); // This generates large random scalars

        println!("Testing large scalar {}/10", i + 1);

        // Compute naive result as ground truth
        let naive_results = naive_scalar_mul(&points, scalar);

        // Test all methods
        let online_results = glv_four_scalar_mul_online(scalar, &points);
        for (j, (naive, glv)) in naive_results.iter().zip(online_results.iter()).enumerate() {
            assert!(
                points_equal(naive, glv),
                "Large scalar test failed for online method: point {}, scalar {:?}",
                j,
                scalar
            );
        }

        let precomputed_data = glv_four_precompute(&points);
        let precomputed_results = glv_four_scalar_mul(&precomputed_data, scalar);
        for (j, (naive, glv)) in naive_results
            .iter()
            .zip(precomputed_results.iter())
            .enumerate()
        {
            assert!(
                points_equal(naive, glv),
                "Large scalar test failed for precomputed method: point {}, scalar {:?}",
                j,
                scalar
            );
        }

        let windowed2_signed_data = glv_four_precompute_windowed2_signed(&points);
        let windowed2_signed_results =
            glv_four_scalar_mul_windowed2_signed(&windowed2_signed_data, scalar);
        for (j, (naive, glv)) in naive_results
            .iter()
            .zip(windowed2_signed_results.iter())
            .enumerate()
        {
            assert!(
                points_equal(naive, glv),
                "Large scalar test failed for windowed2 signed method: point {}, scalar {:?}",
                j,
                scalar
            );
        }
    }

    println!("✓ All large scalar tests passed!");
}
