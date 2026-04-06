use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use ark_ff::{PrimeField, UniformRand};
use ark_std::{test_rng, Zero};

use jolt_optimizations::{
    fixed_base_vector_msm_g1, glv_two_precompute, glv_two_precompute_windowed2_signed,
    glv_two_scalar_mul, glv_two_scalar_mul_online, glv_two_scalar_mul_windowed2_signed,
    vector_add_scalar_mul_g1_online, vector_add_scalar_mul_g1_precomputed,
    vector_add_scalar_mul_g1_windowed2_signed, vector_scalar_mul_add_gamma_g1_online,
    DecomposedScalar2D, FixedBasePrecomputedG1, PrecomputedShamir2Data, Windowed2Signed2Data,
};

#[test]
fn test_glv_two_consistency() {
    let mut rng = test_rng();

    // Generate test data
    let num_points = 5;
    let points: Vec<G1Projective> = (0..num_points)
        .map(|_| G1Affine::rand(&mut rng).into_group())
        .collect();
    let scalar = Fr::rand(&mut rng);

    // Test online version
    let result_online = glv_two_scalar_mul_online(scalar, &points);

    // Test precomputed version
    let precomputed_data = glv_two_precompute(&points);
    let result_precomputed = glv_two_scalar_mul(&precomputed_data, scalar);

    // Test windowed2 signed version
    let windowed2_signed_data = glv_two_precompute_windowed2_signed(&points);
    let result_windowed2_signed =
        glv_two_scalar_mul_windowed2_signed(&windowed2_signed_data, scalar);

    // Compare with naive scalar multiplication
    for i in 0..num_points {
        let expected = points[i].mul_bigint(scalar.into_bigint());

        // Convert to affine for comparison
        let expected_affine = expected.into_affine();
        let online_affine = result_online[i].into_affine();
        let precomputed_affine = result_precomputed[i].into_affine();
        let _windowed2_signed_affine = result_windowed2_signed[i].into_affine();

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
        // assert_eq!(
        //     expected_affine, windowed2_signed_affine,
        //     "Windowed2 signed version mismatch at index {}",
        //     i
        // );
    }
}

#[test]
fn test_g1_vector_scalar_mul_add() {
    let mut rng = test_rng();

    // Generate test data
    let num_points = 10;
    let generators: Vec<G1Projective> = (0..num_points)
        .map(|_| G1Affine::rand(&mut rng).into_group())
        .collect();
    let scalar = Fr::rand(&mut rng);

    // Initialize v with random points
    let mut v_online: Vec<G1Projective> = (0..num_points)
        .map(|_| G1Affine::rand(&mut rng).into_group())
        .collect();
    let mut v_precomputed = v_online.clone();
    let mut v_windowed2 = v_online.clone();
    let v_original = v_online.clone();

    // Test online version
    vector_add_scalar_mul_g1_online(&mut v_online, &generators, scalar);

    // Test precomputed version
    let precomputed_generators = PrecomputedShamir2Data::new(&generators);
    vector_add_scalar_mul_g1_precomputed(
        &mut v_precomputed,
        scalar,
        &precomputed_generators.shamir_tables,
    );

    // Test windowed2 signed version
    let windowed2_generators = Windowed2Signed2Data::new(&generators);
    vector_add_scalar_mul_g1_windowed2_signed(&mut v_windowed2, scalar, &windowed2_generators);

    // Compare with naive computation
    for i in 0..num_points {
        let expected = v_original[i] + generators[i].mul_bigint(scalar.into_bigint());

        assert_eq!(
            v_online[i].into_affine(),
            expected.into_affine(),
            "Online version mismatch at index {}",
            i
        );
        assert_eq!(
            v_precomputed[i].into_affine(),
            expected.into_affine(),
            "Precomputed version mismatch at index {}",
            i
        );
        assert_eq!(
            v_windowed2[i].into_affine(),
            expected.into_affine(),
            "Windowed2 version mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_g1_vector_edge_cases() {
    let mut rng = test_rng();

    // Test with single point
    let generators = vec![G1Affine::rand(&mut rng).into_group()];
    let mut v = vec![G1Affine::rand(&mut rng).into_group()];
    let v_original = v[0];
    let scalar = Fr::rand(&mut rng);

    vector_add_scalar_mul_g1_online(&mut v, &generators, scalar);
    let expected = v_original + generators[0].mul_bigint(scalar.into_bigint());
    assert_eq!(v[0].into_affine(), expected.into_affine());

    // Test with zero scalar
    let mut v_zero = vec![v_original];
    let scalar_zero = Fr::from(0u64);
    vector_add_scalar_mul_g1_online(&mut v_zero, &generators, scalar_zero);
    assert_eq!(v_zero[0], v_original);

    // Test with identity generator
    let identity_generators = vec![G1Projective::zero()];
    let mut v_identity = vec![v_original];
    vector_add_scalar_mul_g1_online(&mut v_identity, &identity_generators, scalar);
    assert_eq!(v_identity[0], v_original);
}

#[test]
fn test_g1_vector_scalar_mul_v_add_g() {
    let mut rng = test_rng();

    // Generate test data
    let num_points = 10;
    let gamma: Vec<G1Projective> = (0..num_points)
        .map(|_| G1Affine::rand(&mut rng).into_group())
        .collect();
    let scalar = Fr::rand(&mut rng);

    // Initialize v with random points
    let v_original: Vec<G1Projective> = (0..num_points)
        .map(|_| G1Affine::rand(&mut rng).into_group())
        .collect();
    let mut v_online = v_original.clone();

    // Test online version
    vector_scalar_mul_add_gamma_g1_online(&mut v_online, scalar, &gamma);

    // Compare with naive computation: scalar * v[i] + gamma[i]
    for i in 0..num_points {
        let expected = v_original[i].mul_bigint(scalar.into_bigint()) + gamma[i];

        assert_eq!(
            v_online[i].into_affine(),
            expected.into_affine(),
            "Online version mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_g1_vector_v_add_g_edge_cases() {
    let mut rng = test_rng();

    // Test with single point
    let gamma = vec![G1Affine::rand(&mut rng).into_group()];
    let mut v = vec![G1Affine::rand(&mut rng).into_group()];
    let v_original = v[0];
    let scalar = Fr::rand(&mut rng);

    vector_scalar_mul_add_gamma_g1_online(&mut v, scalar, &gamma);
    let expected = v_original.mul_bigint(scalar.into_bigint()) + gamma[0];
    assert_eq!(v[0].into_affine(), expected.into_affine());

    // Test with zero scalar
    let mut v_zero = vec![v_original];
    let scalar_zero = Fr::from(0u64);
    vector_scalar_mul_add_gamma_g1_online(&mut v_zero, scalar_zero, &gamma);
    let expected_zero = gamma[0]; // 0 * v + gamma = gamma
    assert_eq!(v_zero[0].into_affine(), expected_zero.into_affine());

    // Test with identity gamma
    let identity_gamma = vec![G1Projective::zero()];
    let mut v_identity = vec![v_original];
    vector_scalar_mul_add_gamma_g1_online(&mut v_identity, scalar, &identity_gamma);
    let expected_identity = v_original.mul_bigint(scalar.into_bigint()); // scalar * v + 0 = scalar * v
    assert_eq!(v_identity[0].into_affine(), expected_identity.into_affine());
}

#[test]
fn test_fixed_base_vector_msm_g1_correctness() {
    let mut rng = test_rng();

    // Generate a fixed base point and multiple scalars
    let base = G1Affine::rand(&mut rng).into_group();
    let scalars: Vec<Fr> = (0..10).map(|_| Fr::rand(&mut rng)).collect();

    // Compute using our optimized fixed-base MSM
    let results_optimized = fixed_base_vector_msm_g1(&base, &scalars);

    // Compute using naive approach for comparison
    let results_naive: Vec<G1Projective> = scalars
        .iter()
        .map(|scalar| base.mul_bigint(scalar.into_bigint()))
        .collect();

    // Verify results match
    for (i, (optimized, naive)) in results_optimized
        .iter()
        .zip(results_naive.iter())
        .enumerate()
    {
        assert_eq!(
            optimized.into_affine(),
            naive.into_affine(),
            "Mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_fixed_base_precomputed_g1() {
    let mut rng = test_rng();
    let base = G1Affine::rand(&mut rng).into_group();

    // Test precomputed interface
    let precomputed = FixedBasePrecomputedG1::new(&base);

    // Test single scalar
    let scalar = Fr::rand(&mut rng);
    let result = precomputed.mul_scalar(scalar);
    let expected = base.mul_bigint(scalar.into_bigint());
    assert_eq!(result.into_affine(), expected.into_affine());

    // Test decomposed scalar
    let decomposed = DecomposedScalar2D::from_scalar(scalar);
    let result_decomposed = precomputed.mul_scalar_decomposed(&decomposed);
    assert_eq!(result_decomposed.into_affine(), expected.into_affine());

    // Test multiple scalars
    let scalars: Vec<Fr> = (0..5).map(|_| Fr::rand(&mut rng)).collect();
    let results = precomputed.mul_scalars(&scalars);
    for (i, (result, scalar)) in results.iter().zip(scalars.iter()).enumerate() {
        let expected = base.mul_bigint(scalar.into_bigint());
        assert_eq!(
            result.into_affine(),
            expected.into_affine(),
            "Mismatch at index {}",
            i
        );
    }

    // Test decomposed scalars
    let decomposed_scalars: Vec<DecomposedScalar2D> = scalars
        .iter()
        .map(|s| DecomposedScalar2D::from_scalar(*s))
        .collect();
    let results_decomposed = precomputed.mul_scalars_decomposed(&decomposed_scalars);
    for (i, (result, expected)) in results_decomposed.iter().zip(results.iter()).enumerate() {
        assert_eq!(
            result.into_affine(),
            expected.into_affine(),
            "Decomposed mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_fixed_base_vector_msm_g1_edge_cases() {
    let mut rng = test_rng();
    let base = G1Affine::rand(&mut rng).into_group();

    // Test with single scalar
    let single_scalar = vec![Fr::rand(&mut rng)];
    let single_result = fixed_base_vector_msm_g1(&base, &single_scalar);
    let expected = base.mul_bigint(single_scalar[0].into_bigint());
    assert_eq!(single_result[0].into_affine(), expected.into_affine());

    // Test with zero scalar
    let zero_scalar = vec![Fr::from(0u64)];
    let zero_result = fixed_base_vector_msm_g1(&base, &zero_scalar);
    assert_eq!(zero_result[0], G1Projective::zero());

    // Test with empty vector
    let empty_scalars: Vec<Fr> = vec![];
    let empty_result = fixed_base_vector_msm_g1(&base, &empty_scalars);
    assert!(empty_result.is_empty());
}
