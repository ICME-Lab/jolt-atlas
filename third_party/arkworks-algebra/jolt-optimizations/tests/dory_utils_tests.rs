use ark_bn254::{Fr, G2Affine, G2Projective};
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use ark_ff::{PrimeField, UniformRand};
use ark_std::{test_rng, Zero};

use jolt_optimizations::{
    vector_scalar_mul_add, vector_scalar_mul_add_online, vector_scalar_mul_add_precomputed,
    vector_scalar_mul_v_add_g_online, vector_scalar_mul_v_add_g_precomputed, VectorScalarMulData,
    VectorScalarMulVData,
};

#[test]
fn test_vector_scalar_mul_add_consistency() {
    let mut rng = test_rng();

    // Generate test data
    let num_generators = 10;
    let generators: Vec<G2Projective> = (0..num_generators)
        .map(|_| G2Affine::rand(&mut rng).into_group())
        .collect();
    let scalar = Fr::rand(&mut rng);

    // Generate initial values
    let initial_values: Vec<G2Projective> = (0..num_generators)
        .map(|_| G2Affine::rand(&mut rng).into_group())
        .collect();

    // Test online version
    let mut v_online = initial_values.clone();
    vector_scalar_mul_add_online(&mut v_online, &generators, scalar);

    // Test precomputed version
    let mut v_precomputed = initial_values.clone();
    vector_scalar_mul_add(&mut v_precomputed, &generators, scalar);

    // Test with separate data creation
    let data = VectorScalarMulData::new(&generators, scalar);
    let mut v_separate = initial_values.clone();
    vector_scalar_mul_add_precomputed(&mut v_separate, &data);

    // Compare with naive computation: v[i] + scalar * generators[i]
    for i in 0..num_generators {
        let expected = initial_values[i] + generators[i].mul_bigint(scalar.into_bigint());

        // Convert to affine for comparison
        let expected_affine = expected.into_affine();
        let online_affine = v_online[i].into_affine();
        let precomputed_affine = v_precomputed[i].into_affine();
        let separate_affine = v_separate[i].into_affine();

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
            expected_affine, separate_affine,
            "Separate data version mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_edge_cases() {
    let mut rng = test_rng();

    // Test with single generator
    let generators = vec![G2Affine::rand(&mut rng).into_group()];
    let scalar = Fr::rand(&mut rng);
    let initial = G2Affine::rand(&mut rng).into_group();

    let mut v = vec![initial];
    vector_scalar_mul_add_online(&mut v, &generators, scalar);

    let expected = initial + generators[0].mul_bigint(scalar.into_bigint());
    assert_eq!(v[0].into_affine(), expected.into_affine());

    // Test with zero scalar (should just keep initial value)
    let scalar_zero = Fr::from(0u64);
    let mut v_zero = vec![initial];
    vector_scalar_mul_add_online(&mut v_zero, &generators, scalar_zero);
    assert_eq!(v_zero[0].into_affine(), initial.into_affine());

    // Test with identity generator (should keep initial value)
    let identity_generators = vec![G2Projective::zero()];
    let mut v_identity = vec![initial];
    vector_scalar_mul_add_online(&mut v_identity, &identity_generators, scalar);
    assert_eq!(v_identity[0].into_affine(), initial.into_affine());

    // Test with zero initial value
    let mut v_zero_init = vec![G2Projective::zero()];
    vector_scalar_mul_add_online(&mut v_zero_init, &generators, scalar);
    let expected_zero_init = generators[0].mul_bigint(scalar.into_bigint());
    assert_eq!(
        v_zero_init[0].into_affine(),
        expected_zero_init.into_affine()
    );
}

#[test]
fn test_vector_scalar_mul_v_add_g_consistency() {
    let mut rng = test_rng();

    // Generate test data
    let num_generators = 10;
    let generators: Vec<G2Projective> = (0..num_generators)
        .map(|_| G2Affine::rand(&mut rng).into_group())
        .collect();
    let scalar = Fr::rand(&mut rng);

    // Generate initial values
    let initial_values: Vec<G2Projective> = (0..num_generators)
        .map(|_| G2Affine::rand(&mut rng).into_group())
        .collect();

    // Test online version
    let mut v_online = initial_values.clone();
    vector_scalar_mul_v_add_g_online(&mut v_online, &generators, scalar);

    // Test precomputed version
    let mut v_precomputed = initial_values.clone();
    let data = VectorScalarMulVData::new(scalar);
    vector_scalar_mul_v_add_g_precomputed(&mut v_precomputed, &generators, &data);

    // Compare with naive computation: scalar * v[i] + generators[i]
    for i in 0..num_generators {
        let expected = initial_values[i].mul_bigint(scalar.into_bigint()) + generators[i];

        // Convert to affine for comparison
        let expected_affine = expected.into_affine();
        let online_affine = v_online[i].into_affine();
        let precomputed_affine = v_precomputed[i].into_affine();

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
    }
}

#[test]
fn test_vector_scalar_mul_v_add_g_edge_cases() {
    let mut rng = test_rng();

    // Test with single generator
    let generators = vec![G2Affine::rand(&mut rng).into_group()];
    let scalar = Fr::rand(&mut rng);
    let initial = G2Affine::rand(&mut rng).into_group();

    let mut v = vec![initial];
    vector_scalar_mul_v_add_g_online(&mut v, &generators, scalar);

    let expected = initial.mul_bigint(scalar.into_bigint()) + generators[0];
    assert_eq!(v[0].into_affine(), expected.into_affine());

    // Test with zero scalar (should just be generators[i])
    let scalar_zero = Fr::from(0u64);
    let mut v_zero = vec![initial];
    vector_scalar_mul_v_add_g_online(&mut v_zero, &generators, scalar_zero);
    assert_eq!(v_zero[0].into_affine(), generators[0].into_affine());

    // Test with identity generator (should just be scalar * v[i])
    let identity_generators = vec![G2Projective::zero()];
    let mut v_identity = vec![initial];
    vector_scalar_mul_v_add_g_online(&mut v_identity, &identity_generators, scalar);
    let expected_identity = initial.mul_bigint(scalar.into_bigint());
    assert_eq!(v_identity[0].into_affine(), expected_identity.into_affine());

    // Test with zero initial value (should just be generators[i])
    let mut v_zero_init = vec![G2Projective::zero()];
    vector_scalar_mul_v_add_g_online(&mut v_zero_init, &generators, scalar);
    assert_eq!(v_zero_init[0].into_affine(), generators[0].into_affine());
}
