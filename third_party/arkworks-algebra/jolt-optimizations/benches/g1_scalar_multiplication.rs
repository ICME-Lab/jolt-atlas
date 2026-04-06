use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;

use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, PrimeGroup};
use ark_ff::{PrimeField, UniformRand};
use ark_std::test_rng;

use jolt_optimizations::{
    glv_two_precompute, glv_two_precompute_windowed2_signed, glv_two_scalar_mul,
    glv_two_scalar_mul_online, glv_two_scalar_mul_windowed2_signed,
};

fn bench_g1_scalar_multiplication(c: &mut Criterion) {
    let mut rng = test_rng();

    // Fix a random scalar for all tests
    let scalar = Fr::rand(&mut rng);

    const NUM_TESTS: usize = 1000;

    // Generate random G1 points
    let points: Vec<G1Projective> = (0..NUM_TESTS)
        .map(|_| G1Affine::rand(&mut rng).into_group())
        .collect();

    let glv_precomputed = glv_two_precompute(&points);
    let glv_windowed2_signed = glv_two_precompute_windowed2_signed(&points);

    let mut group = c.benchmark_group("g1_scalar_multiplication");

    group.bench_with_input(
        BenchmarkId::new("naive", NUM_TESTS),
        &points,
        |b, points| {
            b.iter(|| {
                points
                    .par_iter()
                    .map(|point| black_box(point.mul_bigint(scalar.into_bigint())))
                    .collect::<Vec<G1Projective>>()
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("2d_online", NUM_TESTS),
        &points,
        |b, points| b.iter(|| black_box(glv_two_scalar_mul_online(scalar, points))),
    );

    group.bench_with_input(
        BenchmarkId::new("2d_precomputed", NUM_TESTS),
        &glv_precomputed,
        |b, glv_precomputed| b.iter(|| black_box(glv_two_scalar_mul(glv_precomputed, scalar))),
    );

    group.bench_with_input(
        BenchmarkId::new("2d_windowed2_signed", NUM_TESTS),
        &glv_windowed2_signed,
        |b, glv_windowed2_signed| {
            b.iter(|| {
                black_box(glv_two_scalar_mul_windowed2_signed(
                    glv_windowed2_signed,
                    scalar,
                ))
            })
        },
    );

    group.finish();
}

criterion_group!(benches, bench_g1_scalar_multiplication);
criterion_main!(benches);
