#![allow(non_snake_case)]

use ark_bn254::{Fr, G1Projective, G2Projective};
use ark_ec::PrimeGroup;
use ark_ff::PrimeField;
use ark_std::UniformRand;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_optimizations::{
    dory_g1::{
        precompute_g1_generators, precompute_g1_generators_windowed2_signed,
        vector_add_scalar_mul_g1_online, vector_add_scalar_mul_g1_precomputed,
        vector_add_scalar_mul_g1_windowed2_signed, vector_scalar_mul_add_gamma_g1_online,
    },
    dory_g2::{
        precompute_g2_generators, precompute_g2_generators_windowed2_signed,
        vector_add_scalar_mul_g2_online, vector_add_scalar_mul_g2_precomputed,
        vector_add_scalar_mul_g2_windowed2_signed, vector_scalar_mul_add_gamma_g2_online,
    },
};
use rayon::prelude::*;
use std::time::Instant;

// ============================================================================
// Naive implementations
// ============================================================================

/// Naive: v[i] = v[i] + scalar * g[i] for G1
fn naive_vector_add_scalar_mul_g1(v: &mut [G1Projective], generators: &[G1Projective], scalar: Fr) {
    assert_eq!(v.len(), generators.len());
    v.par_iter_mut()
        .zip(generators.par_iter())
        .for_each(|(vi, gen)| {
            *vi += gen.mul_bigint(scalar.into_bigint());
        });
}

/// Naive: v[i] = scalar * v[i] + gamma[i] for G1
fn naive_vector_scalar_mul_add_gamma_g1(
    v: &mut [G1Projective],
    scalar: Fr,
    gamma: &[G1Projective],
) {
    assert_eq!(v.len(), gamma.len());
    v.par_iter_mut()
        .zip(gamma.par_iter())
        .for_each(|(vi, &gamma_i)| {
            *vi = vi.mul_bigint(scalar.into_bigint()) + gamma_i;
        });
}

/// Naive: v[i] = v[i] + scalar * g[i] for G2
fn naive_vector_add_scalar_mul_g2(v: &mut [G2Projective], generators: &[G2Projective], scalar: Fr) {
    assert_eq!(v.len(), generators.len());
    v.par_iter_mut()
        .zip(generators.par_iter())
        .for_each(|(vi, gen)| {
            *vi += gen.mul_bigint(scalar.into_bigint());
        });
}

/// Naive: v[i] = scalar * v[i] + gamma[i] for G2
fn naive_vector_scalar_mul_add_gamma_g2(
    v: &mut [G2Projective],
    scalar: Fr,
    gamma: &[G2Projective],
) {
    assert_eq!(v.len(), gamma.len());
    v.par_iter_mut()
        .zip(gamma.par_iter())
        .for_each(|(vi, &gamma_i)| {
            *vi = vi.mul_bigint(scalar.into_bigint()) + gamma_i;
        });
}

// ============================================================================
// Benchmark functions
// ============================================================================

fn bench_g1_add_scalar_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("G1_add_scalar_mul");
    let sizes = vec![1000];

    let mut rng = ark_std::test_rng();

    for size in sizes {
        // Generate test data
        let generators: Vec<G1Projective> =
            (0..size).map(|_| G1Projective::rand(&mut rng)).collect();
        let scalar = Fr::rand(&mut rng);

        // Precompute data for precomputed versions
        let precomputed_full = precompute_g1_generators(&generators);
        let precomputed_windowed = precompute_g1_generators_windowed2_signed(&generators);

        // Benchmark naive implementation
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b, _| {
            b.iter_custom(|iters| {
                let mut total_time = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let mut v: Vec<G1Projective> =
                        (0..size).map(|_| G1Projective::rand(&mut rng)).collect();
                    let start = Instant::now();
                    naive_vector_add_scalar_mul_g1(&mut v, &generators, scalar);
                    total_time += start.elapsed();
                    black_box(v);
                }
                total_time
            });
        });

        // Benchmark online implementation
        group.bench_with_input(BenchmarkId::new("online", size), &size, |b, _| {
            b.iter_custom(|iters| {
                let mut total_time = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let mut v: Vec<G1Projective> =
                        (0..size).map(|_| G1Projective::rand(&mut rng)).collect();
                    let start = Instant::now();
                    vector_add_scalar_mul_g1_online(&mut v, &generators, scalar);
                    total_time += start.elapsed();
                    black_box(v);
                }
                total_time
            });
        });

        // Benchmark precomputed implementation
        group.bench_with_input(BenchmarkId::new("precomputed", size), &size, |b, _| {
            b.iter_custom(|iters| {
                let mut total_time = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let mut v: Vec<G1Projective> =
                        (0..size).map(|_| G1Projective::rand(&mut rng)).collect();
                    let start = Instant::now();
                    vector_add_scalar_mul_g1_precomputed(
                        &mut v,
                        scalar,
                        &precomputed_full.shamir_tables,
                    );
                    total_time += start.elapsed();
                    black_box(v);
                }
                total_time
            });
        });

        // Benchmark windowed2 signed implementation
        group.bench_with_input(BenchmarkId::new("windowed2_signed", size), &size, |b, _| {
            b.iter_custom(|iters| {
                let mut total_time = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let mut v: Vec<G1Projective> =
                        (0..size).map(|_| G1Projective::rand(&mut rng)).collect();
                    let start = Instant::now();
                    vector_add_scalar_mul_g1_windowed2_signed(
                        &mut v,
                        scalar,
                        &precomputed_windowed,
                    );
                    total_time += start.elapsed();
                    black_box(v);
                }
                total_time
            });
        });
    }
    group.finish();
}

fn bench_g1_scalar_mul_add_gamma(c: &mut Criterion) {
    let mut group = c.benchmark_group("G1_scalar_mul_add_gamma");
    let sizes = vec![1000];

    let mut rng = ark_std::test_rng();

    for size in sizes {
        // Generate test data
        let gamma: Vec<G1Projective> = (0..size).map(|_| G1Projective::rand(&mut rng)).collect();
        let scalar = Fr::rand(&mut rng);

        // Benchmark naive implementation
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b, _| {
            b.iter_custom(|iters| {
                let mut total_time = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let mut v: Vec<G1Projective> =
                        (0..size).map(|_| G1Projective::rand(&mut rng)).collect();
                    let start = Instant::now();
                    naive_vector_scalar_mul_add_gamma_g1(&mut v, scalar, &gamma);
                    total_time += start.elapsed();
                    black_box(v);
                }
                total_time
            });
        });

        // Benchmark online implementation
        group.bench_with_input(BenchmarkId::new("online", size), &size, |b, _| {
            b.iter_custom(|iters| {
                let mut total_time = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let mut v: Vec<G1Projective> =
                        (0..size).map(|_| G1Projective::rand(&mut rng)).collect();
                    let start = Instant::now();
                    vector_scalar_mul_add_gamma_g1_online(&mut v, scalar, &gamma);
                    total_time += start.elapsed();
                    black_box(v);
                }
                total_time
            });
        });
    }
    group.finish();
}

fn bench_g2_add_scalar_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("G2_add_scalar_mul");
    let sizes = vec![1000];

    let mut rng = ark_std::test_rng();

    for size in sizes {
        // Generate test data
        let generators: Vec<G2Projective> =
            (0..size).map(|_| G2Projective::rand(&mut rng)).collect();
        let scalar = Fr::rand(&mut rng);

        // Precompute data for precomputed versions
        let precomputed_full = precompute_g2_generators(&generators);
        let precomputed_windowed = precompute_g2_generators_windowed2_signed(&generators);

        // Benchmark naive implementation
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b, _| {
            b.iter_custom(|iters| {
                let mut total_time = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let mut v: Vec<G2Projective> =
                        (0..size).map(|_| G2Projective::rand(&mut rng)).collect();
                    let start = Instant::now();
                    naive_vector_add_scalar_mul_g2(&mut v, &generators, scalar);
                    total_time += start.elapsed();
                    black_box(v);
                }
                total_time
            });
        });

        // Benchmark online implementation
        group.bench_with_input(BenchmarkId::new("online", size), &size, |b, _| {
            b.iter_custom(|iters| {
                let mut total_time = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let mut v: Vec<G2Projective> =
                        (0..size).map(|_| G2Projective::rand(&mut rng)).collect();
                    let start = Instant::now();
                    vector_add_scalar_mul_g2_online(&mut v, &generators, scalar);
                    total_time += start.elapsed();
                    black_box(v);
                }
                total_time
            });
        });

        // Benchmark precomputed implementation
        group.bench_with_input(BenchmarkId::new("precomputed", size), &size, |b, _| {
            b.iter_custom(|iters| {
                let mut total_time = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let mut v: Vec<G2Projective> =
                        (0..size).map(|_| G2Projective::rand(&mut rng)).collect();
                    let start = Instant::now();
                    vector_add_scalar_mul_g2_precomputed(
                        &mut v,
                        scalar,
                        &precomputed_full.shamir_tables,
                    );
                    total_time += start.elapsed();
                    black_box(v);
                }
                total_time
            });
        });

        // Benchmark windowed2 signed implementation
        group.bench_with_input(BenchmarkId::new("windowed2_signed", size), &size, |b, _| {
            b.iter_custom(|iters| {
                let mut total_time = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let mut v: Vec<G2Projective> =
                        (0..size).map(|_| G2Projective::rand(&mut rng)).collect();
                    let start = Instant::now();
                    vector_add_scalar_mul_g2_windowed2_signed(
                        &mut v,
                        scalar,
                        &precomputed_windowed,
                    );
                    total_time += start.elapsed();
                    black_box(v);
                }
                total_time
            });
        });
    }
    group.finish();
}

fn bench_g2_scalar_mul_add_gamma(c: &mut Criterion) {
    let mut group = c.benchmark_group("G2_scalar_mul_add_gamma");
    let sizes = vec![1000];

    let mut rng = ark_std::test_rng();

    for size in sizes {
        // Generate test data
        let gamma: Vec<G2Projective> = (0..size).map(|_| G2Projective::rand(&mut rng)).collect();
        let scalar = Fr::rand(&mut rng);

        // Benchmark naive implementation
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b, _| {
            b.iter_custom(|iters| {
                let mut total_time = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let mut v: Vec<G2Projective> =
                        (0..size).map(|_| G2Projective::rand(&mut rng)).collect();
                    let start = Instant::now();
                    naive_vector_scalar_mul_add_gamma_g2(&mut v, scalar, &gamma);
                    total_time += start.elapsed();
                    black_box(v);
                }
                total_time
            });
        });

        // Benchmark online implementation
        group.bench_with_input(BenchmarkId::new("online", size), &size, |b, _| {
            b.iter_custom(|iters| {
                let mut total_time = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let mut v: Vec<G2Projective> =
                        (0..size).map(|_| G2Projective::rand(&mut rng)).collect();
                    let start = Instant::now();
                    vector_scalar_mul_add_gamma_g2_online(&mut v, scalar, &gamma);
                    total_time += start.elapsed();
                    black_box(v);
                }
                total_time
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_g1_add_scalar_mul,
    bench_g1_scalar_mul_add_gamma,
    bench_g2_add_scalar_mul,
    bench_g2_scalar_mul_add_gamma
);
criterion_main!(benches);
