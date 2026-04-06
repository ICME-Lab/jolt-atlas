use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Instant;

use ark_bn254::{Fr, G2Affine, G2Projective};
use ark_ec::AffineRepr;
use ark_ff::{PrimeField, UniformRand};
use ark_std::test_rng;

use jolt_optimizations::{glv_four_precompute, vector_scalar_mul_add_gamma_g2_online};

fn bench_vector_scalar_mul_add_gamma_g2(c: &mut Criterion) {
    let mut rng = test_rng();

    // Test with different vector sizes
    let vector_sizes = [10000];

    for &size in &vector_sizes {
        // Generate test data
        let gamma: Vec<G2Projective> = (0..size)
            .map(|_| G2Affine::rand(&mut rng).into_group())
            .collect();
        let scalar = Fr::rand(&mut rng);

        // Initial values for the vector v
        let initial_v: Vec<G2Projective> = (0..size)
            .map(|_| G2Affine::rand(&mut rng).into_group())
            .collect();

        let mut group = c.benchmark_group(format!("vector_scalar_mul_add_gamma_g2_{}", size));
        group.sample_size(10); // Reduce sample size for larger vectors

        // Benchmark online version
        group.bench_with_input(
            BenchmarkId::new("online", size),
            &(&initial_v, &scalar, &gamma),
            |b, &(initial_v, scalar, gamma)| {
                b.iter(|| {
                    let mut v = initial_v.clone();
                    vector_scalar_mul_add_gamma_g2_online(&mut v, *scalar, gamma);
                    black_box(v)
                })
            },
        );

        // Benchmark hypothetical precomputed version with actual precomputation
        // This measures: precompute on v + use precomputed tables
        group.bench_with_input(
            BenchmarkId::new("precompute_v_then_use", size),
            &(&initial_v, &scalar, &gamma),
            |b, &(initial_v, scalar, gamma)| {
                b.iter_custom(|iters| {
                    let mut total_time = std::time::Duration::default();

                    for _ in 0..iters {
                        let mut v = initial_v.clone();

                        // Time includes both precomputation and the operation
                        let start = Instant::now();

                        // Precompute tables for v
                        let precomputed_v = glv_four_precompute(&v);

                        // Now compute scalar * v[i] using precomputed tables
                        // We use glv_four_scalar_mul which uses the precomputed data
                        use jolt_optimizations::glv_four_scalar_mul;
                        use rayon::prelude::*;
                        let products = glv_four_scalar_mul(&precomputed_v, *scalar);

                        // Add gamma to get v[i] = scalar * v[i] + gamma[i] in parallel
                        v.par_iter_mut()
                            .zip(products.par_iter())
                            .zip(gamma.par_iter())
                            .for_each(|((vi, &prod), &gamma_i)| {
                                *vi = prod + gamma_i;
                            });

                        total_time += start.elapsed();
                        black_box(v);
                    }

                    total_time
                })
            },
        );

        // Benchmark naive approach for comparison
        group.bench_with_input(
            BenchmarkId::new("naive", size),
            &(&initial_v, &scalar, &gamma),
            |b, &(initial_v, scalar, gamma)| {
                b.iter(|| {
                    let mut v = initial_v.clone();
                    // Naive implementation: v[i] = scalar * v[i] + gamma[i]
                    use ark_ec::PrimeGroup;
                    for (v_i, g_i) in v.iter_mut().zip(gamma.iter()) {
                        *v_i = v_i.mul_bigint(scalar.into_bigint()) + g_i;
                    }
                    black_box(v)
                })
            },
        );

        group.finish();
    }
}

criterion_group!(benches, bench_vector_scalar_mul_add_gamma_g2,);
criterion_main!(benches);
