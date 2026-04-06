use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Instant;

use ark_bn254::{Fr, G2Affine, G2Projective};
use ark_ec::{AffineRepr, PrimeGroup};
use ark_ff::{PrimeField, UniformRand};
use ark_std::test_rng;

use jolt_optimizations::{
    vector_scalar_mul_add, vector_scalar_mul_add_online, vector_scalar_mul_add_precomputed,
    VectorScalarMulData,
};

fn bench_vector_scalar_mul_add(c: &mut Criterion) {
    let mut rng = test_rng();

    // Test with different vector sizes
    let vector_sizes = [1000];

    for &size in &vector_sizes {
        // Generate test data
        let generators: Vec<G2Projective> = (0..size)
            .map(|_| G2Affine::rand(&mut rng).into_group())
            .collect();
        let scalar = Fr::rand(&mut rng);

        // Initial values for the vector
        let initial_values: Vec<G2Projective> = (0..size)
            .map(|_| G2Affine::rand(&mut rng).into_group())
            .collect();

        let mut group = c.benchmark_group(format!("vector_scalar_mul_add_{}", size));

        // Benchmark naive approach
        group.bench_with_input(
            BenchmarkId::new("naive", size),
            &(&generators, &scalar, &initial_values),
            |b, &(generators, scalar, initial_values)| {
                b.iter(|| {
                    let mut v = initial_values.clone();
                    // Naive implementation: v[i] += scalar * generators[i]
                    for (v_i, g_i) in v.iter_mut().zip(generators.iter()) {
                        *v_i += g_i.mul_bigint(scalar.into_bigint());
                    }
                    black_box(v)
                })
            },
        );

        // Benchmark online GLV version
        group.bench_with_input(
            BenchmarkId::new("glv_online", size),
            &(&generators, &scalar, &initial_values),
            |b, &(generators, scalar, initial_values)| {
                b.iter(|| {
                    let mut v = initial_values.clone();
                    vector_scalar_mul_add_online(&mut v, generators, *scalar);
                    black_box(v)
                })
            },
        );

        // Benchmark precomputed GLV version (not counting precomputation time)
        let precomputed_data = VectorScalarMulData::new(&generators, scalar);
        group.bench_with_input(
            BenchmarkId::new("glv_precomputed", size),
            &(&precomputed_data, &initial_values),
            |b, &(data, initial_values)| {
                b.iter(|| {
                    let mut v = initial_values.clone();
                    vector_scalar_mul_add_precomputed(&mut v, data);
                    black_box(v)
                })
            },
        );

        // Benchmark precomputed GLV with precomputation time included
        group.bench_with_input(
            BenchmarkId::new("glv_precomputed_with_setup", size),
            &(&generators, &scalar, &initial_values),
            |b, &(generators, scalar, initial_values)| {
                b.iter_custom(|iters| {
                    let mut total_time = std::time::Duration::default();

                    for _ in 0..iters {
                        let mut v = initial_values.clone();

                        // Time includes both precomputation and multiplication
                        let start = Instant::now();
                        let data = VectorScalarMulData::new(generators, *scalar);
                        vector_scalar_mul_add_precomputed(&mut v, &data);
                        total_time += start.elapsed();

                        black_box(v);
                    }

                    total_time
                })
            },
        );

        // Benchmark the convenience function (which includes precomputation internally)
        group.bench_with_input(
            BenchmarkId::new("glv_convenience", size),
            &(&generators, &scalar, &initial_values),
            |b, &(generators, scalar, initial_values)| {
                b.iter(|| {
                    let mut v = initial_values.clone();
                    vector_scalar_mul_add(&mut v, generators, *scalar);
                    black_box(v)
                })
            },
        );

        group.finish();
    }
}

criterion_group!(benches, bench_vector_scalar_mul_add,);
criterion_main!(benches);
