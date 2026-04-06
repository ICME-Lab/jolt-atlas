// Benchmark for BigInt operations
#[cfg(feature = "bn254")]
use ark_ff::{BigInt, BigInteger};
#[cfg(feature = "bn254")]
use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
#[cfg(feature = "bn254")]
use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(feature = "bn254")]
fn bigint_add_bench(c: &mut Criterion) {
    const SAMPLES: usize = 1000;
    // Use a fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(0u64);

    // Generate random BigInt<4> instances for benchmarking
    let a_bigints = (0..SAMPLES)
        .map(|_| {
            BigInt::<4>([
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
            ])
        })
        .collect::<Vec<_>>();

    let b_bigints = (0..SAMPLES)
        .map(|_| {
            BigInt::<4>([
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
            ])
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("BigInt<4> Addition Comparison");

    // Benchmark add_trunc with same limb count (4 -> 4)
    group.bench_function("add_trunc<4, 4>", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_bigints[i].add_trunc::<4, 4>(&b_bigints[i]))
        })
    });

    // Benchmark add_trunc with truncation (4 -> 3 limbs)
    group.bench_function("add_trunc<4, 3>", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_bigints[i].add_trunc::<4, 3>(&b_bigints[i]))
        })
    });

    // Benchmark add_trunc with expansion (4 -> 5 limbs)
    group.bench_function("add_trunc<4, 5>", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_bigints[i].add_trunc::<4, 5>(&b_bigints[i]))
        })
    });

    // Benchmark regular addition using add_with_carry
    group.bench_function("add_with_carry (regular add)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            let mut result = a_bigints[i];
            let carry = result.add_with_carry(&b_bigints[i]);
            criterion::black_box((result, carry))
        })
    });

    // Benchmark regular addition that ignores carry (for fair comparison)
    group.bench_function("add_with_carry (ignore carry)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            let mut result = a_bigints[i];
            result.add_with_carry(&b_bigints[i]);
            criterion::black_box(result)
        })
    });

    // Benchmark add_assign_trunc with same limb count (4 -> 4)
    group.bench_function("add_assign_trunc<4, 4>", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            let mut result = a_bigints[i];
            result.add_assign_trunc::<4, 4>(&b_bigints[i]);
            criterion::black_box(result)
        })
    });

    // Benchmark add_assign_trunc with truncation (4 -> 3 limbs)
    group.bench_function("add_assign_trunc<4, 3>", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            let mut result = a_bigints[i];
            result.add_assign_trunc::<4, 3>(&b_bigints[i]);
            criterion::black_box(result)
        })
    });

    // Benchmark add_assign_trunc with expansion (4 -> 5 limbs)
    group.bench_function("add_assign_trunc<4, 5>", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            let mut result = a_bigints[i];
            result.add_assign_trunc::<4, 5>(&b_bigints[i]);
            criterion::black_box(result)
        })
    });

    // Test case: addition that would overflow to compare truncation behavior
    let max_bigints = (0..SAMPLES)
        .map(|_| BigInt::<4>([u64::MAX, u64::MAX, u64::MAX, u64::MAX]))
        .collect::<Vec<_>>();

    group.bench_function("add_trunc overflow case", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            // This will overflow and be truncated
            criterion::black_box(max_bigints[i].add_trunc::<4, 4>(&max_bigints[i]))
        })
    });

    group.bench_function("add_with_carry overflow case", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            let mut result = max_bigints[i];
            let carry = result.add_with_carry(&max_bigints[i]);
            criterion::black_box((result, carry))
        })
    });

    group.bench_function("add_assign_trunc overflow case", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            let mut result = max_bigints[i];
            result.add_assign_trunc::<4, 4>(&max_bigints[i]);
            criterion::black_box(result)
        })
    });

    group.finish();
}

#[cfg(feature = "bn254")]
criterion_group!(benches, bigint_add_bench);
#[cfg(feature = "bn254")]
criterion_main!(benches);

#[cfg(not(feature = "bn254"))]
fn main() {}
