// This bench prefers bn254; if not enabled, provide a no-op main
#[cfg(feature = "bn254")]
use ark_ff::{BigInteger, UniformRand};
#[cfg(feature = "bn254")]
use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
#[cfg(feature = "bn254")]
use ark_test_curves::bn254::Fr;
#[cfg(feature = "bn254")]
use criterion::{criterion_group, criterion_main, Criterion};

// Hack: copy over the helper functions from the Montgomery backend to be benched

#[cfg(feature = "bn254")]
fn mul_small_bench(c: &mut Criterion) {
    const SAMPLES: usize = 1000;
    // Use a fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(0u64);

    let a_s = (0..SAMPLES).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
    // let a_limbs_s = a_s.iter().map(|a| a.0.0).collect::<Vec<_>>();

    let b_u64_s = (0..SAMPLES).map(|_| rng.gen::<u64>()).collect::<Vec<_>>();
    // Convert u64 to Fr for standard multiplication benchmark
    let b_fr_s = b_u64_s.iter().map(|&b| Fr::from(b)).collect::<Vec<_>>();

    // Generate another set of random Fr elements for addition
    let c_s = (0..SAMPLES).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();

    // Generate test data for reduction benchmarks
    use ark_ff::BigInt;
    // Extract BigInt<4> from Fr elements for mul_u64_w_carry benchmark
    let a_bigints = a_s.iter().map(|a| a.0).collect::<Vec<_>>();

    // For Montgomery reduction: 2N-limb inputs (N=4 for bn254, so 2N=8)
    let bigint_2n_s = (0..SAMPLES)
        .map(|_| {
            BigInt::<8>([
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
            ])
        })
        .collect::<Vec<_>>();

    // For Barrett reductions: N+1, N+2, N+3 limb inputs
    let bigint_nplus1_s = (0..SAMPLES)
        .map(|_| {
            BigInt::<5>([
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
            ])
        })
        .collect::<Vec<_>>();

    let bigint_nplus2_s = (0..SAMPLES)
        .map(|_| {
            BigInt::<6>([
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
            ])
        })
        .collect::<Vec<_>>();

    let bigint_nplus3_s = (0..SAMPLES)
        .map(|_| {
            BigInt::<7>([
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
            ])
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("Fr Arithmetic Comparison");

    // Uncommented to compare with mul_u64_w_carry + Barrett reduction
    group.bench_function("mul_u64", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            // bn254 Fr has N=4 limbs => N+1 = 5
            criterion::black_box(a_s[i].mul_u64::<5>(b_u64_s[i]))
        })
    });

    // Benchmark just the multiplication phase (without Barrett reduction)
    group.bench_function("mul_u64_w_carry (multiplication only)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            // This is just the multiplication step, returns BigInt<5>
            criterion::black_box(a_bigints[i].mul_u64_w_carry::<5>(b_u64_s[i]))
        })
    });

    // group.bench_function("mul_i64", |bench| {
    //     let mut i = 0;
    //     bench.iter(|| {
    //         i = (i + 1) % SAMPLES;
    //         criterion::black_box(a_s[i].mul_i64::<5>(b_i64_s[i]))
    //     })
    // });

    // // Note: results might be worse than in real applications due to branch prediction being wrong
    // // 50% of the time
    // group.bench_function("mul_u128", |bench| {
    //     let mut i = 0;
    //     bench.iter(|| {
    //         i = (i + 1) % SAMPLES;
    //         // bn254 Fr has N=4 limbs => N+1 = 5, N+2 = 6
    //         criterion::black_box(a_s[i].mul_u128::<5, 6>(b_u128_s[i]))
    //     })
    // });

    // group.bench_function("mul_i128", |bench| {
    //     let mut i = 0;
    //     bench.iter(|| {
    //         i = (i + 1) % SAMPLES;
    //         criterion::black_box(a_s[i].mul_i128::<5, 6>(b_i128_s[i]))
    //     })
    // });

    group.bench_function("standard mul (Fr * Fr)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_s[i] * b_fr_s[i])
        })
    });

    // Benchmark the auxiliary function directly (assuming it's made public)
    // Note: Requires mul_u128_aux to be pub in montgomery_backend.rs
    // Need to import it if not already done via wildcard/specific import
    // Let's assume it's accessible via a_s[i].mul_u128_aux(...) for now
    // group.bench_function("mul_u128_aux (u128 inputs)", |bench| {
    //     let mut i = 0;
    //     bench.iter(|| {
    //         i = (i + 1) % SAMPLES;
    //         criterion::black_box(a_s[i].mul_u128_aux::<5, 6>(b_u128_s[i]))
    //     })
    // });

    group.bench_function("Addition (Fr + Fr)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(a_s[i] + c_s[i])
        })
    });

    // Reduction benchmarks
    group.bench_function("montgomery_reduce_in_place core (L=8)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            let mut x = bigint_2n_s[i];
            criterion::black_box(Fr::montgomery_reduce_in_place::<8>(&mut x))
        })
    });

    group.bench_function("from_montgomery_reduce (L=2N)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(Fr::from_montgomery_reduce::<8, 5>(bigint_2n_s[i]))
        })
    });

    // L=9 inputs: derive by zero-extending L=8 inputs
    let bigint_9_s = bigint_2n_s
        .iter()
        .map(|b8| ark_ff::BigInt::<9>::zero_extend_from::<8>(b8))
        .collect::<Vec<_>>();

    group.bench_function("montgomery_reduce_in_place core (L=9)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            let mut x = bigint_9_s[i];
            criterion::black_box(Fr::montgomery_reduce_in_place::<9>(&mut x))
        })
    });

    group.bench_function("from_montgomery_reduce (L=9)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(Fr::from_montgomery_reduce::<9, 5>(bigint_9_s[i]))
        })
    });

    // Barrett reductions
    group.bench_function("from_barrett_reduce (L=5)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(Fr::from_barrett_reduce::<5, 5>(bigint_nplus1_s[i]))
        })
    });

    group.bench_function("from_barrett_reduce (L=6)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(Fr::from_barrett_reduce::<6, 5>(bigint_nplus2_s[i]))
        })
    });

    group.bench_function("from_barrett_reduce (L=7)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            criterion::black_box(Fr::from_barrett_reduce::<7, 5>(bigint_nplus3_s[i]))
        })
    });

    // Linear combination benchmarks
    group.bench_function("linear_combination_u64 (2 terms)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            let pairs = [(a_s[i], b_u64_s[i]), (c_s[i], b_u64_s[(i + 1) % SAMPLES])];
            criterion::black_box(Fr::linear_combination_u64::<5>(&pairs))
        })
    });

    group.bench_function("linear_combination_u64 (4 terms)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            let pairs = [
                (a_s[i], b_u64_s[i]),
                (c_s[i], b_u64_s[(i + 1) % SAMPLES]),
                (a_s[(i + 2) % SAMPLES], b_u64_s[(i + 2) % SAMPLES]),
                (c_s[(i + 3) % SAMPLES], b_u64_s[(i + 3) % SAMPLES]),
            ];
            criterion::black_box(Fr::linear_combination_u64::<5>(&pairs))
        })
    });

    group.bench_function("linear_combination_i64 (2+2 terms)", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            let pos = [(a_s[i], b_u64_s[i]), (c_s[i], b_u64_s[(i + 1) % SAMPLES])];
            let neg = [
                (a_s[(i + 2) % SAMPLES], b_u64_s[(i + 2) % SAMPLES]),
                (c_s[(i + 3) % SAMPLES], b_u64_s[(i + 3) % SAMPLES]),
            ];
            criterion::black_box(Fr::linear_combination_i64::<5>(&pos, &neg))
        })
    });

    // Comparison: naive approach vs linear combination (using mul_u64 for fair comparison)
    group.bench_function("naive 2-term combination", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            let term1 = a_s[i].mul_u64::<5>(b_u64_s[i]);
            let term2 = c_s[i].mul_u64::<5>(b_u64_s[(i + 1) % SAMPLES]);
            criterion::black_box(term1 + term2)
        })
    });

    group.bench_function("naive 4-term combination", |bench| {
        let mut i = 0;
        bench.iter(|| {
            i = (i + 1) % SAMPLES;
            let term1 = a_s[i].mul_u64::<5>(b_u64_s[i]);
            let term2 = c_s[i].mul_u64::<5>(b_u64_s[(i + 1) % SAMPLES]);
            let term3 = a_s[(i + 2) % SAMPLES].mul_u64::<5>(b_u64_s[(i + 2) % SAMPLES]);
            let term4 = c_s[(i + 3) % SAMPLES].mul_u64::<5>(b_u64_s[(i + 3) % SAMPLES]);
            criterion::black_box(term1 + term2 + term3 + term4)
        })
    });

    group.finish();
}

#[cfg(feature = "bn254")]
criterion_group!(benches, mul_small_bench);
#[cfg(feature = "bn254")]
criterion_main!(benches);

#[cfg(not(feature = "bn254"))]
fn main() {}
