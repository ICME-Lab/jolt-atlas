#[cfg(feature = "bn254")]
use ark_ec::pairing::{CompressedPairing, Pairing};
#[cfg(feature = "bn254")]
use ark_ff::UniformRand;
#[cfg(feature = "bn254")]
use ark_std::test_rng;
#[cfg(feature = "bn254")]
use ark_test_curves::bn254::Bn254;
#[cfg(feature = "bn254")]
use ark_test_curves::bn254::CompressedFq12;
#[cfg(feature = "bn254")]
use ark_test_curves::bn254::Fq2;
#[cfg(feature = "bn254")]
use ark_test_curves::bn254::{G1Projective, G2Projective};
#[cfg(feature = "bn254")]
use criterion::{black_box, BatchSize, Criterion};

#[cfg(feature = "bn254")]
fn pairing_bench(compressed: bool, len: usize) -> impl FnOnce(&mut Criterion) {
    move |c: &mut Criterion| {
        let bench_name = format!(
            "{}_pairing ({} pairs), ",
            if compressed {
                "compressed"
            } else {
                "uncompressed"
            },
            len
        );

        let mut rng = test_rng();

        c.bench_function(&bench_name, |b| {
            b.iter_batched(
                || {
                    let g1 = (0..len)
                        .map(|_| G1Projective::rand(&mut rng))
                        .collect::<Vec<_>>();
                    let g2 = (0..len)
                        .map(|_| G2Projective::rand(&mut rng))
                        .collect::<Vec<_>>();
                    (g1, g2)
                },
                |(g1, g2)| {
                    // NOTE: this will cause a problem if the following is wrapped in a black_box.
                    // See related issue: https://github.com/bheisler/criterion.rs/issues/862
                    // and the article on std::hint::black_box https://gendignoux.com/blog/2022/01/31/rust-benchmarks.html
                    // which the criterion black_box method is a wrapper of.
                    if compressed {
                        let _ = Bn254::compressed_multi_pairing(g1, g2);
                    } else {
                        let _ = Bn254::multi_pairing(g1, g2);
                    }
                },
                BatchSize::LargeInput,
            )
        });
    }
}

#[cfg(feature = "bn254")]
fn decompression(c: &mut Criterion) {
    let mut rng = test_rng();

    c.bench_function("pairing_value_decompression (1 pair)", |b| {
        b.iter_batched(
            || CompressedFq12((Fq2::rand(&mut rng), Fq2::rand(&mut rng))),
            |compressed_value| black_box(compressed_value.decompress_to_fq12()),
            BatchSize::SmallInput,
        )
    });
}

#[cfg(feature = "bn254")]
fn compressed_multi_pairing_len_100_bench(c: &mut Criterion) {
    pairing_bench(true, 100)(c);
}

#[cfg(feature = "bn254")]
fn uncompressed_multi_pairing_len_100_bench(c: &mut Criterion) {
    pairing_bench(false, 100)(c);
}

#[cfg(feature = "bn254")]
fn compressed_multi_pairing_len_1_bench(c: &mut Criterion) {
    pairing_bench(true, 1)(c);
}

#[cfg(feature = "bn254")]
fn uncompressed_multi_pairing_len_1_bench(c: &mut Criterion) {
    pairing_bench(false, 1)(c);
}

#[cfg(feature = "bn254")]
criterion::criterion_group!(
    benches,
    compressed_multi_pairing_len_1_bench,
    uncompressed_multi_pairing_len_1_bench,
    compressed_multi_pairing_len_100_bench,
    uncompressed_multi_pairing_len_100_bench,
    decompression
);
#[cfg(feature = "bn254")]
criterion::criterion_main!(benches);

#[cfg(not(feature = "bn254"))]
fn main() {}
