//! Akita commit microbenchmark (Phase 3 of `docs/measurement-plan.md`): the cost of committing
//! one 4-bit lookup digit per row as a K = 16 one-hot column, against dense columns holding the
//! same information in [0, 16), as two columns in [0, 4), or as four Boolean columns.
//!
//! Usage: `commit_bench [--log-rows N] [--columns C]` (defaults: 20 rows, 64 columns, the shape of
//! GPT-2's largest one-hot classes). Each arm reports witness-build time, commit time, and the
//! commit cost per digit and per key bit.
use std::time::Instant;

use atlas_algebra::{
    field::fp128::Fp128,
    poly::{multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial},
};
use atlas_pcs::{akita::AkitaScheme, commitment_scheme::CommitmentScheme};
use jolt_akita::AKITA_ONE_HOT_K16;

/// Deterministic xorshift, so every arm sees the same digit stream.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str, default: usize| -> usize {
        argv.iter()
            .position(|a| a == flag)
            .map_or(default, |i| argv[i + 1].parse().expect("a number"))
    };
    let log_rows = value("--log-rows", 20);
    let columns = value("--columns", 64);
    let rows = 1usize << log_rows;
    let digits = rows * columns;
    println!(
        "rows 2^{log_rows}, {columns} columns: {digits} four-bit digits ({} key bits) per arm",
        digits * 4
    );

    // One-hot K = 16: 4 address variables + log_rows cycle variables.
    let setup = AkitaScheme::setup_prover((log_rows + 4).max(log_rows + columns.ilog2() as usize));

    let run = |name: &str, build: &mut dyn FnMut() -> Vec<MultilinearPolynomial<Fp128>>| {
        let t = Instant::now();
        let polys = build();
        let witness = t.elapsed();
        let t = Instant::now();
        let (batch, per_poly) = AkitaScheme::batch_commit_all(&polys, &setup);
        let commit = t.elapsed();
        let cells: usize = polys.iter().map(|p| p.len()).sum();
        println!(
            "{name:<28} polys {:>4}  witness {:>8.3?}  commit {:>8.3?}  {:>6.1} ns/digit  {:>6.1} ns/key-bit  ({} classes, {} cells)",
            polys.len(),
            witness,
            commit,
            commit.as_nanos() as f64 / digits as f64,
            commit.as_nanos() as f64 / (digits * 4) as f64,
            batch.classes.len(),
            cells
        );
        std::hint::black_box(per_poly);
    };

    let digit = |rng: &mut Rng| (rng.next() >> 60) as u8; // uniform in [0, 16)

    run("one-hot K=16 (today)", &mut || {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        (0..columns)
            .map(|_| {
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    (0..rows).map(|_| Some(digit(&mut rng) as u16)).collect(),
                    AKITA_ONE_HOT_K16,
                ))
            })
            .collect()
    });
    // The slack chunks of the clamp buckets: columns that are empty on every row. How much an
    // empty member of a one-hot class costs decides whether declared widths save commit time.
    run("one-hot, 24 of 64 cols empty", &mut || {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        (0..columns)
            .map(|c| {
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    (0..rows)
                        .map(|_| (c * 8 < columns * 5).then(|| digit(&mut rng) as u16))
                        .collect(),
                    AKITA_ONE_HOT_K16,
                ))
            })
            .collect()
    });
    run("dense u8 in [0,16), 1 col", &mut || {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        (0..columns)
            .map(|_| {
                MultilinearPolynomial::from((0..rows).map(|_| digit(&mut rng)).collect::<Vec<u8>>())
            })
            .collect()
    });
    run("dense u8 in [0,4), 2 cols", &mut || {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        (0..columns)
            .flat_map(|_| {
                let d: Vec<u8> = (0..rows).map(|_| digit(&mut rng)).collect();
                [
                    MultilinearPolynomial::from(d.iter().map(|x| x & 3).collect::<Vec<u8>>()),
                    MultilinearPolynomial::from(d.iter().map(|x| x >> 2).collect::<Vec<u8>>()),
                ]
            })
            .collect()
    });
    run("dense bool, 4 cols", &mut || {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        (0..columns)
            .flat_map(|_| {
                let d: Vec<u8> = (0..rows).map(|_| digit(&mut rng)).collect();
                (0..4)
                    .map(|b| {
                        MultilinearPolynomial::from(
                            d.iter().map(|x| (x >> b) & 1 == 1).collect::<Vec<bool>>(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    });
}
