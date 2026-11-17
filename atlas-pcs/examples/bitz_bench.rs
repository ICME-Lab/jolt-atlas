//! Commits to random bits with BitZ, proves one evaluation claim over Fp128 and verifies it.
//!
//! `bitz_bench <n> [--reps R] [--trace]` runs at `2^n` bits and a 128-bit target, and prints
//! the median commit, prove and verify times over `R` repetitions (default 3), the proof size
//! and the process's peak memory. `--trace` prints the prover's phases as they close. The
//! thread count is rayon's (`RAYON_NUM_THREADS`).

use atlas_algebra::field::fp128::Fp128;
use atlas_algebra::serialize::Canonical;
use atlas_pcs::bitz::{proof::eq_table, prover, verifier, Params};
use atlas_transcript::{Blake2bProver, Blake2bVerifier, ProvingTranscript, VerifyingTranscript};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use std::time::{Duration, Instant};

const LABEL: &[u8] = b"bitz-bench";

fn median(mut times: Vec<Duration>) -> Duration {
    times.sort();
    times[times.len() / 2]
}

/// The process's peak resident set in bytes, where the platform reports it.
fn peak_memory() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let kilobytes: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kilobytes * 1024)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut num_vars = None;
    let mut reps = 3usize;
    let mut trace = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--reps" => {
                reps = args[i + 1].parse().expect("--reps takes a count");
                i += 2;
            }
            "--trace" => {
                trace = true;
                i += 1;
            }
            n => {
                num_vars = Some(n.parse::<usize>().expect("n is a number of variables"));
                i += 1;
            }
        }
    }
    let num_vars = num_vars.expect("usage: bitz_bench <n> [--reps R] [--trace]");
    let params = Params::new(num_vars, 128).expect("n is between 12 and 31");
    if trace {
        use tracing_subscriber::fmt::format::FmtSpan;
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::CLOSE)
            .with_target(false)
            .init();
    }

    println!(
        "bitz_bench n={num_vars} rows=2^{} columns=2^{} threads={} reps={reps}",
        params.log_rows,
        params.log_columns,
        rayon::current_num_threads()
    );
    println!(
        "target {} bits; projection: primes in [2^{}, 2^{}), error 2^-{:.1} before {} bits of proof of work",
        params.target_bits,
        params.prime_low_bits,
        params.prime_high_bits,
        params.projection_error_bits(),
        params.projection_pow_bits
    );
    println!(
        "ligerito: folds {:?}, {} queries per level at rate 1/{}, query error 2^-{:.1}",
        params.ligerito.folds,
        params.ligerito.queries,
        1 << params.ligerito.log_inv_rate,
        params.ligerito.query_error_bits()
    );

    let mut rng = StdRng::seed_from_u64(1);
    let words: Vec<u64> = (0..1usize << (num_vars - 6))
        .map(|_| rng.next_u64())
        .collect();
    let point: Vec<Fp128> = (0..num_vars)
        .map(|_| Fp128::from(((rng.next_u64() as u128) << 64) | rng.next_u64() as u128))
        .collect();

    // An evaluation claim: the row coordinates, and the `eq` table of the column coordinates.
    let (rows, columns) = point.split_at(params.log_rows);
    let weights = eq_table(columns);

    let (mut commits, mut proves, mut verifies) = (Vec::new(), Vec::new(), Vec::new());
    let mut proof_bytes = 0;
    for _ in 0..reps {
        let words = words.clone();
        let start = Instant::now();
        let (commitment, committed) = prover::commit(words, &params);
        commits.push(start.elapsed());

        let mut transcript = Blake2bProver::new(LABEL);
        let start = Instant::now();
        let (claim, proof) = prover::prove(
            &commitment,
            &committed,
            rows,
            &weights,
            &params,
            &mut transcript,
        );
        proves.push(start.elapsed());
        let narg = transcript.into_narg();
        proof_bytes = proof.encoded_len() + narg.len();

        let mut transcript = Blake2bVerifier::new(LABEL, narg);
        let start = Instant::now();
        verifier::verify(
            &commitment,
            rows,
            &weights,
            &claim,
            &proof,
            &params,
            &mut transcript,
        )
        .expect("the proof verifies");
        verifies.push(start.elapsed());
    }

    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let (commit, prove, verify) = (median(commits), median(proves), median(verifies));
    println!("commit: median {:.1} ms", ms(commit));
    println!("prove: median {:.1} ms", ms(prove));
    println!("commit + prove: {:.1} ms", ms(commit + prove));
    println!("verify: median {:.1} ms", ms(verify));
    println!("proof: {proof_bytes} bytes (encoded proof and proof-of-work nonces)");
    match peak_memory() {
        Some(bytes) => println!("peak memory: {:.2} GB", bytes as f64 / 1e9),
        None => println!("peak memory: not reported on this platform"),
    }
}
