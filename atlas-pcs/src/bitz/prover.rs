//! The BitZ prover: commitment, and the opening of one evaluation claim over Fp128.

use super::gf128::{Gf128, GENERATOR};
use super::integers::{sample_prime, Wide};
use super::pack::pack;
use super::params::Params;
use super::proof::{bind_statement, eq_table, Commitment, Proof};
use super::{gkr, ligerito, ring_switch};
use atlas_algebra::field::fp128::Fp128;
use atlas_transcript::Transcript;
use rayon::prelude::*;

/// What the prover keeps of a committed bit vector.
pub struct Committed {
    words: Vec<u64>,
    packed: Vec<Gf128>,
    level: ligerito::prover::Level,
}

/// Commits to `2^params.num_vars` bits, given as words (bit `b` at position `b mod 64` of
/// word `b / 64`).
pub fn commit(words: Vec<u64>, params: &Params) -> (Commitment, Committed) {
    assert_eq!(words.len() * 64, 1 << params.num_vars, "bit count");
    let packed = pack(&words);
    let (root, level) = ligerito::prover::commit(&packed, &params.ligerito);
    (
        Commitment { root },
        Committed {
            words,
            packed,
            level,
        },
    )
}

/// For each column, the sum over the integers of `weights[row]` at the rows whose bit is set,
/// as the low 128 bits and the overflow above them.
fn column_sums(words: &[u64], weights: &[u128]) -> Vec<(u128, u64)> {
    words
        .par_chunks(weights.len() / 64)
        .map(|column| {
            let (mut low, mut high) = (0u128, 0u64);
            for (i, &word) in column.iter().enumerate() {
                let mut rest = word;
                while rest != 0 {
                    let row = 64 * i + rest.trailing_zeros() as usize;
                    let (sum, carry) = low.overflowing_add(weights[row]);
                    low = sum;
                    high += carry as u64;
                    rest &= rest - 1;
                }
            }
            (low, high)
        })
        .collect()
}

/// Proves the value of `Σ_x eq(point, x)·bits(x)` in Fp128 and returns it with the proof.
pub fn prove<T: Transcript>(
    commitment: &Commitment,
    committed: Committed,
    point: &[Fp128],
    params: &Params,
    transcript: &mut T,
) -> (Fp128, Proof) {
    assert_eq!(point.len(), params.num_vars, "point length");
    let Committed {
        words,
        packed,
        level,
    } = committed;
    let (row_point, column_point) = point.split_at(params.log_rows);
    let lift =
        |table: Vec<Fp128>| -> Vec<u128> { table.iter().map(|x| x.to_canonical_u128()).collect() };
    let (row_lifts, column_lifts) = (lift(eq_table(row_point)), lift(eq_table(column_point)));

    // Round 1: the claim over the integers, then its projection onto a random prime.
    let span = tracing::info_span!("bitz_projection").entered();
    #[allow(clippy::expect_used)]
    let lifted_claim = column_sums(&words, &row_lifts)
        .iter()
        .zip(&column_lifts)
        .map(|(&sum, &weight)| Wide::product(sum, weight))
        .try_fold(Wide::default(), |acc, term| acc.checked_add(&term))
        .expect("the lifted claim is below 2^320");
    let claim = lifted_claim.to_fp128();
    bind_statement(commitment, point, &claim, params, transcript);
    transcript.append_serializable(&lifted_claim);
    #[allow(clippy::expect_used)]
    let prime = transcript
        .pow(params.projection_pow_bits)
        .and_then(|()| sample_prime(params.prime_low_bits, params.prime_high_bits, transcript))
        .expect("the prover's proof of work and prime draw cannot fail");

    // Round 2: one integer per column, small enough to be an exponent.
    let row_residues: Vec<u128> = row_lifts.par_iter().map(|&lift| lift % prime).collect();
    let column_sums: Vec<u128> = column_sums(&words, &row_residues)
        .into_iter()
        .map(|(low, high)| {
            debug_assert_eq!(high, 0, "a column sum of residues fits 128 bits");
            low
        })
        .collect();
    transcript.append_serializable(&column_sums);
    drop(span);

    // Phase 3: each column sum in the exponent of the generator, as a grand product.
    let span = tracing::info_span!("bitz_grand_product").entered();
    let factors: Vec<Gf128> = row_residues
        .par_iter()
        .map(|&residue| GENERATOR.pow(residue))
        .collect();
    let (grand_product, outputs, bits_point, _) = gkr::prover::prove(
        &words,
        &factors,
        params.log_columns,
        params.target_bits,
        transcript,
    );
    debug_assert!(outputs
        .iter()
        .zip(&column_sums)
        .all(|(&output, &sum)| output == GENERATOR.pow(sum)));

    drop(span);

    // Phase 4: the claim left on the bits, against the packed commitment.
    let span = tracing::info_span!("bitz_ring_switch").entered();
    let (ring_columns, weights, _) =
        ring_switch::prover::reduce(&packed, &bits_point, params.target_bits, transcript);
    drop(span);
    let span = tracing::info_span!("bitz_ligerito").entered();
    let opening = ligerito::prover::prove(packed, weights, level, &params.ligerito, transcript);
    drop(span);
    (
        claim,
        Proof {
            lifted_claim,
            column_sums,
            grand_product,
            ring_columns,
            opening,
        },
    )
}
