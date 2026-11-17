//! The BitZ verifier.
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::unimplemented,
    clippy::todo,
    clippy::indexing_slicing
)]

use super::gf128::{Gf128, GENERATOR};
use super::integers::{add_mod, mul_mod, sample_prime, Wide, FP128_MODULUS};
use super::params::Params;
use super::proof::{bind_statement, eq_table, Commitment, Proof};
use super::{gkr, ligerito, ring_switch};
use atlas_algebra::field::fp128::Fp128;
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_transcript::Transcript;
use rayon::prelude::*;

fn rejected(what: &str) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(format!("BitZ: {what}"))
}

/// Verifies that the committed bits satisfy `Σ_x eq(point, x)·bits(x) = claim` in Fp128.
pub fn verify<T: Transcript>(
    commitment: &Commitment,
    point: &[Fp128],
    claim: &Fp128,
    proof: &Proof,
    params: &Params,
    transcript: &mut T,
) -> Result<(), ProofVerifyError> {
    if point.len() != params.num_vars {
        return Err(ProofVerifyError::InvalidInputLength(
            params.num_vars,
            point.len(),
        ));
    }
    let (row_point, column_point) = point
        .split_at_checked(params.log_rows)
        .ok_or_else(|| rejected("point"))?;
    let lift =
        |table: Vec<Fp128>| -> Vec<u128> { table.iter().map(|x| x.to_canonical_u128()).collect() };
    let (row_lifts, column_lifts) = (lift(eq_table(row_point)), lift(eq_table(column_point)));

    // Round 1. The lifted claim of an honest prover is at most 2^n·(p − 1)²; it must reduce to
    // the claim modulo p.
    let top = FP128_MODULUS - 1;
    let bound = Wide::product((top, 0), top)
        .checked_shl(params.num_vars as u32)
        .ok_or_else(|| rejected("size"))?;
    if !proof.lifted_claim.le(&bound) {
        return Err(rejected("the lifted claim is out of range"));
    }
    if proof.lifted_claim.to_fp128() != *claim {
        return Err(rejected("the lifted claim does not reduce to the claim"));
    }
    bind_statement(commitment, point, claim, params, transcript);
    transcript.append_serializable(&proof.lifted_claim);
    transcript.pow(params.projection_pow_bits)?;
    let prime = sample_prime(params.prime_low_bits, params.prime_high_bits, transcript)?;

    // Round 2. Each column sum is at most 2^log_rows·(q − 1), below the generator's order, and
    // the sums combine to the lifted claim modulo q.
    if proof.column_sums.len() != column_lifts.len() {
        return Err(ProofVerifyError::Malformed("BitZ: column sum count".into()));
    }
    let sum_bound = (prime - 1)
        .checked_shl(params.log_rows as u32)
        .filter(|bound| bound >> params.log_rows == prime - 1)
        .ok_or_else(|| rejected("parameters"))?;
    if proof.column_sums.iter().any(|&sum| sum > sum_bound) {
        return Err(rejected("a column sum is out of range"));
    }
    let combined = proof
        .column_sums
        .iter()
        .zip(&column_lifts)
        .fold(0, |acc, (&sum, &lift)| {
            add_mod(acc, mul_mod(sum % prime, lift % prime, prime), prime)
        });
    if combined != proof.lifted_claim.rem(prime) {
        return Err(rejected("the column sums do not combine to the claim"));
    }
    transcript.append_serializable(&proof.column_sums);

    // Phase 3. In the exponent of the generator, each column sum is a product over its rows.
    let outputs: Vec<Gf128> = proof
        .column_sums
        .par_iter()
        .map(|&sum| GENERATOR.pow(sum))
        .collect();
    let factors_plus_one: Vec<Gf128> = row_lifts
        .par_iter()
        .map(|&lift| GENERATOR.pow(lift % prime) + Gf128::ONE)
        .collect();
    let (bits_point, bits_value) = gkr::verify(
        &outputs,
        &factors_plus_one,
        &proof.grand_product,
        params.target_bits,
        transcript,
    )?;

    // Phase 4. The claim on the bits, against the packed commitment.
    let (reduced, weight) = ring_switch::reduce(
        &proof.ring_columns,
        &bits_point,
        bits_value,
        params.target_bits,
        transcript,
    )?;
    ligerito::verifier::verify(
        &commitment.root,
        reduced,
        |challenges| weight.tail(challenges),
        &proof.opening,
        &params.ligerito,
        transcript,
    )
}
