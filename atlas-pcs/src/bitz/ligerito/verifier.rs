//! The Ligerito verifier.
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::unimplemented,
    clippy::todo,
    clippy::indexing_slicing
)]

use super::{Proof, Schedule};
use crate::bitz::gf128::Gf128;
use crate::bitz::grinding;
use crate::bitz::merkle::{verify_row, Digest};
use crate::bitz::multilinear::{eq_table, tensor_expand};
use crate::bitz::rs::Code;
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_transcript::Transcript;

fn malformed(what: &str) -> ProofVerifyError {
    ProofVerifyError::Malformed(format!("Ligerito: {what}"))
}

fn rejected(what: &str) -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof(format!("Ligerito: {what}"))
}

/// The symbols a level's queries fixed: "generator row `row` of the level's code, times the
/// next polynomial, is `symbol`", batched into the running claim with powers of `beta`.
struct LevelClaims {
    code: Code,
    beta: Gf128,
    rows: Vec<usize>,
    /// The number of fold challenges drawn up to and including this level.
    challenges_before_next: usize,
}

/// Verifies `Σ_b w(b)·f(b) = claim` for the polynomial `f` committed as `root`.
///
/// The verifier never reads the weight `w`. `initial_weight_tail` receives every fold
/// challenge, in order, and returns `w` with its first variables fixed to them: its values on
/// the cube of the `schedule.final_vars()` variables left.
pub fn verify<T: Transcript>(
    root: &Digest,
    claim: Gf128,
    initial_weight_tail: impl FnOnce(&[Gf128]) -> Result<Vec<Gf128>, ProofVerifyError>,
    proof: &Proof,
    schedule: &Schedule,
    transcript: &mut T,
) -> Result<(), ProofVerifyError> {
    let levels = schedule.levels();
    if proof.roots.len() + 1 != levels || proof.openings.len() != levels {
        return Err(malformed("level count"));
    }
    let final_len = 1usize
        .checked_shl(schedule.final_vars() as u32)
        .ok_or_else(|| malformed("final size"))?;
    if proof.final_values.len() != final_len {
        return Err(malformed("final values length"));
    }

    let mut claim = claim;
    let mut root = *root;
    let mut rounds = proof.rounds.iter();
    let mut nonces = proof.nonces.iter();
    let mut next_roots = proof.roots.iter();
    let mut challenges: Vec<Gf128> = Vec::new();
    let mut earlier: Vec<LevelClaims> = Vec::new();

    for (level, (&fold, openings)) in schedule.folds.iter().zip(&proof.openings).enumerate() {
        let row_vars = schedule
            .row_vars(level)
            .ok_or_else(|| malformed("schedule"))?;
        let log_block = row_vars + schedule.log_inv_rate;
        let pow_bits = schedule
            .round_pow_bits(level)
            .ok_or_else(|| malformed("schedule"))?;

        let fold_start = challenges.len();
        for _ in 0..fold {
            let message = rounds.next().ok_or_else(|| malformed("missing round"))?;
            let [constant, leading] = *message;
            transcript.append_serializable(message);
            grinding::verify(
                pow_bits,
                *nonces.next().ok_or_else(|| malformed("missing nonce"))?,
                transcript,
            )?;
            let r = Gf128(transcript.challenge_u128());
            // h(0) + h(1) = claim fixes the linear coefficient.
            let linear = claim + leading;
            claim = constant + r * (linear + r * leading);
            challenges.push(r);
        }
        let fold_challenges = challenges
            .get(fold_start..)
            .ok_or_else(|| malformed("challenges"))?;

        let next_root = if level + 1 < levels {
            let next = next_roots.next().ok_or_else(|| malformed("missing root"))?;
            transcript.append_bytes(next);
            Some(*next)
        } else {
            transcript.append_serializable(&proof.final_values);
            None
        };

        if openings.len() != schedule.queries {
            return Err(malformed("query count"));
        }
        let block_mask = 1usize
            .checked_shl(log_block as u32)
            .ok_or_else(|| malformed("block length"))?
            - 1;
        let fold_weights = eq_table(fold_challenges);
        let mut rows = Vec::with_capacity(openings.len());
        let mut symbols = Vec::with_capacity(openings.len());
        for opening in openings {
            let row = transcript.challenge_u128() as usize & block_mask;
            if opening.row.len() != fold_weights.len() {
                return Err(malformed("opened row length"));
            }
            if !verify_row(&root, log_block, row, &opening.row, &opening.path) {
                return Err(rejected("Merkle path"));
            }
            let symbol = fold_weights
                .iter()
                .zip(&opening.row)
                .fold(Gf128::ZERO, |acc, (&w, &x)| acc + w * x);
            rows.push(row);
            symbols.push(symbol);
        }

        let code = Code::new(row_vars, schedule.log_inv_rate);
        match next_root {
            Some(next) => {
                grinding::verify(
                    schedule.batch_pow_bits(),
                    *nonces.next().ok_or_else(|| malformed("missing nonce"))?,
                    transcript,
                )?;
                let beta = Gf128(transcript.challenge_u128());
                let mut power = beta;
                for &symbol in &symbols {
                    claim += power * symbol;
                    power *= beta;
                }
                earlier.push(LevelClaims {
                    code,
                    beta,
                    rows,
                    challenges_before_next: challenges.len(),
                });
                root = next;
            }
            None => {
                // The last fold is in the clear: check the opened rows against its encoding.
                for (&row, &symbol) in rows.iter().zip(&symbols) {
                    let generator_row = tensor_expand(&code.row_factors(row));
                    let expected = generator_row
                        .iter()
                        .zip(&proof.final_values)
                        .fold(Gf128::ZERO, |acc, (&g, &y)| acc + g * y);
                    if expected != symbol {
                        return Err(rejected("final codeword symbol"));
                    }
                }
            }
        }
    }
    if rounds.next().is_some() || next_roots.next().is_some() || nonces.next().is_some() {
        return Err(malformed("trailing rounds, roots or nonces"));
    }

    // The weight left after every fold: the caller's, plus each earlier level's batched
    // generator rows with their folded variables fixed.
    let mut weight = initial_weight_tail(&challenges)?;
    if weight.len() != final_len {
        return Err(malformed("weight tail length"));
    }
    for level in &earlier {
        let later = challenges
            .get(level.challenges_before_next..)
            .ok_or_else(|| malformed("challenges"))?;
        let mut power = level.beta;
        for &row in &level.rows {
            let factors = level.code.row_factors(row);
            if factors.len() < later.len() {
                return Err(malformed("fold longer than the level"));
            }
            let (folded, free) = factors.split_at(later.len());
            // A factor pair (1, T) at the challenge r is 1 + r·(1 + T).
            let scalar = folded.iter().zip(later).fold(power, |acc, (&t, &r)| {
                acc * (Gf128::ONE + r * (Gf128::ONE + t))
            });
            for (w, &g) in weight.iter_mut().zip(&tensor_expand(free)) {
                *w += scalar * g;
            }
            power *= level.beta;
        }
    }
    let expected = weight
        .iter()
        .zip(&proof.final_values)
        .fold(Gf128::ZERO, |acc, (&w, &y)| acc + w * y);
    if expected != claim {
        return Err(ProofVerifyError::SumcheckVerificationError);
    }
    Ok(())
}
