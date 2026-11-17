//! The Ligerito prover.

use super::{Proof, RowOpening, Schedule};
use crate::bitz::gf128::Gf128;
use crate::bitz::merkle::{Digest, Tree};
use crate::bitz::multilinear::{fold_first, tensor_expand};
use crate::bitz::rs::Code;
use atlas_transcript::Transcript;
use rayon::prelude::*;

/// One committed level: its code, encoded matrix and Merkle tree.
pub struct Level {
    code: Code,
    width: usize,
    encoded: Vec<Gf128>,
    tree: Tree,
}

impl Level {
    fn commit(values: &[Gf128], fold: usize, log_inv_rate: usize) -> Self {
        let width = 1usize << fold;
        let row_vars = values.len().trailing_zeros() as usize - fold;
        let code = Code::new(row_vars, log_inv_rate);
        let encoded = code.encode_columns(values, width);
        let tree = Tree::commit(&encoded, width);
        Self {
            code,
            width,
            encoded,
            tree,
        }
    }

    fn open(&self, index: usize) -> RowOpening {
        RowOpening {
            row: self.encoded[index * self.width..(index + 1) * self.width].to_vec(),
            path: self.tree.open(index),
        }
    }
}

/// Commits to a polynomial of `schedule.num_vars` variables: the root and the first level.
pub fn commit(values: &[Gf128], schedule: &Schedule) -> (Digest, Level) {
    assert_eq!(values.len(), 1 << schedule.num_vars, "polynomial size");
    let level = Level::commit(values, schedule.folds[0], schedule.log_inv_rate);
    (level.tree.root(), level)
}

/// Proves `Σ_b weights(b)·values(b) = claim` for the polynomial committed as `first`. The
/// claim itself is not sent: the caller's protocol fixes it.
pub fn prove<T: Transcript>(
    values: Vec<Gf128>,
    weights: Vec<Gf128>,
    first: &Level,
    schedule: &Schedule,
    transcript: &mut T,
) -> Proof {
    assert_eq!(values.len(), 1 << schedule.num_vars, "polynomial size");
    assert_eq!(weights.len(), values.len(), "weight size");
    let (mut values, mut weights) = (values, weights);
    // The level being opened: the caller's first, then each one committed here.
    let mut later: Option<Level> = None;
    let mut proof = Proof {
        roots: Vec::new(),
        rounds: Vec::new(),
        final_values: Vec::new(),
        openings: Vec::new(),
    };
    for (index, &fold) in schedule.folds.iter().enumerate() {
        #[allow(clippy::expect_used)]
        let pow_bits = schedule
            .round_pow_bits(index)
            .expect("the level is in the schedule");
        for _ in 0..fold {
            let message = round_message(&values, &weights);
            transcript.append_serializable(&message);
            grind(transcript, pow_bits);
            let r = Gf128(transcript.challenge_u128());
            values = fold_first(&values, r);
            weights = fold_first(&weights, r);
            proof.rounds.push(message);
        }
        let next = if let Some(&next_fold) = schedule.folds.get(index + 1) {
            let next = Level::commit(&values, next_fold, schedule.log_inv_rate);
            transcript.append_bytes(&next.tree.root());
            proof.roots.push(next.tree.root());
            Some(next)
        } else {
            transcript.append_serializable(&values);
            proof.final_values = values.clone();
            None
        };

        let level = later.as_ref().unwrap_or(first);
        let block_mask = (1usize << level.code.log_block()) - 1;
        let queried: Vec<usize> = (0..schedule.queries)
            .map(|_| transcript.challenge_u128() as usize & block_mask)
            .collect();
        proof
            .openings
            .push(queried.par_iter().map(|&row| level.open(row)).collect());

        if let Some(next) = next {
            grind(transcript, schedule.batch_pow_bits());
            let beta = Gf128(transcript.challenge_u128());
            // Each opened row, folded, is a symbol of the next level's codeword: the inner
            // product of the generator row with the next polynomial. Batch those claims into
            // the weight with powers of `beta`.
            let mut coefficients = Vec::with_capacity(queried.len());
            let mut power = beta;
            for _ in &queried {
                coefficients.push(power);
                power *= beta;
            }
            let code = &level.code;
            let rows = queried
                .par_iter()
                .zip(coefficients.par_iter())
                .fold(
                    || vec![Gf128::ZERO; weights.len()],
                    |mut acc, (&row, &coefficient)| {
                        let generator_row = tensor_expand(&code.row_factors(row));
                        for (a, g) in acc.iter_mut().zip(&generator_row) {
                            *a += coefficient * *g;
                        }
                        acc
                    },
                )
                .reduce(
                    || vec![Gf128::ZERO; weights.len()],
                    |mut a, b| {
                        for (x, y) in a.iter_mut().zip(&b) {
                            *x += *y;
                        }
                        a
                    },
                );
            for (w, row) in weights.iter_mut().zip(&rows) {
                *w += *row;
            }
            later = Some(next);
        }
    }
    proof
}

/// A sumcheck round for `Σ weights·values` over the first variable: with
/// `h(t) = a + b·t + c·t²`, the pair `(a, c)`. The verifier recovers `b` from the claim.
fn round_message(values: &[Gf128], weights: &[Gf128]) -> [Gf128; 2] {
    let (constant, leading) = values
        .par_chunks_exact(2)
        .zip(weights.par_chunks_exact(2))
        .map(|(v, w)| (v[0] * w[0], (v[0] + v[1]) * (w[0] + w[1])))
        .reduce(|| (Gf128::ZERO, Gf128::ZERO), |x, y| (x.0 + y.0, x.1 + y.1));
    [constant, leading]
}

fn grind<T: Transcript>(transcript: &mut T, bits: u32) {
    #[allow(clippy::expect_used)]
    transcript
        .pow(bits)
        .expect("the prover's proof of work cannot fail");
}
