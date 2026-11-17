//! Ring switching (Diamond and Posen, ePrint 2024/504, §3): an evaluation claim on a
//! multilinear polynomial with bit values becomes an inner-product claim on its packing.
//!
//! The bits `f` are packed 128 to an element, `t'(w) = Σ_v f(v, w)·β_v` with `β_v = x^v`, so
//! the low seven variables `v` of a bit's index sit inside an element. For a claim
//! `f(r_low, r_high) = a` the prover sends the columns `ŝ_v = Σ_w eq(r_high, w)·f(v, w)`, which
//! the verifier checks against `a`. Together they are the element `ŝ = Σ_v ŝ_v ⊗ β_v` of the
//! tensor algebra `L ⊗ L`. Written instead as `Σ_u β_u ⊗ ŝ^row_u`, its rows are
//! `ŝ^row_u = Σ_w e_u(w)·t'(w)`, where `e_u(w)` is coordinate `u` of `eq(r_high, w)`. Batching
//! the rows with `eq(u, r'')` for a random `r''` gives one claim `Σ_w A(w)·t'(w) = s_0` with
//! `A(w) = Σ_u eq(u, r'')·e_u(w)`.
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::unimplemented,
    clippy::todo,
    clippy::indexing_slicing
)]

use super::gf128::Gf128;
use super::multilinear::eq_table;
use super::pack::LOG_PACKING;
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_transcript::Transcript;

pub mod prover;

/// The bits per packed element.
const PACKING: usize = 1 << LOG_PACKING;

/// The batching challenge's error is `LOG_PACKING / 2^128`: the proof-of-work bits that bring
/// it to `target_bits`.
fn batch_pow_bits(target_bits: u32) -> u32 {
    let error_bits = usize::BITS - LOG_PACKING.leading_zeros();
    (target_bits + error_bits).saturating_sub(128)
}

/// Absorbs the columns and draws the batching point `r''`, returning `eq(·, r'')`.
fn draw_batching<T: Transcript>(
    columns: &[Gf128],
    target_bits: u32,
    transcript: &mut T,
) -> Result<Vec<Gf128>, ProofVerifyError> {
    transcript.append_serializable(&columns.to_vec());
    transcript.pow(batch_pow_bits(target_bits))?;
    let batching: Vec<Gf128> = (0..LOG_PACKING)
        .map(|_| Gf128(transcript.challenge_u128()))
        .collect();
    Ok(eq_table(&batching))
}

/// The rows of the tensor-algebra element whose columns are given: row `u` holds, at
/// coordinate `v`, coordinate `u` of column `v`.
fn transpose(columns: &[Gf128]) -> Vec<Gf128> {
    (0..PACKING)
        .map(|u| {
            let bits = columns
                .iter()
                .enumerate()
                .fold(0u128, |row, (v, column)| row | ((column.0 >> u) & 1) << v);
            Gf128(bits)
        })
        .collect()
}

/// The verifier's handle on the weight `A`: enough to evaluate it at the point a sumcheck
/// ends on.
pub struct Weight {
    /// `eq(u, r'')` for the 128 coordinates `u`.
    batching: Vec<Gf128>,
    /// `r_high`, the claim's coordinates on the packed polynomial's variables.
    high: Vec<Gf128>,
}

/// The verifier's side: checks the columns against the claim `f(point) = claim` and returns
/// the inner-product claim `s_0` on the packed polynomial together with its weight.
pub fn reduce<T: Transcript>(
    columns: &[Gf128],
    point: &[Gf128],
    claim: Gf128,
    target_bits: u32,
    transcript: &mut T,
) -> Result<(Gf128, Weight), ProofVerifyError> {
    if columns.len() != PACKING {
        return Err(ProofVerifyError::Malformed(
            "ring switching: column count".into(),
        ));
    }
    if point.len() < LOG_PACKING {
        return Err(ProofVerifyError::Malformed(
            "ring switching: point shorter than a packed element".into(),
        ));
    }
    let (low, high) = point.split_at(LOG_PACKING);
    let combined = eq_table(low)
        .iter()
        .zip(columns)
        .fold(Gf128::ZERO, |acc, (&w, &column)| acc + w * column);
    if combined != claim {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "ring switching: the columns do not combine to the claim".into(),
        ));
    }
    let batching = draw_batching(columns, target_bits, transcript)?;
    let reduced = batching
        .iter()
        .zip(&transpose(columns))
        .fold(Gf128::ZERO, |acc, (&w, &row)| acc + w * row);
    Ok((
        reduced,
        Weight {
            batching,
            high: high.to_vec(),
        },
    ))
}

/// An element of `L ⊗ L` as its rows: `Σ_u β_u ⊗ rows_u`.
struct Tensor {
    rows: Vec<Gf128>,
}

impl Tensor {
    /// `1 ⊗ 1`.
    fn one() -> Self {
        let mut rows = vec![Gf128::ZERO; PACKING];
        if let Some(first) = rows.first_mut() {
            *first = Gf128::ONE;
        }
        Self { rows }
    }

    /// `self · (left ⊗ right)`: row `v` of the product is `right` times the sum of the rows
    /// `u` for which `β_u · left` has coordinate `v`.
    fn times(&self, left: Gf128, right: Gf128) -> Self {
        let mut rows = vec![Gf128::ZERO; PACKING];
        let mut shifted = left;
        for &row in &self.rows {
            if !row.is_zero() {
                for (v, out) in rows.iter_mut().enumerate() {
                    if (shifted.0 >> v) & 1 == 1 {
                        *out += row;
                    }
                }
            }
            // β_{u+1} · left = x · (β_u · left).
            shifted *= Gf128(2);
        }
        for row in &mut rows {
            *row *= right;
        }
        Self { rows }
    }

    fn plus(mut self, other: &Self) -> Self {
        for (a, &b) in self.rows.iter_mut().zip(&other.rows) {
            *a += b;
        }
        self
    }
}

impl Weight {
    /// `A` with its first variables fixed to `challenges`, on the cube of the variables left.
    ///
    /// `A(x) = Σ_u eq(u, r'')·E_u(x)` for the rows `E_u` of
    /// `Π_i ((1 + r_i) ⊗ (1 + x_i) + r_i ⊗ x_i)`, the tensor-algebra form of `eq(r_high, x)`.
    pub fn tail(&self, challenges: &[Gf128]) -> Result<Vec<Gf128>, ProofVerifyError> {
        if challenges.len() > self.high.len() {
            return Err(ProofVerifyError::Malformed(
                "ring switching: more challenges than variables".into(),
            ));
        }
        let (folded, free) = self.high.split_at(challenges.len());
        let prefix = folded
            .iter()
            .zip(challenges)
            .fold(Tensor::one(), |tensor, (&r, &x)| {
                tensor
                    .times(Gf128::ONE + r, Gf128::ONE + x)
                    .plus(&tensor.times(r, x))
            });
        // On a cube point of the free variables the remaining factors are `eq(r_free, z) ⊗ 1`.
        Ok(eq_table(free)
            .iter()
            .map(|&scalar| {
                prefix
                    .times(scalar, Gf128::ONE)
                    .rows
                    .iter()
                    .zip(&self.batching)
                    .fold(Gf128::ZERO, |acc, (&row, &w)| acc + w * row)
            })
            .collect())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]
mod tests {
    use super::*;
    use crate::bitz::ligerito::{self, Schedule};
    use crate::bitz::multilinear::{fold_first, inner_product};
    use crate::bitz::pack::{bit, pack};
    use atlas_transcript::{
        Blake2bProver, Blake2bVerifier, ProvingTranscript, VerifyingTranscript,
    };
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    fn random(rng: &mut StdRng) -> Gf128 {
        Gf128(((rng.next_u64() as u128) << 64) | rng.next_u64() as u128)
    }

    /// The multilinear extension of the bit vector at `point`, one bit at a time.
    fn evaluate_bits(words: &[u64], point: &[Gf128]) -> Gf128 {
        (0..1usize << point.len())
            .filter(|&b| bit(words, b))
            .fold(Gf128::ZERO, |acc, b| {
                acc + point.iter().enumerate().fold(Gf128::ONE, |w, (j, &p)| {
                    w * if (b >> j) & 1 == 1 { p } else { Gf128::ONE + p }
                })
            })
    }

    fn setup(n: usize, seed: u64) -> (Vec<u64>, Vec<Gf128>, Gf128) {
        let mut rng = StdRng::seed_from_u64(seed);
        let words: Vec<u64> = (0..1usize << (n - 6)).map(|_| rng.next_u64()).collect();
        let point: Vec<Gf128> = (0..n).map(|_| random(&mut rng)).collect();
        let claim = evaluate_bits(&words, &point);
        (words, point, claim)
    }

    #[test]
    fn the_reduced_claim_is_the_inner_product_with_the_weight() {
        let n = 12;
        let (words, point, claim) = setup(n, 3);
        let packed = pack(&words);
        let mut pt = Blake2bProver::new(b"ring-switch-test");
        let (columns, weights, reduced) = prover::reduce(&packed, &point, 128, &mut pt);
        assert_eq!(inner_product(&weights, &packed), reduced);

        let mut vt = Blake2bVerifier::new(b"ring-switch-test", pt.into_narg());
        let (verifier_reduced, weight) = reduce(&columns, &point, claim, 128, &mut vt).unwrap();
        assert_eq!(verifier_reduced, reduced);

        // The verifier's tail of the weight is the prover's weight folded by the same
        // challenges, for every split of the variables.
        let mut rng = StdRng::seed_from_u64(11);
        let challenges: Vec<Gf128> = (0..n - LOG_PACKING).map(|_| random(&mut rng)).collect();
        for split in 0..=challenges.len() {
            let folded = challenges[..split]
                .iter()
                .fold(weights.clone(), |w, &r| fold_first(&w, r));
            assert_eq!(
                weight.tail(&challenges[..split]).unwrap(),
                folded,
                "split {split}"
            );
        }
        assert!(weight.tail(&[Gf128::ONE; 6]).is_err());
    }

    #[test]
    fn wrong_claims_and_columns_are_rejected() {
        let (words, point, claim) = setup(12, 4);
        let packed = pack(&words);
        let mut pt = Blake2bProver::new(b"ring-switch-test");
        let (columns, _, _) = prover::reduce(&packed, &point, 128, &mut pt);
        let narg = pt.into_narg();
        let run = |columns: &[Gf128], claim: Gf128| {
            let mut vt = Blake2bVerifier::new(b"ring-switch-test", narg.clone());
            reduce(columns, &point, claim, 128, &mut vt).map(|(reduced, _)| reduced)
        };
        let honest = run(&columns, claim).unwrap();
        assert!(run(&columns, claim + Gf128::ONE).is_err());
        assert!(run(&columns[1..], claim).is_err());
        // Columns that still combine to the claim but are not the prover's are rejected at the
        // proof of work or change the reduced claim, which the opening then fails.
        let weights = eq_table(&point[..LOG_PACKING]);
        let mut forged = columns.clone();
        forged[0] += weights[1];
        forged[1] += weights[0];
        assert!(run(&forged, claim).map_or(true, |reduced| reduced != honest));
    }

    #[test]
    fn bit_evaluation_claims_open_against_the_packed_commitment() {
        for n in [11, 12, 14, 17] {
            let (words, point, claim) = setup(n, n as u64);
            let packed = pack(&words);
            let schedule = Schedule::new(n - LOG_PACKING, 128).unwrap();
            let (root, level) = ligerito::prover::commit(&packed, &schedule);

            let mut pt = Blake2bProver::new(b"ring-switch-test");
            pt.append_bytes(&root);
            let (columns, weights, _) = prover::reduce(&packed, &point, 128, &mut pt);
            let proof = ligerito::prover::prove(packed, weights, &level, &schedule, &mut pt);
            let narg = pt.into_narg();

            let check = |claim: Gf128| {
                let mut vt = Blake2bVerifier::new(b"ring-switch-test", narg.clone());
                vt.append_bytes(&root);
                let (reduced, weight) = reduce(&columns, &point, claim, 128, &mut vt)?;
                ligerito::verifier::verify(
                    &root,
                    reduced,
                    |challenges| weight.tail(challenges),
                    &proof,
                    &schedule,
                    &mut vt,
                )
            };
            check(claim).unwrap_or_else(|e| panic!("n = {n}: {e}"));
            assert!(check(claim + Gf128::ONE).is_err());
        }
    }
}
