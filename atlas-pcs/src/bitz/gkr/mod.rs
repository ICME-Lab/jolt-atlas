//! The batched grand product of BitZ (ePrint 2026/2141, §2.3 and §6), by GKR.
//!
//! The bits form a matrix of `2^t` rows and `2^s` columns, bit `(b, c)` at index `b + 2^t·c`.
//! With a public factor `y_b` per row, the statement is that every column's product
//! `Π_b (y_b if bit (b, c) else 1)` is a given output. The product tree multiplies adjacent
//! rows: layer `i` has `i` row variables, `L_i(b, c) = L_{i+1}(0, b, c)·L_{i+1}(1, b, c)` with
//! the new variable first, `L_0` is the outputs and `L_t(b, c) = 1 + (y_b + 1)·bit(b, c)`.
//!
//! A claim on `L_i` at a point is reduced to one on `L_{i+1}` by a sumcheck of
//! `eq · L_{i+1}(0, ·) · L_{i+1}(1, ·)` and a random combination of the two values it ends on.
//! A last sumcheck over the rows separates the factors from the bits, leaving one claim on
//! the multilinear extension of the bits.
//!
//! Every sumcheck round sends a cubic `h` as `(h(0), c_2, c_3)`; the linear coefficient
//! follows from `h(0) + h(1)` being the running claim.
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
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_transcript::Transcript;
use common::impl_canonical_struct;

pub mod prover;

/// A cubic sumcheck round over a field of `2^128` elements errs with probability `3 / 2^128`:
/// the proof-of-work bits that bring it to `target_bits`.
fn round_pow_bits(target_bits: u32) -> u32 {
    (target_bits + 2).saturating_sub(128)
}

/// `eq(a, b)` for two points of equal length.
fn eq_points(a: &[Gf128], b: &[Gf128]) -> Gf128 {
    a.iter()
        .zip(b)
        .fold(Gf128::ONE, |acc, (&x, &y)| acc * (Gf128::ONE + x + y))
}

/// The reduction of one layer: its sumcheck rounds and the two values of the layer below.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerProof {
    pub rounds: Vec<[Gf128; 3]>,
    pub children: [Gf128; 2],
}

impl_canonical_struct!([] LayerProof { rounds, children });

/// A grand-product proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    /// One reduction per row variable, from the outputs down.
    pub layers: Vec<LayerProof>,
    /// The sumcheck over the rows that separates the factors from the bits.
    pub leaf_rounds: Vec<[Gf128; 3]>,
    /// The multilinear extension of the bits at the point the proof ends on.
    pub bits_value: Gf128,
}

impl_canonical_struct!([] Proof { layers, leaf_rounds, bits_value });

/// Runs the verifier's side of a cubic sumcheck: returns the claim it ends on and the
/// challenges.
fn verify_rounds<T: Transcript>(
    mut claim: Gf128,
    rounds: &[[Gf128; 3]],
    target_bits: u32,
    transcript: &mut T,
) -> Result<(Gf128, Vec<Gf128>), ProofVerifyError> {
    let mut challenges = Vec::with_capacity(rounds.len());
    for message in rounds {
        let [constant, quadratic, cubic] = *message;
        transcript.append_serializable(message);
        transcript.pow(round_pow_bits(target_bits))?;
        let r = Gf128(transcript.challenge_u128());
        let linear = claim + quadratic + cubic;
        claim = constant + r * (linear + r * (quadratic + r * cubic));
        challenges.push(r);
    }
    Ok((claim, challenges))
}

/// Draws the point the outputs are first evaluated at, an error of `log_columns / 2^128`.
fn draw_output_point<T: Transcript>(
    log_columns: usize,
    target_bits: u32,
    transcript: &mut T,
) -> Result<Vec<Gf128>, ProofVerifyError> {
    let error_bits = usize::BITS - log_columns.leading_zeros();
    transcript.pow((target_bits + error_bits).saturating_sub(128))?;
    Ok((0..log_columns)
        .map(|_| Gf128(transcript.challenge_u128()))
        .collect())
}

/// Draws the challenge that combines a layer's two values into one claim.
fn draw_combine<T: Transcript>(
    target_bits: u32,
    transcript: &mut T,
) -> Result<Gf128, ProofVerifyError> {
    transcript.pow(round_pow_bits(target_bits))?;
    Ok(Gf128(transcript.challenge_u128()))
}

/// Verifies that the columns' products are `outputs`, for row factors given as
/// `factors_plus_one[b] = y_b + 1`. Returns the point and value of the claim left on the
/// multilinear extension of the bits; the point's first `t` coordinates are the row variables.
///
/// The outputs must already be bound to the transcript.
pub fn verify<T: Transcript>(
    outputs: &[Gf128],
    factors_plus_one: &[Gf128],
    proof: &Proof,
    target_bits: u32,
    transcript: &mut T,
) -> Result<(Vec<Gf128>, Gf128), ProofVerifyError> {
    let malformed = |what: &str| ProofVerifyError::Malformed(format!("grand product: {what}"));
    if !outputs.len().is_power_of_two() || !factors_plus_one.len().is_power_of_two() {
        return Err(malformed("shape"));
    }
    let log_columns = outputs.len().trailing_zeros() as usize;
    let log_rows = factors_plus_one.len().trailing_zeros() as usize;
    if proof.layers.len() != log_rows || proof.leaf_rounds.len() != log_rows {
        return Err(malformed("layer count"));
    }

    let mut point = draw_output_point(log_columns, target_bits, transcript)?;
    let mut claim = eq_table(&point)
        .iter()
        .zip(outputs)
        .fold(Gf128::ZERO, |acc, (&w, &y)| acc + w * y);

    for (row_vars, layer) in proof.layers.iter().enumerate() {
        if layer.rounds.len() != row_vars + log_columns {
            return Err(malformed("round count"));
        }
        let (reduced, challenges) = verify_rounds(claim, &layer.rounds, target_bits, transcript)?;
        let [low, high] = layer.children;
        transcript.append_serializable(&layer.children);
        if reduced != eq_points(&point, &challenges) * low * high {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }
        let combine = draw_combine(target_bits, transcript)?;
        claim = low + combine * (low + high);
        point = std::iter::once(combine).chain(challenges).collect();
    }

    // L_t(b, c) = 1 + (y_b + 1)·bit(b, c): a sumcheck over the rows of eq · (y + 1) · bits.
    let (rows, columns) = point
        .split_at_checked(log_rows)
        .ok_or_else(|| malformed("point"))?;
    let (reduced, challenges) = verify_rounds(
        claim + Gf128::ONE,
        &proof.leaf_rounds,
        target_bits,
        transcript,
    )?;
    transcript.append_serializable(&proof.bits_value);
    let factor = eq_table(&challenges)
        .iter()
        .zip(factors_plus_one)
        .fold(Gf128::ZERO, |acc, (&w, &y)| acc + w * y);
    if reduced != eq_points(rows, &challenges) * factor * proof.bits_value {
        return Err(ProofVerifyError::SumcheckVerificationError);
    }
    let bits_point = challenges.iter().chain(columns).copied().collect();
    Ok((bits_point, proof.bits_value))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]
mod tests {
    use super::*;
    use crate::bitz::pack::bit;
    use atlas_transcript::{
        Blake2bProver, Blake2bVerifier, ProvingTranscript, VerifyingTranscript,
    };
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    fn random(rng: &mut StdRng) -> Gf128 {
        Gf128(((rng.next_u64() as u128) << 64) | rng.next_u64() as u128)
    }

    struct Instance {
        words: Vec<u64>,
        factors_plus_one: Vec<Gf128>,
        outputs: Vec<Gf128>,
        proof: Proof,
        narg: Vec<u8>,
    }

    fn instance(log_rows: usize, log_columns: usize, seed: u64) -> Instance {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = log_rows + log_columns;
        let words: Vec<u64> = (0..1usize << (n - 6)).map(|_| rng.next_u64()).collect();
        let factors: Vec<Gf128> = (0..1usize << log_rows).map(|_| random(&mut rng)).collect();
        // Each column's product, one row at a time.
        let outputs: Vec<Gf128> = (0..1usize << log_columns)
            .map(|c| {
                (0..1usize << log_rows)
                    .filter(|b| bit(&words, b + (c << log_rows)))
                    .fold(Gf128::ONE, |acc, b| acc * factors[b])
            })
            .collect();
        let mut pt = Blake2bProver::new(b"gkr-test");
        let (proof, prover_outputs, _, _) =
            prover::prove(&words, &factors, log_columns, 128, &mut pt);
        assert_eq!(prover_outputs, outputs);
        Instance {
            words,
            factors_plus_one: factors.iter().map(|&y| y + Gf128::ONE).collect(),
            outputs,
            proof,
            narg: pt.into_narg(),
        }
    }

    fn check(
        instance: &Instance,
        outputs: &[Gf128],
        proof: &Proof,
    ) -> Result<(Vec<Gf128>, Gf128), ProofVerifyError> {
        let mut vt = Blake2bVerifier::new(b"gkr-test", instance.narg.clone());
        verify(outputs, &instance.factors_plus_one, proof, 128, &mut vt)
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

    #[test]
    fn products_verify_and_leave_a_true_claim_on_the_bits() {
        for (log_rows, log_columns) in [(7, 5), (7, 0), (8, 6), (9, 5), (10, 7)] {
            let instance = instance(log_rows, log_columns, (log_rows * 31 + log_columns) as u64);
            let (point, value) = check(&instance, &instance.outputs, &instance.proof)
                .unwrap_or_else(|e| panic!("{log_rows} x {log_columns}: {e}"));
            assert_eq!(point.len(), log_rows + log_columns);
            assert_eq!(value, evaluate_bits(&instance.words, &point));
        }
    }

    #[test]
    fn wrong_outputs_and_mutated_proofs_are_rejected() {
        let instance = instance(8, 5, 77);
        let accepts = |outputs: &[Gf128], proof: &Proof| check(&instance, outputs, proof).is_ok();
        assert!(accepts(&instance.outputs, &instance.proof));

        let mut outputs = instance.outputs.clone();
        outputs[3] += Gf128::ONE;
        assert!(!accepts(&outputs, &instance.proof));

        for layer in 0..instance.proof.layers.len() {
            for round in 0..instance.proof.layers[layer].rounds.len() {
                for part in 0..3 {
                    let mut proof = instance.proof.clone();
                    proof.layers[layer].rounds[round][part] += Gf128::ONE;
                    assert!(
                        !accepts(&instance.outputs, &proof),
                        "layer {layer} round {round} part {part}"
                    );
                }
            }
            for child in 0..2 {
                let mut proof = instance.proof.clone();
                proof.layers[layer].children[child] += Gf128::ONE;
                assert!(
                    !accepts(&instance.outputs, &proof),
                    "layer {layer} child {child}"
                );
            }
        }
        for round in 0..instance.proof.leaf_rounds.len() {
            for part in 0..3 {
                let mut proof = instance.proof.clone();
                proof.leaf_rounds[round][part] += Gf128::ONE;
                assert!(
                    !accepts(&instance.outputs, &proof),
                    "leaf round {round} part {part}"
                );
            }
        }
        let mut proof = instance.proof.clone();
        proof.bits_value += Gf128::ONE;
        assert!(!accepts(&instance.outputs, &proof));
        let mut proof = instance.proof.clone();
        proof.layers.pop();
        assert!(!accepts(&instance.outputs, &proof));
        let mut proof = instance.proof.clone();
        proof.layers[2].rounds.pop();
        assert!(!accepts(&instance.outputs, &proof));
        let mut proof = instance.proof.clone();
        proof.leaf_rounds.push([Gf128::ZERO; 3]);
        assert!(!accepts(&instance.outputs, &proof));
    }
}
