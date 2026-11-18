//! Ligerito (Novakovic and Angeris, ePrint 2025/1187): a commitment to a multilinear polynomial
//! over `F_{2^128}` and a proof of one inner-product claim `Σ_b w(b)·f(b) = α` against it.
//!
//! The polynomial's values are a row-major matrix: the first `k'` variables index the columns
//! and the rest the rows. The columns are Reed–Solomon encoded and the rows of the encoded
//! matrix committed in a Merkle tree. A sumcheck over the column variables folds the matrix to
//! one column, which is committed the same way at the next level; the verifier then opens
//! random rows of the level before, folds them with the same challenges, and adds "this symbol
//! of the next level's codeword is that value" to the running sumcheck claim. The last fold is
//! sent in the clear.
//!
//! Soundness is in the unique-decoding regime: `queries` is chosen so that a word farther than
//! half the distance from the code survives with probability below the target, and each
//! challenge is preceded by proof of work covering the terms that are `O(block length / 2^128)`.

use super::gf128::Gf128;
use super::merkle::Digest;
use common::impl_canonical_struct;

pub mod prover;
pub mod verifier;

/// The shape and security parameters of one opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// The number of variables of the committed polynomial.
    pub num_vars: usize,
    /// The number of variables folded at each committed level.
    pub folds: Vec<usize>,
    /// `log2` of the inverse code rate at every level.
    pub log_inv_rate: usize,
    /// The number of rows opened at each level.
    pub queries: usize,
    /// The security target in bits.
    pub target_bits: u32,
}

/// The largest fold of the first level, and of every later level.
const FIRST_FOLD: usize = 6;
const LATER_FOLD: usize = 4;
/// Folding stops once at most this many variables remain; they are sent in the clear.
const MAX_FINAL_VARS: usize = 6;
/// The fewest variables left after the first level, so its code has a message to encode.
const MIN_ROW_VARS: usize = 3;

impl Schedule {
    /// The schedule for a polynomial of `num_vars ≥ 4` variables at rate 1/2.
    pub fn new(num_vars: usize, target_bits: u32) -> Option<Self> {
        if num_vars <= MIN_ROW_VARS {
            return None;
        }
        let log_inv_rate = 1;
        let mut folds = vec![FIRST_FOLD.min(num_vars - MIN_ROW_VARS)];
        let mut remaining = num_vars - folds[0];
        while remaining > MAX_FINAL_VARS {
            let fold = LATER_FOLD.min(remaining - LATER_FOLD);
            folds.push(fold);
            remaining -= fold;
        }
        let mut schedule = Self {
            num_vars,
            folds,
            log_inv_rate,
            queries: 0,
            target_bits,
        };
        let level_bits = (schedule.levels() as f64).log2();
        schedule.queries =
            ((target_bits as f64 + level_bits) / schedule.per_query_bits()).ceil() as usize;
        Some(schedule)
    }

    /// A word outside the unique-decoding radius passes one query with probability at most
    /// `(1 + rate) / 2`: `−log2` of that.
    fn per_query_bits(&self) -> f64 {
        -((1.0 + 0.5f64.powi(self.log_inv_rate as i32)) / 2.0).log2()
    }

    /// `−log2` of the probability that some level's queries all miss a word outside the
    /// unique-decoding radius, by a union bound over the levels.
    pub fn query_error_bits(&self) -> f64 {
        self.queries as f64 * self.per_query_bits() - (self.levels() as f64).log2()
    }

    pub fn levels(&self) -> usize {
        self.folds.len()
    }

    /// The number of row variables of level `level`: those left after its fold.
    pub fn row_vars(&self, level: usize) -> Option<usize> {
        let folded: usize = self.folds.get(..=level)?.iter().sum();
        self.num_vars.checked_sub(folded)
    }

    /// The number of variables sent in the clear after the last level.
    pub fn final_vars(&self) -> usize {
        self.num_vars - self.folds.iter().sum::<usize>()
    }

    /// `log2` of the block length of level `level`.
    pub fn log_block(&self, level: usize) -> Option<usize> {
        Some(self.row_vars(level)? + self.log_inv_rate)
    }

    /// The proof-of-work bits before each sumcheck challenge of `level`. A round's error is at
    /// most `(block length + 2) / 2^128`: the degree-2 sumcheck round and the proximity gap of
    /// folding the interleaved word by one more variable.
    pub fn round_pow_bits(&self, level: usize) -> Option<u32> {
        let field_bits = 128u32;
        let error_bits = self.log_block(level)? as u32 + 1;
        Some((self.target_bits + error_bits).saturating_sub(field_bits))
    }

    /// The proof-of-work bits before a batching challenge, whose error is at most
    /// `queries / 2^128`.
    pub fn batch_pow_bits(&self) -> u32 {
        let field_bits = 128u32;
        let error_bits = usize::BITS - self.queries.leading_zeros();
        (self.target_bits + error_bits).saturating_sub(field_bits)
    }
}

/// One opened row of a level's encoded matrix with its Merkle path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowOpening {
    pub row: Vec<Gf128>,
    pub path: Vec<Digest>,
}

impl_canonical_struct!([] RowOpening { row, path });

/// A Ligerito opening proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    /// The Merkle roots of the levels after the first, whose root is the commitment.
    pub roots: Vec<Digest>,
    /// Each sumcheck round's message `(h(0), leading coefficient)`, all levels in order.
    pub rounds: Vec<[Gf128; 2]>,
    /// The last fold, in the clear.
    pub final_values: Vec<Gf128>,
    /// The opened rows of each level, in query order.
    pub openings: Vec<Vec<RowOpening>>,
    /// The proof-of-work nonces, in the order the challenges they guard are drawn: one per
    /// sumcheck round and one per batching challenge.
    pub nonces: Vec<u64>,
}

impl_canonical_struct!([] Proof { roots, rounds, final_values, openings, nonces });

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitz::multilinear::{eq_table, inner_product};
    use atlas_algebra::serialize::{from_bytes, to_bytes};
    use atlas_algebra::utils::errors::ProofVerifyError;
    use atlas_transcript::{
        Blake2bProver as Prover, Blake2bVerifier as Verifier, ProvingTranscript, Transcript,
        VerifyingTranscript,
    };
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    fn random(rng: &mut StdRng) -> Gf128 {
        Gf128(((rng.next_u64() as u128) << 64) | rng.next_u64() as u128)
    }

    struct Instance {
        schedule: Schedule,
        root: Digest,
        point: Vec<Gf128>,
        claim: Gf128,
        proof: Proof,
        narg: Vec<u8>,
    }

    /// Commits to a random polynomial and proves its evaluation at a random point, the weight
    /// being the `eq` table of the point.
    fn instance(num_vars: usize, seed: u64) -> Instance {
        let mut rng = StdRng::seed_from_u64(seed);
        let schedule = Schedule::new(num_vars, 128).unwrap();
        let values: Vec<Gf128> = (0..1usize << num_vars).map(|_| random(&mut rng)).collect();
        let point: Vec<Gf128> = (0..num_vars).map(|_| random(&mut rng)).collect();
        let weights = eq_table(&point);
        let claim = inner_product(&weights, &values);
        let (root, level) = prover::commit(&values, &schedule);
        let mut transcript = Prover::new(b"ligerito-test");
        transcript.append_bytes(&root);
        let proof = prover::prove(&values, weights, &level, &schedule, &mut transcript);
        Instance {
            schedule,
            root,
            point,
            claim,
            proof,
            narg: transcript.into_narg(),
        }
    }

    fn check(
        instance: &Instance,
        root: &Digest,
        claim: Gf128,
        proof: &Proof,
    ) -> Result<(), ProofVerifyError> {
        let mut transcript = Verifier::new(b"ligerito-test", instance.narg.clone());
        transcript.append_bytes(root);
        let point = &instance.point;
        let tail = |challenges: &[Gf128]| {
            // eq(point, (challenges, z)) = eq(point_low, challenges) · eq(point_high, z).
            let (low, high) = point.split_at(challenges.len());
            let scalar = low
                .iter()
                .zip(challenges)
                .fold(Gf128::ONE, |acc, (&p, &r)| {
                    acc * (p * r + (Gf128::ONE + p) * (Gf128::ONE + r))
                });
            Ok(eq_table(high).into_iter().map(|w| scalar * w).collect())
        };
        verifier::verify(
            root,
            claim,
            tail,
            proof,
            &instance.schedule,
            &mut transcript,
        )
    }

    #[test]
    fn schedules_fold_down_to_a_short_tail() {
        for num_vars in 4..=24 {
            let schedule = Schedule::new(num_vars, 128).unwrap();
            assert!(schedule.final_vars() >= 1 && schedule.final_vars() <= MAX_FINAL_VARS);
            assert!(schedule.folds.iter().all(|&fold| fold >= 1));
            assert_eq!(
                schedule.folds.iter().sum::<usize>() + schedule.final_vars(),
                num_vars
            );
            // (3/4)^queries ≤ 2^-128 / levels.
            let bits = schedule.queries as f64 * -(0.75f64.log2());
            assert!(bits >= 128.0 + (schedule.levels() as f64).log2());
        }
        assert!(Schedule::new(3, 128).is_none());
        assert_eq!(Schedule::new(22, 128).unwrap().folds, vec![6, 4, 4, 4]);
    }

    #[test]
    fn evaluation_claims_verify() {
        for num_vars in [4, 5, 7, 10, 13] {
            let instance = instance(num_vars, num_vars as u64);
            check(&instance, &instance.root, instance.claim, &instance.proof)
                .unwrap_or_else(|e| panic!("num_vars = {num_vars}: {e}"));
            let bytes = to_bytes(&instance.proof).unwrap();
            assert_eq!(from_bytes::<Proof>(&bytes).unwrap(), instance.proof);
        }
    }

    #[test]
    fn wrong_statements_are_rejected() {
        let instance = instance(10, 1);
        assert!(check(
            &instance,
            &instance.root,
            instance.claim + Gf128::ONE,
            &instance.proof
        )
        .is_err());
        let mut root = instance.root;
        root[0] ^= 1;
        assert!(check(&instance, &root, instance.claim, &instance.proof).is_err());
    }

    #[test]
    fn every_proof_field_is_checked() {
        let instance = instance(13, 2);
        let accepts =
            |proof: &Proof| check(&instance, &instance.root, instance.claim, proof).is_ok();
        assert!(accepts(&instance.proof));
        let levels = instance.schedule.levels();
        assert!(levels >= 2, "the instance has a recursion level");

        for i in 0..instance.proof.roots.len() {
            let mut proof = instance.proof.clone();
            proof.roots[i][5] ^= 1;
            assert!(!accepts(&proof), "root {i}");
        }
        for i in 0..instance.proof.rounds.len() {
            for part in 0..2 {
                let mut proof = instance.proof.clone();
                proof.rounds[i][part] += Gf128::ONE;
                assert!(!accepts(&proof), "round {i} part {part}");
            }
        }
        for i in 0..instance.proof.final_values.len() {
            let mut proof = instance.proof.clone();
            proof.final_values[i] += Gf128::ONE;
            assert!(!accepts(&proof), "final value {i}");
        }
        for level in 0..levels {
            for query in [0, instance.schedule.queries - 1] {
                let mut proof = instance.proof.clone();
                proof.openings[level][query].row[0] += Gf128::ONE;
                assert!(!accepts(&proof), "row at level {level} query {query}");
                let mut proof = instance.proof.clone();
                proof.openings[level][query].path[0][0] ^= 1;
                assert!(!accepts(&proof), "path at level {level} query {query}");
            }
        }
        // Lengths.
        let mut proof = instance.proof.clone();
        proof.rounds.pop();
        assert!(!accepts(&proof));
        let mut proof = instance.proof.clone();
        proof.rounds.push([Gf128::ZERO; 2]);
        assert!(!accepts(&proof));
        let mut proof = instance.proof.clone();
        proof.final_values.push(Gf128::ZERO);
        assert!(!accepts(&proof));
        let mut proof = instance.proof.clone();
        proof.openings[0].pop();
        assert!(!accepts(&proof));
        let mut proof = instance.proof.clone();
        proof.openings[0][0].row.pop();
        assert!(!accepts(&proof));
        let mut proof = instance.proof.clone();
        proof.roots.push([0; 32]);
        assert!(!accepts(&proof));
    }
}
