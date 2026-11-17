//! End-to-end tests of the BitZ scheme: every claim is over Fp128 and is checked against the
//! multilinear extension computed directly.

use super::integers::{Wide, FP128_MODULUS};
use super::pack::bit;
use super::params::{Params, MIN_VARS};
use super::proof::{Commitment, Proof};
use super::{prover, verifier};
use atlas_algebra::field::fp128::Fp128;
use atlas_algebra::serialize::{from_bytes, to_bytes};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_transcript::{Blake2bProver, Blake2bVerifier, ProvingTranscript, VerifyingTranscript};
use rand::{rngs::StdRng, RngCore, SeedableRng};

const LABEL: &[u8] = b"bitz-test";

fn random_point(num_vars: usize, rng: &mut StdRng) -> Vec<Fp128> {
    (0..num_vars)
        .map(|_| Fp128::from(((rng.next_u64() as u128) << 64) | rng.next_u64() as u128))
        .collect()
}

/// `Σ_x eq(point, x)·bits(x)` from the definition: for every set bit, the product over the
/// variables of the coordinate or its complement.
fn evaluate(words: &[u64], point: &[Fp128]) -> Fp128 {
    let one = Fp128::from(1u128);
    (0..1usize << point.len())
        .filter(|&x| bit(words, x))
        .fold(Fp128::from(0u128), |acc, x| {
            acc + point.iter().enumerate().fold(one, |weight, (j, &r)| {
                weight * if (x >> j) & 1 == 1 { r } else { one - r }
            })
        })
}

struct Instance {
    params: Params,
    words: Vec<u64>,
    commitment: Commitment,
    point: Vec<Fp128>,
    claim: Fp128,
    proof: Proof,
    narg: Vec<u8>,
}

fn instance_of(words: Vec<u64>, seed: u64) -> Instance {
    let num_vars = (words.len() * 64).trailing_zeros() as usize;
    let params = Params::new(num_vars, 128).unwrap();
    let mut rng = StdRng::seed_from_u64(seed);
    let point = random_point(num_vars, &mut rng);
    let (commitment, committed) = prover::commit(words.clone(), &params);
    let mut transcript = Blake2bProver::new(LABEL);
    let (claim, proof) = prover::prove(&commitment, committed, &point, &params, &mut transcript);
    Instance {
        params,
        words,
        commitment,
        point,
        claim,
        proof,
        narg: transcript.into_narg(),
    }
}

fn random_instance(num_vars: usize, seed: u64) -> Instance {
    let mut rng = StdRng::seed_from_u64(seed ^ 0xb172);
    let words = (0..1usize << (num_vars - 6))
        .map(|_| rng.next_u64())
        .collect();
    instance_of(words, seed)
}

impl Instance {
    fn check(
        &self,
        commitment: &Commitment,
        point: &[Fp128],
        claim: &Fp128,
        proof: &Proof,
    ) -> Result<(), ProofVerifyError> {
        self.check_with(&self.params, commitment, point, claim, proof)
    }

    fn check_with(
        &self,
        params: &Params,
        commitment: &Commitment,
        point: &[Fp128],
        claim: &Fp128,
        proof: &Proof,
    ) -> Result<(), ProofVerifyError> {
        let mut transcript = Blake2bVerifier::new(LABEL, self.narg.clone());
        verifier::verify(commitment, point, claim, proof, params, &mut transcript)
    }

    fn accepts(&self, proof: &Proof) -> bool {
        self.check(&self.commitment, &self.point, &self.claim, proof)
            .is_ok()
    }
}

#[test]
fn claims_over_fp128_verify_at_every_small_size() {
    for num_vars in MIN_VARS..=17 {
        let instance = random_instance(num_vars, num_vars as u64);
        assert_eq!(
            instance.claim,
            evaluate(&instance.words, &instance.point),
            "n = {num_vars}"
        );
        instance
            .check(
                &instance.commitment,
                &instance.point,
                &instance.claim,
                &instance.proof,
            )
            .unwrap_or_else(|e| panic!("n = {num_vars}: {e}"));
    }
}

#[test]
fn constant_vectors_verify() {
    for num_vars in [12, 15] {
        let len = 1usize << (num_vars - 6);
        for words in [vec![0u64; len], vec![u64::MAX; len]] {
            let ones = words[0] != 0;
            let instance = instance_of(words, 5);
            // Σ_x eq(point, x) = 1.
            assert_eq!(instance.claim, Fp128::from(ones as u128));
            assert!(
                instance.accepts(&instance.proof),
                "n = {num_vars}, ones = {ones}"
            );
        }
    }
}

#[test]
fn larger_sizes_verify() {
    for num_vars in 18..=24 {
        let instance = random_instance(num_vars, num_vars as u64);
        assert!(instance.accepts(&instance.proof), "n = {num_vars}");
    }
}

#[test]
fn wrong_statements_are_rejected() {
    let instance = random_instance(14, 1);
    let other = random_instance(14, 2);
    let (c, p, v, proof) = (
        &instance.commitment,
        &instance.point,
        &instance.claim,
        &instance.proof,
    );
    assert!(instance.check(c, p, v, proof).is_ok());

    // A wrong value, including one that differs by a small integer.
    assert!(instance
        .check(c, p, &(*v + Fp128::from(1u128)), proof)
        .is_err());
    assert!(instance.check(c, p, &other.claim, proof).is_err());
    // A wrong point.
    assert!(instance.check(c, &other.point, v, proof).is_err());
    let mut point = p.clone();
    point[3] += Fp128::from(1u128);
    assert!(instance.check(c, &point, v, proof).is_err());
    assert!(instance.check(c, &p[1..], v, proof).is_err());
    // A commitment to a different vector.
    assert!(instance.check(&other.commitment, p, v, proof).is_err());
    // A proof for a different size.
    let smaller = Params::new(13, 128).unwrap();
    assert!(instance.check_with(&smaller, c, &p[1..], v, proof).is_err());
    let larger = Params::new(15, 128).unwrap();
    let mut longer = p.clone();
    longer.push(Fp128::from(7u128));
    assert!(instance.check_with(&larger, c, &longer, v, proof).is_err());
}

#[test]
fn a_lifted_claim_off_by_the_field_modulus_is_rejected() {
    // Adding p to the integer claim keeps it a lift of the same field element and inside the
    // range check; only the projection onto the random prime tells it from the true integer.
    let instance = random_instance(13, 3);
    let mut proof = instance.proof.clone();
    let modulus = Wide([FP128_MODULUS as u64, (FP128_MODULUS >> 64) as u64, 0, 0, 0]);
    proof.lifted_claim = proof.lifted_claim.checked_add(&modulus).unwrap();
    assert_eq!(proof.lifted_claim.to_fp128(), instance.claim);
    assert!(!instance.accepts(&proof));
}

#[test]
fn every_proof_field_is_checked() {
    let instance = random_instance(16, 4);
    assert!(instance.accepts(&instance.proof));
    let one = super::gf128::Gf128::ONE;

    for limb in 0..5 {
        let mut proof = instance.proof.clone();
        proof.lifted_claim.0[limb] ^= 1;
        assert!(!instance.accepts(&proof), "lifted claim limb {limb}");
    }
    for column in 0..instance.proof.column_sums.len() {
        let mut proof = instance.proof.clone();
        proof.column_sums[column] ^= 1;
        assert!(!instance.accepts(&proof), "column sum {column}");
    }
    let mut proof = instance.proof.clone();
    proof.column_sums.pop();
    assert!(!instance.accepts(&proof));
    let mut proof = instance.proof.clone();
    proof.column_sums[0] = u128::MAX;
    assert!(!instance.accepts(&proof));

    for layer in 0..instance.proof.grand_product.layers.len() {
        let rounds = instance.proof.grand_product.layers[layer].rounds.len();
        for round in 0..rounds {
            for part in 0..3 {
                let mut proof = instance.proof.clone();
                proof.grand_product.layers[layer].rounds[round][part] += one;
                assert!(
                    !instance.accepts(&proof),
                    "layer {layer} round {round} part {part}"
                );
            }
        }
        for child in 0..2 {
            let mut proof = instance.proof.clone();
            proof.grand_product.layers[layer].children[child] += one;
            assert!(!instance.accepts(&proof), "layer {layer} child {child}");
        }
    }
    for round in 0..instance.proof.grand_product.leaf_rounds.len() {
        for part in 0..3 {
            let mut proof = instance.proof.clone();
            proof.grand_product.leaf_rounds[round][part] += one;
            assert!(!instance.accepts(&proof), "leaf round {round} part {part}");
        }
    }
    let mut proof = instance.proof.clone();
    proof.grand_product.bits_value += one;
    assert!(!instance.accepts(&proof));

    for column in 0..instance.proof.ring_columns.len() {
        let mut proof = instance.proof.clone();
        proof.ring_columns[column] += one;
        assert!(!instance.accepts(&proof), "ring column {column}");
    }
    let mut proof = instance.proof.clone();
    proof.ring_columns.pop();
    assert!(!instance.accepts(&proof));

    for root in 0..instance.proof.opening.roots.len() {
        let mut proof = instance.proof.clone();
        proof.opening.roots[root][0] ^= 1;
        assert!(!instance.accepts(&proof), "root {root}");
    }
    for round in 0..instance.proof.opening.rounds.len() {
        for part in 0..2 {
            let mut proof = instance.proof.clone();
            proof.opening.rounds[round][part] += one;
            assert!(
                !instance.accepts(&proof),
                "opening round {round} part {part}"
            );
        }
    }
    for index in 0..instance.proof.opening.final_values.len() {
        let mut proof = instance.proof.clone();
        proof.opening.final_values[index] += one;
        assert!(!instance.accepts(&proof), "final value {index}");
    }
    for level in 0..instance.proof.opening.openings.len() {
        for query in [0, instance.proof.opening.openings[level].len() - 1] {
            let mut proof = instance.proof.clone();
            proof.opening.openings[level][query].row[0] += one;
            assert!(
                !instance.accepts(&proof),
                "row at level {level} query {query}"
            );
            let mut proof = instance.proof.clone();
            proof.opening.openings[level][query].path[0][0] ^= 1;
            assert!(
                !instance.accepts(&proof),
                "path at level {level} query {query}"
            );
        }
    }
}

#[test]
fn proofs_have_one_encoding() {
    let instance = random_instance(13, 6);
    let bytes = to_bytes(&instance.proof).unwrap();
    assert_eq!(from_bytes::<Proof>(&bytes).unwrap(), instance.proof);
    assert_eq!(
        from_bytes::<Commitment>(&to_bytes(&instance.commitment).unwrap()).unwrap(),
        instance.commitment
    );

    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(from_bytes::<Proof>(&trailing).is_err());
    assert!(from_bytes::<Proof>(&bytes[..bytes.len() - 1]).is_err());
    // The first vector is the column sums, after the five limbs of the lifted claim: a forged
    // length fails at the end of the input.
    let mut forged = bytes.clone();
    forged[40..48].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(from_bytes::<Proof>(&forged).is_err());
    // A field element of the ring-switching columns is any 16 bytes, but a claim is not: the
    // canonical encoding of Fp128 rejects values at or above the modulus.
    assert!(from_bytes::<Fp128>(&FP128_MODULUS.to_le_bytes()).is_err());
}
