use ark_bn254::Fr;
use ark_ff::{Field, One};
use itertools::Itertools;
use joltworks::field::JoltField;
use joltworks::transcripts::Blake2bTranscript;
use qwen3_prover::{
    layer::EvalClaim,
    ops::mul::{MulParams, MulProverInput, MulWitness, prove_mul},
};
use qwen3_verifier::ops::mul::verify_mul;

fn eq_eval(index: usize, point: &[Fr]) -> Fr {
    point
        .iter()
        .enumerate()
        .map(|(bit, point)| {
            if ((index >> bit) & 1) == 1 {
                *point
            } else {
                Fr::one() - point
            }
        })
        .product()
}

fn zero_bool_bits(len: usize) -> [Vec<bool>; 8] {
    std::array::from_fn(|_| vec![false; len])
}

fn shift() -> Fr {
    Field::inverse(&Fr::from(256_u64)).unwrap()
}

#[test]
fn proves_and_verifies_mul_with_rounding_bits() {
    let a = vec![20_i32, 19, 27, 43];
    let b = vec![10_i32, 20, 19, 3];
    let c = a
        .iter()
        .zip_eq(&b)
        .map(|(a, b)| {
            let product = a * b;
            let remainder = product.rem_euclid(256);
            product.div_euclid(256) + i32::from(remainder >= 128)
        })
        .collect::<Vec<i32>>();
    let bits = std::array::from_fn::<_, 8, _>(|bit| {
        a.iter()
            .zip_eq(&b)
            .map(|(a, b)| ((a * b).rem_euclid(256) >> bit & 1) != 0)
            .collect::<Vec<_>>()
    });
    let claim_point = vec![Fr::from(2_u64), Fr::from(5_u64)];
    let claim = c
        .iter()
        .enumerate()
        .map(|(index, c)| eq_eval(index, &claim_point) * Fr::from_i32(*c))
        .sum::<Fr>();

    let mut prover_transcript = Blake2bTranscript::default();
    let output = prove_mul(
        EvalClaim::new(claim, claim_point.clone()),
        MulProverInput {
            params: MulParams::new(a.len()).unwrap(),
            witness: MulWitness { a, b, bits },
        },
        &mut prover_transcript,
    )
    .expect("honest prover creates rounded mul proof");

    let mut verifier_transcript = Blake2bTranscript::default();
    let verified = verify_mul(
        EvalClaim::new(claim, claim_point),
        &output.proof,
        &mut verifier_transcript,
    )
    .expect("honest rounded mul proof verifies");

    assert_eq!(verified.lhs.value, output.lhs.value);
    assert_eq!(verified.rhs.value, output.rhs.value);
}

#[test]
fn proves_and_verifies_mul_sumcheck() {
    let lhs = vec![1_i32, 3, 2, 5];
    let rhs = vec![7_i32, 11, 13, 17];
    let claim_point = vec![Fr::from(2_u64), Fr::from(5_u64)];
    let claim = lhs
        .iter()
        .zip_eq(rhs.iter())
        .enumerate()
        .map(|(index, (lhs, rhs))| {
            eq_eval(index, &claim_point) * Fr::from_i32(*lhs) * Fr::from_i32(*rhs) * shift()
        })
        .sum::<Fr>();

    let mut prover_transcript = Blake2bTranscript::default();
    let output = prove_mul(
        EvalClaim::new(claim, claim_point.clone()),
        MulProverInput {
            params: MulParams::new(lhs.len()).unwrap(),
            witness: MulWitness {
                a: lhs,
                b: rhs,
                bits: zero_bool_bits(4),
            },
        },
        &mut prover_transcript,
    )
    .expect("honest prover creates mul proof");

    let mut verifier_transcript = Blake2bTranscript::default();
    verify_mul(
        EvalClaim::new(claim, claim_point),
        &output.proof,
        &mut verifier_transcript,
    )
    .expect("honest mul proof verifies");
}
