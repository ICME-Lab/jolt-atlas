use ark_bn254::Fr;
use ark_ff::One;
use joltworks::field::JoltField;
use joltworks::transcripts::Blake2bTranscript;
use qwen3_common::ops::matmul::MatMulVerifierInput;
use qwen3_prover::{
    layer::EvalClaim,
    ops::matmul::{MatMulParams, MatMulProverInput, MatMulWitness, prove_matmul},
};
use qwen3_verifier::ops::matmul::verify_matmul;

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

fn eval_matrix(values: &[i32], rows: usize, cols: usize, point: &[Fr]) -> Fr {
    let row_vars = rows.ilog2() as usize;
    let (row_point, col_point) = point.split_at(row_vars);
    (0..rows)
        .map(|row| {
            eq_eval(row, row_point)
                * (0..cols)
                    .map(|col| eq_eval(col, col_point) * Fr::from_i32(values[col * rows + row]))
                    .sum::<Fr>()
        })
        .sum()
}

fn round_q8(value: i64) -> i32 {
    let rem = value.rem_euclid(256);
    ((value + ((rem >> 7) * 256) - rem) / 256) as i32
}

fn remainders(values: &[i64]) -> Vec<u8> {
    values
        .iter()
        .map(|value| value.rem_euclid(256) as u8)
        .collect()
}

fn matmul_acc(lhs: &[i32], rhs: &[i32], rows: usize, cols: usize, inner: usize) -> Vec<i64> {
    let mut acc = vec![0_i64; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            acc[col * rows + row] = (0..inner)
                .map(|k| i64::from(lhs[k * rows + row]) * i64::from(rhs[col * inner + k]))
                .sum();
        }
    }
    acc
}

#[test]
fn proves_and_verifies_matmul_sumcheck() {
    let rows = 2;
    let cols = 2;
    let inner = 2;
    let lhs = vec![128_i32, -64, 256, 32];
    let rhs = vec![64_i32, 128, -96, 16];
    let acc = matmul_acc(&lhs, &rhs, rows, cols, inner);
    let output = acc.iter().map(|value| round_q8(*value)).collect::<Vec<_>>();
    let claim_point = vec![Fr::from(2_u64), Fr::from(5_u64)];
    let claim = eval_matrix(&output, rows, cols, &claim_point);
    let params = MatMulParams::new(rows, cols, inner).unwrap();

    let mut prover_transcript = Blake2bTranscript::default();
    let result = prove_matmul(
        EvalClaim::new(claim, claim_point.clone()),
        MatMulProverInput {
            params,
            witness: MatMulWitness {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                output,
                output_remainder: remainders(&acc),
            },
        },
        &mut prover_transcript,
    )
    .expect("honest prover creates matmul proof");

    let mut verifier_transcript = Blake2bTranscript::default();
    let verified = verify_matmul(
        EvalClaim::new(claim, claim_point),
        MatMulVerifierInput {
            params,
            weight: rhs.clone(),
        },
        &result.proof,
        &mut verifier_transcript,
    )
    .expect("honest matmul proof verifies");

    assert_eq!(
        verified.lhs.value,
        eval_matrix(&lhs, rows, inner, &verified.lhs.point)
    );
    assert_eq!(
        verified.rhs.value,
        eval_matrix(&rhs, inner, cols, &verified.rhs.point)
    );
}
