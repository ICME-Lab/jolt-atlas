use ark_bn254::Fr;
use ark_ff::One;
use itertools::Itertools;
use joltworks::field::JoltField;
use joltworks::transcripts::Blake2bTranscript;
use qwen3_common::LayerRmsNormVerifierInput;
use qwen3_prover::{
    layer::EvalClaim,
    ops::rms_norm::{
        RmsNormAdvice, RmsNormParams, RmsNormProverInput, RmsNormWitness, prove_rms_norm,
        rms_inv_from_square_sum,
    },
};
use qwen3_verifier::ops::rms_norm::verify_rms_norm;

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

fn eval_i32(values: &[i32], rows: usize, cols: usize, point: &[Fr]) -> Fr {
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

fn remainder_bits(values: &[i64]) -> [Vec<bool>; 8] {
    std::array::from_fn(|bit| {
        values
            .iter()
            .map(|value| ((value.rem_euclid(256) >> bit) & 1) == 1)
            .collect()
    })
}

#[test]
fn proves_and_verifies_rms_norm() {
    let rows = 2;
    let cols = 2;
    let input = vec![256_i32, 512, -128, 384];
    let weight = vec![64_i32, 128, 64, 128];
    let mut sum_x2 = vec![0_i64; rows];
    for row in 0..rows {
        for col in 0..cols {
            let value = i64::from(input[col * rows + row]);
            sum_x2[row] += value * value;
        }
    }
    let inv_by_row = sum_x2
        .iter()
        .map(|sum| rms_inv_from_square_sum(*sum, cols))
        .collect::<Vec<_>>();
    let mut inv_rms = Vec::with_capacity(rows * cols);
    for _col in 0..cols {
        for row in 0..rows {
            inv_rms.push(inv_by_row[row]);
        }
    }
    let norm_acc = input
        .iter()
        .zip_eq(&inv_rms)
        .map(|(input, inv)| i64::from(*input) * i64::from(*inv))
        .collect::<Vec<_>>();
    let norm = norm_acc
        .iter()
        .map(|value| round_q8(*value))
        .collect::<Vec<_>>();
    let output_acc = norm
        .iter()
        .zip_eq(&weight)
        .map(|(norm, weight)| i64::from(*norm) * i64::from(*weight))
        .collect::<Vec<_>>();
    let output = output_acc
        .iter()
        .map(|value| round_q8(*value))
        .collect::<Vec<_>>();
    let output_claims = [
        vec![Fr::from(2_u64), Fr::from(5_u64)],
        vec![Fr::from(7_u64), Fr::from(11_u64)],
    ]
    .map(|point| EvalClaim::new(eval_i32(&output, rows, cols, &point), point))
    .to_vec();
    let params = RmsNormParams::new(rows, cols).unwrap();
    let mut prover_transcript = Blake2bTranscript::default();
    let result = prove_rms_norm(
        output_claims.clone(),
        RmsNormProverInput {
            params,
            advice: RmsNormAdvice {
                sum_x2: sum_x2.clone(),
            },
            witness: RmsNormWitness {
                input: input.clone(),
                inv_rms,
                norm,
                weight: weight.clone(),
                output,
                norm_remainder_bits: remainder_bits(&norm_acc),
                output_remainder_bits: remainder_bits(&output_acc),
            },
        },
        &mut prover_transcript,
    )
    .expect("honest prover creates RMSNorm proof");

    let mut verifier_transcript = Blake2bTranscript::default();
    let verified = verify_rms_norm(
        output_claims,
        params,
        LayerRmsNormVerifierInput {
            advice: RmsNormAdvice { sum_x2 },
            weight: weight.clone(),
        },
        &result.proof,
        &mut verifier_transcript,
    )
    .expect("honest RMSNorm proof verifies");

    assert_eq!(
        verified.input.value,
        eval_i32(&input, rows, cols, &verified.input.point)
    );
}
