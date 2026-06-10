use ark_bn254::Fr;
use ark_ff::One;
use joltworks::field::JoltField;
use joltworks::transcripts::Blake2bTranscript;
use qwen3_prover::{
    layer::EvalClaim,
    ops::softmax::{
        SoftmaxAdvice, SoftmaxParams, SoftmaxProverInput, SoftmaxWitness, prove_softmax,
    },
};
use qwen3_verifier::ops::softmax::verify_softmax;

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

fn zero_bits(len: usize) -> [Vec<bool>; 8] {
    std::array::from_fn(|_| vec![false; len])
}

#[test]
fn proves_and_verifies_softmax() {
    let rows = 2;
    let cols = 2;
    let params = SoftmaxParams::new(rows, cols).unwrap();

    let input = vec![100_i32, 200, -999, 200];
    let row_max = vec![100_i32, 200];
    let sum = vec![256_i32, 512];
    let output = vec![256_i32, 128, 0, 128];
    let exp = vec![256_i32; rows * cols];
    let exp_acc = vec![65_536_i64; rows * cols];
    let acc = vec![16_777_216_i64, 8_388_608, 0, 8_388_608];
    let floor = vec![65_536_i32, 32_768, 0, 32_768];
    let point = vec![Fr::from(2_u64), Fr::from(5_u64)];
    let claim = EvalClaim::new(eval_matrix(&output, rows, cols, &point), point);

    let mut prover_transcript = Blake2bTranscript::default();
    let result = prove_softmax(
        claim.clone(),
        SoftmaxProverInput {
            params,
            advice: SoftmaxAdvice {
                min_diff: 0,
                max_diff: 0,
                row_max,
                max_index: vec![0; rows],
                sum: sum.clone(),
            },
            witness: SoftmaxWitness {
                input: input.clone(),
                output,
                ra: vec![1_u8; rows * cols],
                exp_acc,
                exp,
                frac_bits: zero_bits(rows * cols),
                exp_remainder_bits: zero_bits(rows * cols),
                acc,
                floor,
                floor_remainder_bits: zero_bits(rows * cols),
                output_remainder_bits: zero_bits(rows * cols),
            },
        },
        &mut prover_transcript,
    )
    .expect("honest prover creates softmax proof");

    let mut verifier_transcript = Blake2bTranscript::default();
    let verified = verify_softmax(
        claim,
        params,
        SoftmaxAdvice {
            min_diff: 0,
            max_diff: 0,
            row_max: vec![100_i32, 200],
            max_index: vec![0; rows],
            sum,
        },
        &result.proof,
        &mut verifier_transcript,
    )
    .expect("honest softmax proof verifies");

    assert_eq!(
        verified.input.value,
        eval_matrix(&input, rows, cols, &verified.input.point)
    );
}
