use ark_bn254::Fr;
use ark_ff::One;
use itertools::Itertools;
use joltworks::transcripts::Blake2bTranscript;
use qwen3_prover::{
    layer::EvalClaim,
    ops::add::{AddParams, AddProverInput, AddWitness, prove_add},
};
use qwen3_verifier::ops::add::verify_add;

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
                    .map(|col| eq_eval(col, col_point) * Fr::from(values[col * rows + row]))
                    .sum::<Fr>()
        })
        .sum()
}

#[test]
fn proves_add_without_sumcheck() {
    let params = AddParams::new(2, 2).unwrap();
    let lhs = vec![3_i32, -5, 7, 11];
    let rhs = vec![13_i32, 17, -19, 23];
    let output = lhs
        .iter()
        .zip_eq(&rhs)
        .map(|(lhs, rhs)| lhs + rhs)
        .collect::<Vec<_>>();
    let point = vec![Fr::from(2_u64), Fr::from(5_u64)];
    let output_claim = EvalClaim {
        value: eval_i32(&output, 2, 2, &point),
        point: point.clone(),
    };

    let mut prover_transcript = Blake2bTranscript::default();
    let proof = prove_add(
        output_claim.clone(),
        AddProverInput {
            params,
            witness: AddWitness {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
            },
        },
        &mut prover_transcript,
    )
    .expect("honest add claim splits into lhs and rhs claims");

    assert_eq!(proof.lhs_claim.point, point);
    assert_eq!(proof.rhs_claim.point, point);
    assert_eq!(proof.lhs_claim.value, eval_i32(&lhs, 2, 2, &point));
    assert_eq!(proof.rhs_claim.value, eval_i32(&rhs, 2, 2, &point));
    let mut transcript = Blake2bTranscript::default();
    verify_add(output_claim, &proof.proof, &mut transcript).expect("honest add proof verifies");
}

#[test]
fn rejects_inconsistent_add_claim() {
    let params = AddParams::new(2, 2).unwrap();
    let point = vec![Fr::from(2_u64), Fr::from(5_u64)];
    let mut transcript = Blake2bTranscript::default();
    let proof = prove_add(
        EvalClaim {
            value: Fr::from(123_u64),
            point,
        },
        AddProverInput {
            params,
            witness: AddWitness {
                lhs: vec![3_i32, -5, 7, 11],
                rhs: vec![13_i32, 17, -19, 23],
            },
        },
        &mut transcript,
    );

    assert!(proof.is_none());
}
