use ark_bn254::Fr;
use ark_ff::One;
use joltworks::{
    field::JoltField,
    transcripts::{Blake2bTranscript, Transcript},
};
use qwen3_common::LayerRopeVerifierInput;
use qwen3_prover::{
    layer::EvalClaim,
    ops::rope::{RopeParams, RopeProverInput, RopeWitness, draw_rope_challenges, prove_rope},
    round_message::SumCheckRounds,
};
use qwen3_verifier::ops::rope::verify_rope;

fn verify_sumcheck_rounds(
    mut claim: Fr,
    rounds: &SumCheckRounds<4>,
    transcript: &mut Blake2bTranscript,
) -> Option<(Fr, Vec<<Fr as JoltField>::Challenge>)> {
    let mut challenges = Vec::with_capacity(rounds.round_polys.len());
    for round in &rounds.round_polys {
        if claim != round.eval(Fr::from(0_u64)) + round.eval(Fr::from(1_u64)) {
            return None;
        }
        qwen3_prover::round_message::append_round_statement(transcript, claim, round);
        let r = joltworks::transcripts::Transcript::challenge_scalar_optimized::<Fr>(transcript);
        claim = round.eval(r.into());
        challenges.push(r);
    }
    (claim == rounds.final_claim).then_some((claim, challenges))
}

fn eval_tensor(values: &[i32], params: &RopeParams, point: &[Fr]) -> Fr {
    let seq_vars = params.seq.ilog2() as usize;
    let head_vars = params.heads.ilog2() as usize;
    let (seq_point, rest) = point.split_at(seq_vars);
    let (head_point, dim_point) = rest.split_at(head_vars);
    (0..params.seq)
        .map(|seq| {
            eq_eval(seq, seq_point)
                * (0..params.heads)
                    .map(|head| {
                        eq_eval(head, head_point)
                            * (0..params.head_dim)
                                .map(|dim| {
                                    eq_eval(dim, dim_point)
                                        * Fr::from(
                                            values[dim * (params.seq * params.heads)
                                                + head * params.seq
                                                + seq],
                                        )
                                })
                                .sum::<Fr>()
                    })
                    .sum::<Fr>()
        })
        .sum()
}

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
        .fold(Fr::one(), |acc, value| acc * value)
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

fn tensor_index(seq: usize, head: usize, dim: usize, params: &RopeParams) -> usize {
    dim * (params.seq * params.heads) + head * params.seq + seq
}

fn rope_acc(input: &[i32], cos: &[i32], sin: &[i32], params: &RopeParams) -> Vec<i64> {
    let mut acc = vec![0_i64; input.len()];
    let half = params.head_dim / 2;
    for seq in 0..params.seq {
        for head in 0..params.heads {
            for pair in 0..half {
                let first = tensor_index(seq, head, pair, params);
                let second = tensor_index(seq, head, pair + half, params);
                let coeff = pair * params.seq + seq;
                acc[first] = i64::from(input[first]) * i64::from(cos[coeff])
                    - i64::from(input[second]) * i64::from(sin[coeff]);
                acc[second] = i64::from(input[second]) * i64::from(cos[coeff])
                    + i64::from(input[first]) * i64::from(sin[coeff]);
            }
        }
    }
    acc
}

#[test]
fn proves_and_verifies_rope_sumchecks() {
    let params = RopeParams::new(2, 2, 4).unwrap();
    let input = vec![
        13_i32, -7, 19, 5, 11, 23, -3, 17, 29, -31, 37, 41, -43, 47, 53, -59,
    ];
    let cos = vec![220_i32, 201, 147, 113];
    let sin = vec![91_i32, -73, 164, -128];
    let acc_i64 = rope_acc(&input, &cos, &sin, &params);
    let output = acc_i64
        .iter()
        .map(|value| round_q8(*value))
        .collect::<Vec<_>>();
    let claim_point = vec![
        Fr::from(3_u64),
        Fr::from(5_u64),
        Fr::from(7_u64),
        Fr::from(11_u64),
    ];
    let claim = eval_tensor(&output, &params, &claim_point);

    let mut prover_transcript = Blake2bTranscript::default();
    let output = prove_rope(
        EvalClaim::new(claim, claim_point.clone()),
        RopeProverInput {
            params,
            witness: RopeWitness {
                input: input.clone(),
                output: output.clone(),
                output_remainder_bits: remainder_bits(&acc_i64),
                cos: cos.clone(),
                sin: sin.clone(),
            },
        },
        &mut prover_transcript,
    )
    .expect("honest prover creates RoPE sumchecks");

    let mut verifier_transcript = Blake2bTranscript::default();
    verifier_transcript.append_scalar(&claim);
    verifier_transcript.append_scalars(&claim_point);
    draw_rope_challenges(&mut verifier_transcript).unwrap();
    let (final_claim, _) =
        verify_sumcheck_rounds(claim, &output.proof.rounds, &mut verifier_transcript)
            .expect("honest RoPE verifies");
    assert_eq!(output.proof.rounds.final_claim, final_claim);

    let _input_first_half_claim = output
        .input_first_half_claim
        .first()
        .expect("expected single eval claim");
    let _input_second_half_claim = output
        .input_second_half_claim
        .first()
        .expect("expected single eval claim");

    let mut full_verifier_transcript = Blake2bTranscript::default();
    let verified = verify_rope(
        EvalClaim::new(claim, claim_point),
        params,
        LayerRopeVerifierInput {
            cos: cos.clone(),
            sin: sin.clone(),
        },
        &output.proof,
        &mut full_verifier_transcript,
    )
    .expect("honest RoPE proof verifies through verify_rope");
    assert_eq!(
        verified.input_first_half.value,
        output.input_first_half.value
    );
    assert_eq!(
        verified.input_second_half.value,
        output.input_second_half.value
    );
}
