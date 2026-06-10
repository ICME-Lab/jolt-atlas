use ark_bn254::Fr;
use ark_ff::One;
use joltworks::{
    field::JoltField,
    transcripts::{Blake2bTranscript, Transcript},
};
use qwen3_prover::{
    layer::EvalClaim,
    ops::qk_score::{
        QkScoreParams, QkScoreProverInput, QkScoreWitness,
        draw_qk_score_dot_bit_booleanity_challenges, draw_qk_score_dot_challenges,
        draw_qk_score_scale_challenges, prove_qk_score, qk_score_inv_sqrt_q8,
    },
    round::verify_sumcheck_rounds,
};
use qwen3_verifier::ops::qk_score::verify_qk_score;

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

fn eval_score(values: &[i32], params: &QkScoreParams, point: &[Fr]) -> Fr {
    let q_vars = params.seq.ilog2() as usize;
    let h_vars = params.q_heads.ilog2() as usize;
    let (q_point, rest) = point.split_at(q_vars);
    let (h_point, k_point) = rest.split_at(h_vars);
    (0..params.q_heads)
        .map(|head| {
            eq_eval(head, h_point)
                * (0..params.seq)
                    .map(|qpos| {
                        eq_eval(qpos, q_point)
                            * (0..params.seq)
                                .map(|kpos| {
                                    eq_eval(kpos, k_point)
                                        * Fr::from_i32(
                                            values[kpos * (params.seq * params.q_heads)
                                                + head * params.seq
                                                + qpos],
                                        )
                                })
                                .sum::<Fr>()
                    })
                    .sum::<Fr>()
        })
        .sum()
}

fn eval_q(values: &[i32], params: &QkScoreParams, point: &[Fr]) -> Fr {
    let q_vars = params.seq.ilog2() as usize;
    let h_vars = params.q_heads.ilog2() as usize;
    let (q_point, rest) = point.split_at(q_vars);
    let (h_point, d_point) = rest.split_at(h_vars);
    (0..params.seq)
        .map(|qpos| {
            eq_eval(qpos, q_point)
                * (0..params.q_heads)
                    .map(|head| {
                        eq_eval(head, h_point)
                            * (0..params.head_dim)
                                .map(|d| {
                                    eq_eval(d, d_point)
                                        * Fr::from_i32(
                                            values[d * (params.seq * params.q_heads)
                                                + head * params.seq
                                                + qpos],
                                        )
                                })
                                .sum::<Fr>()
                    })
                    .sum::<Fr>()
        })
        .sum()
}

fn eval_k(values: &[i32], params: &QkScoreParams, point: &[Fr]) -> Fr {
    let k_vars = params.seq.ilog2() as usize;
    let kv_vars = params.kv_heads.ilog2() as usize;
    let (k_point, rest) = point.split_at(k_vars);
    let (kv_point, d_point) = rest.split_at(kv_vars);
    (0..params.seq)
        .map(|kpos| {
            eq_eval(kpos, k_point)
                * (0..params.kv_heads)
                    .map(|kv_head| {
                        eq_eval(kv_head, kv_point)
                            * (0..params.head_dim)
                                .map(|d| {
                                    eq_eval(d, d_point)
                                        * Fr::from_i32(
                                            values[d * (params.seq * params.kv_heads)
                                                + kv_head * params.seq
                                                + kpos],
                                        )
                                })
                                .sum::<Fr>()
                    })
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

fn qk_raw_acc(q: &[i32], k: &[i32], params: &QkScoreParams) -> Vec<i64> {
    let mut out = vec![0_i64; params.q_heads * params.seq * params.seq];
    for head in 0..params.q_heads {
        let kv_head = head / 2;
        for qpos in 0..params.seq {
            for kpos in 0..params.seq {
                out[kpos * (params.seq * params.q_heads) + head * params.seq + qpos] = (0..params
                    .head_dim)
                    .map(|d| {
                        let q_idx = d * (params.seq * params.q_heads) + head * params.seq + qpos;
                        let k_idx =
                            d * (params.seq * params.kv_heads) + kv_head * params.seq + kpos;
                        i64::from(q[q_idx]) * i64::from(k[k_idx])
                    })
                    .sum();
            }
        }
    }
    out
}

#[test]
fn proves_and_verifies_qk_score_sumchecks() {
    let params = QkScoreParams::new(2, 2, 1, 2).unwrap();
    let q = vec![300_i32, 200, 100, 400, 250, 350, 150, 450];
    let k = vec![120_i32, 220, 320, 420];
    let raw_acc = qk_raw_acc(&q, &k, &params);
    let dot = raw_acc
        .iter()
        .map(|value| round_q8(*value))
        .collect::<Vec<_>>();
    let scaled_acc = dot
        .iter()
        .map(|value| i64::from(*value) * i64::from(qk_score_inv_sqrt_q8(params.head_dim)))
        .collect::<Vec<_>>();
    let score = scaled_acc
        .iter()
        .map(|value| round_q8(*value))
        .collect::<Vec<_>>();
    let claim_point = vec![Fr::from(2_u64), Fr::from(5_u64), Fr::from(7_u64)];
    let claim = eval_score(&score, &params, &claim_point);

    let mut prover_transcript = Blake2bTranscript::default();
    let output = prove_qk_score(
        EvalClaim::new(claim, claim_point.clone()),
        QkScoreProverInput {
            params,
            witness: QkScoreWitness {
                q: q.clone(),
                k: k.clone(),
                dot: dot.clone(),
                score_remainder_bits: remainder_bits(&scaled_acc),
                dot_remainder_bits: remainder_bits(&raw_acc),
            },
        },
        &mut prover_transcript,
    )
    .expect("honest prover creates QK score sumchecks");

    let mut verifier_transcript = Blake2bTranscript::default();
    verifier_transcript.append_scalar(&claim);
    verifier_transcript.append_scalars(&claim_point);
    draw_qk_score_scale_challenges(&mut verifier_transcript).unwrap();
    let scale_sumcheck = verify_sumcheck_rounds(
        claim,
        &output.proof.scale.rounds,
        params.score_vars(),
        &mut verifier_transcript,
    )
    .expect("honest QK scale verifies");

    let dot_claim = EvalClaim::new(output.proof.scale.dot, scale_sumcheck.point);
    verifier_transcript.append_scalar(&dot_claim.value);
    verifier_transcript.append_scalars(&dot_claim.point);
    verifier_transcript.append_scalar(&output.proof.dot.rem);
    verifier_transcript.append_scalars(&dot_claim.point);
    verifier_transcript.append_scalar(&output.proof.dot.msb);
    verifier_transcript.append_scalars(&dot_claim.point);
    draw_qk_score_dot_challenges(&mut verifier_transcript).unwrap();
    let dot_sumcheck = verify_sumcheck_rounds(
        dot_claim.value * Fr::from(256_u64) + output.proof.dot.rem
            - Fr::from(256_u64) * output.proof.dot.msb,
        &output.proof.dot.k_reduction.rounds,
        params.head_vars() + params.head_dim.ilog2() as usize,
        &mut verifier_transcript,
    )
    .expect("honest QK dot verifies");

    let ([rem_gamma, msb_gamma], _) =
        draw_qk_score_dot_bit_booleanity_challenges(&mut verifier_transcript).unwrap();
    let dot_bits_sumcheck = verify_sumcheck_rounds(
        rem_gamma * output.proof.dot.rem + msb_gamma * output.proof.dot.msb,
        &output.proof.dot.rounding_bits.rounds,
        dot_claim.point.len(),
        &mut verifier_transcript,
    )
    .expect("honest QK dot bit booleanity verifies");

    assert_eq!(
        output.proof.scale.rounds.final_claim,
        scale_sumcheck.final_claim
    );
    assert_eq!(
        output.proof.dot.rounding_bits.rounds.final_claim,
        dot_bits_sumcheck.final_claim
    );
    assert_eq!(
        output.proof.dot.k_reduction.rounds.final_claim,
        dot_sumcheck.final_claim
    );
    assert_eq!(output.dot.q.value, eval_q(&q, &params, &output.dot.q.point));
    assert_eq!(output.dot.k.value, eval_k(&k, &params, &output.dot.k.point));
    assert_eq!(
        dot_sumcheck.point.len(),
        params.head_dim.ilog2() as usize + params.q_heads.ilog2() as usize
    );

    let mut full_verifier_transcript = Blake2bTranscript::default();
    let verified = verify_qk_score(
        EvalClaim::new(claim, claim_point),
        params,
        &output.proof,
        &mut full_verifier_transcript,
    )
    .expect("honest QK score proof verifies through verify_qk_score");
    assert_eq!(
        verified.dot.q.value,
        eval_q(&q, &params, &verified.dot.q.point)
    );
    assert_eq!(
        verified.dot.k.value,
        eval_k(&k, &params, &verified.dot.k.point)
    );
}
