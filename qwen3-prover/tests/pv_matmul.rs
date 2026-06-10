use ark_bn254::Fr;
use ark_ff::One;
use joltworks::{
    field::JoltField,
    transcripts::{Blake2bTranscript, Transcript},
};
use qwen3_prover::{
    layer::EvalClaim,
    ops::pv_matmul::{
        PvMatmulParams, PvMatmulProverInput, PvMatmulWitness,
        draw_pv_matmul_bit_booleanity_challenges, draw_pv_matmul_challenges, prove_pv_matmul,
    },
    round::verify_sumcheck_rounds,
};
use qwen3_verifier::ops::pv_matmul::verify_pv_matmul;

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

fn eval_context(values: &[i32], params: &PvMatmulParams, point: &[Fr]) -> Fr {
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

fn eval_p(values: &[i32], params: &PvMatmulParams, point: &[Fr]) -> Fr {
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

fn eval_v(values: &[i32], params: &PvMatmulParams, point: &[Fr]) -> Fr {
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

fn pv_acc(p: &[i32], v: &[i32], params: &PvMatmulParams) -> Vec<i64> {
    let mut out = vec![0_i64; params.seq * params.q_heads * params.head_dim];
    for qpos in 0..params.seq {
        for head in 0..params.q_heads {
            let kv_head = head / 2;
            for d in 0..params.head_dim {
                out[d * (params.seq * params.q_heads) + head * params.seq + qpos] = (0..params.seq)
                    .map(|kpos| {
                        let p_idx = kpos * (params.seq * params.q_heads) + head * params.seq + qpos;
                        let v_idx =
                            d * (params.seq * params.kv_heads) + kv_head * params.seq + kpos;
                        i64::from(p[p_idx]) * i64::from(v[v_idx])
                    })
                    .sum();
            }
        }
    }
    out
}

#[test]
fn proves_and_verifies_pv_matmul_sumcheck() {
    let params = PvMatmulParams::new(2, 2, 1, 2).unwrap();
    let p = vec![1_i32, 2, 3, 4, 5, 6, 7, 8];
    let v = vec![2_i32, 1, 4, 3];
    let acc = pv_acc(&p, &v, &params);
    let context = acc.iter().map(|value| round_q8(*value)).collect::<Vec<_>>();
    let claim_point = vec![Fr::from(3_u64), Fr::from(5_u64), Fr::from(11_u64)];
    let claim = eval_context(&context, &params, &claim_point);

    let mut prover_transcript = Blake2bTranscript::default();
    let output = prove_pv_matmul(
        EvalClaim::new(claim, claim_point.clone()),
        PvMatmulProverInput {
            params,
            witness: PvMatmulWitness {
                p: p.clone(),
                v: v.clone(),
                context_remainder_bits: remainder_bits(&acc),
            },
        },
        &mut prover_transcript,
    )
    .expect("honest prover creates PV matmul sumcheck");

    let mut verifier_transcript = Blake2bTranscript::default();
    verifier_transcript.append_scalar(&claim);
    verifier_transcript.append_scalars(&claim_point);
    draw_pv_matmul_challenges(&mut verifier_transcript).unwrap();
    verifier_transcript.append_scalar(&output.proof.rem);
    verifier_transcript.append_scalars(&claim_point);
    verifier_transcript.append_scalar(&output.proof.msb);
    verifier_transcript.append_scalars(&claim_point);
    let reduction = verify_sumcheck_rounds(
        claim * Fr::from(256_u64) + output.proof.rem - Fr::from(256_u64) * output.proof.msb,
        &output.proof.k_reduction.rounds,
        params.q_heads.ilog2() as usize + params.seq.ilog2() as usize,
        &mut verifier_transcript,
    )
    .expect("honest PV matmul verifies");

    let ([rem_gamma, msb_gamma], _) =
        draw_pv_matmul_bit_booleanity_challenges(&mut verifier_transcript).unwrap();
    let rounding = verify_sumcheck_rounds(
        rem_gamma * output.proof.rem + msb_gamma * output.proof.msb,
        &output.proof.rounding_bits.rounds,
        claim_point.len(),
        &mut verifier_transcript,
    )
    .expect("honest PV matmul bit booleanity verifies");

    assert_eq!(
        output.proof.rounding_bits.rounds.final_claim,
        rounding.final_claim
    );
    assert_eq!(
        output.proof.k_reduction.rounds.final_claim,
        reduction.final_claim
    );
    assert_eq!(output.p.value, eval_p(&p, &params, &output.p.point));
    assert_eq!(output.v.value, eval_v(&v, &params, &output.v.point));

    let mut full_verifier_transcript = Blake2bTranscript::default();
    let verified = verify_pv_matmul(
        EvalClaim::new(claim, claim_point),
        params,
        &output.proof,
        &mut full_verifier_transcript,
    )
    .expect("honest PV matmul proof verifies through verify_pv_matmul");
    assert_eq!(verified.p.value, eval_p(&p, &params, &verified.p.point));
    assert_eq!(verified.v.value, eval_v(&v, &params, &verified.v.point));
    assert_eq!(
        verified.p.point.len(),
        params.seq.ilog2() as usize + params.q_heads.ilog2() as usize + params.seq.ilog2() as usize
    );
    assert_eq!(
        verified.v.point.len(),
        params.seq.ilog2() as usize
            + params.head_dim.ilog2() as usize
            + params.kv_heads.ilog2() as usize
    );
}
