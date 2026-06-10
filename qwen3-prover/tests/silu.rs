use ark_bn254::Fr;
use ark_ff::Zero;
use itertools::Itertools;
use joltworks::{field::JoltField, transcripts::Blake2bTranscript};
use qwen3_prover::{
    layer::EvalClaim,
    ops::silu::{
        SiluAdvice, SiluParams, SiluProverInput, SiluWitness, draw_silu_lookup_challenges,
        draw_silu_tensor_challenges, prove_silu, prove_silu_lookup, prove_silu_tensor,
    },
    round_message::SumCheckRounds,
};
use qwen3_verifier::ops::silu::verify_silu;

fn verify_sumcheck_rounds<const D: usize>(
    mut claim: Fr,
    rounds: &SumCheckRounds<D>,
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

fn fr_challenges(challenges: &[<Fr as JoltField>::Challenge]) -> Vec<Fr> {
    challenges.iter().copied().map(Into::into).collect()
}

fn round_q8(value: i64) -> i32 {
    let rem = value.rem_euclid(256);
    ((value + ((rem >> 7) * 256) - rem) / 256) as i32
}

fn remainder_bits(values: &[i64]) -> [Vec<i32>; 8] {
    std::array::from_fn(|bit| {
        values
            .iter()
            .map(|value| ((value.rem_euclid(256) >> bit) & 1) as i32)
            .collect()
    })
}

fn remainder_bool_bits(values: &[i64]) -> [Vec<bool>; 8] {
    remainder_bits(values).map(|bits| bits.into_iter().map(|bit| bit != 0).collect())
}

fn acc_values(input: &[i32], index: &[i32], min_n: i32, base: &[i32], slope: &[i32]) -> Vec<i64> {
    input
        .iter()
        .zip_eq(index)
        .zip_eq(base.iter().zip_eq(slope))
        .map(|((input, index), (base, slope))| {
            let n = min_n + *index;
            i64::from(*base) + (i64::from(*input) - 256_i64 * i64::from(n)) * i64::from(*slope)
        })
        .collect()
}

#[test]
fn proves_and_verifies_silu_tensor_sumcheck() {
    let input = vec![320_i32, 448, -64, 127];
    let index = input
        .iter()
        .map(|value| round_q8(i64::from(*value)))
        .collect::<Vec<i32>>();
    let base = vec![128_i32, 192, 64, 96];
    let slope = vec![42_i32, 50, 30, 10];
    let acc = acc_values(&input, &index, 0, &base, &slope);
    let out = acc
        .iter()
        .map(|value| round_q8(*value))
        .collect::<Vec<i32>>();
    let input_values = input
        .iter()
        .map(|value| i64::from(*value))
        .collect::<Vec<_>>();
    let claim = EvalClaim::new(Fr::zero(), vec![Fr::from(2_u64), Fr::from(5_u64)]);
    let params = SiluParams::new(2, 2).unwrap();

    let mut prover_transcript = Blake2bTranscript::default();
    let output = prove_silu_tensor(
        claim.clone(),
        params,
        Fr::zero(),
        input,
        index,
        base,
        slope,
        out,
        remainder_bool_bits(&input_values),
        remainder_bool_bits(&acc),
        &mut prover_transcript,
    )
    .expect("honest prover creates SiLU tensor sumcheck rounds");

    let mut verifier_transcript = Blake2bTranscript::default();
    draw_silu_tensor_challenges(&mut verifier_transcript).unwrap();
    let (final_claim, verifier_challenges) =
        verify_sumcheck_rounds(claim.value, &output.proof.rounds, &mut verifier_transcript)
            .expect("honest SiLU tensor sumcheck verifies");

    assert_eq!(final_claim, output.proof.rounds.final_claim);
    assert_eq!(output.index.point, fr_challenges(&verifier_challenges));
    assert_eq!(output.base.point, output.index.point);
    assert_eq!(output.slope.point, output.index.point);
}

#[test]
fn proves_and_verifies_silu_lookup_sumcheck() {
    let id_table = vec![3_i32; 4];
    let base_table = vec![128_i32; 4];
    let slope_table = vec![42_i32; 4];
    let ra = vec![1_i32, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let tensor_point = vec![Fr::from(2_u64), Fr::from(5_u64)];

    let mut prover_transcript = Blake2bTranscript::default();
    let output = prove_silu_lookup(
        tensor_point.clone(),
        Fr::from_i32(id_table[0]),
        Fr::from_i32(base_table[0]),
        Fr::from_i32(slope_table[0]),
        ra.clone().into_iter().map(|value| value as u8).collect(),
        id_table.clone(),
        base_table.clone(),
        slope_table.clone(),
        SiluParams::new(2, 2).unwrap().shape,
        &mut prover_transcript,
    )
    .expect("honest prover creates SiLU lookup sumcheck rounds");

    let mut verifier_transcript = Blake2bTranscript::default();
    let (read_challenges, _) = draw_silu_lookup_challenges(&mut verifier_transcript).unwrap();
    let [one_challenge, id_challenge, base_challenge, slope_challenge] = read_challenges;
    let lookup_claim = one_challenge
        + id_challenge * Fr::from_i32(id_table[0])
        + base_challenge * Fr::from_i32(base_table[0])
        + slope_challenge * Fr::from_i32(slope_table[0]);
    let (read_final_claim, read_challenges) =
        verify_sumcheck_rounds(lookup_claim, &output.read.rounds, &mut verifier_transcript)
            .expect("honest SiLU read sumcheck verifies");
    let (ra_virtual_final_claim, ra_virtual_challenges) = verify_sumcheck_rounds(
        output.read.ra,
        &output.ra_virtual.rounds,
        &mut verifier_transcript,
    )
    .expect("honest SiLU RA virtual sumcheck verifies");
    let (hamming_final_claim, hamming_challenges) = verify_sumcheck_rounds(
        Fr::from(1_u64),
        &output.hamming_weight.rounds,
        &mut verifier_transcript,
    )
    .expect("honest SiLU hamming-weight sumcheck verifies");
    let _booleanity_address_point = joltworks::transcripts::Transcript::challenge_vector::<Fr>(
        &mut verifier_transcript,
        id_table.len().ilog2() as usize,
    );
    let (booleanity_final_claim, booleanity_challenges) = verify_sumcheck_rounds(
        Fr::zero(),
        &output.booleanity.rounds,
        &mut verifier_transcript,
    )
    .expect("honest SiLU RA booleanity sumcheck verifies");

    assert_eq!(read_final_claim, output.read.rounds.final_claim);
    assert_eq!(ra_virtual_final_claim, output.ra_virtual.rounds.final_claim);
    assert_eq!(
        hamming_final_claim,
        output.hamming_weight.rounds.final_claim
    );
    assert_eq!(booleanity_final_claim, output.booleanity.rounds.final_claim);
    assert_eq!(read_challenges.len(), id_table.len().ilog2() as usize);
    assert_eq!(ra_virtual_challenges.len(), tensor_point.len());
    assert_eq!(hamming_challenges.len(), id_table.len().ilog2() as usize);
    assert_eq!(
        booleanity_challenges.len(),
        tensor_point.len() + id_table.len().ilog2() as usize
    );
}

#[test]
fn proves_silu_by_chaining_tensor_and_lookup_sumchecks() {
    let input = vec![256_i32, 300, 320, 383];
    let min_n = 1_i32;
    let index = vec![0_i32; input.len()];
    let base = vec![187_i32 * 256_i32; input.len()];
    let slope = vec![237_i32; input.len()];
    let acc = acc_values(&input, &index, min_n, &base, &slope);
    let out = acc
        .iter()
        .map(|value| round_q8(*value))
        .collect::<Vec<i32>>();
    let tensor_claim = EvalClaim::new(Fr::zero(), vec![Fr::from(2_u64), Fr::from(5_u64)]);
    let params = SiluParams::new(2, 2).unwrap();
    let ra = vec![1_u8; input.len()];

    let mut prover_transcript = Blake2bTranscript::default();
    let output = prove_silu(
        tensor_claim.clone(),
        SiluProverInput {
            params,
            advice: SiluAdvice {
                min_n: i64::from(min_n),
                max_n: i64::from(min_n),
            },
            witness: SiluWitness {
                input: input.clone(),
                output: out,
                ra,
                input_remainder_bits: remainder_bool_bits(
                    &input
                        .iter()
                        .map(|value| i64::from(*value))
                        .collect::<Vec<_>>(),
                ),
                output_remainder_bits: remainder_bool_bits(&acc),
            },
        },
        &mut prover_transcript,
    )
    .expect("honest prover creates chained SiLU sumchecks");

    let mut verifier_transcript = Blake2bTranscript::default();
    let verified = verify_silu(
        tensor_claim,
        params,
        SiluAdvice {
            min_n: i64::from(min_n),
            max_n: i64::from(min_n),
        },
        &output.proof,
        &mut verifier_transcript,
    )
    .expect("honest SiLU proof verifies");
    assert_eq!(verified.input, output.input);
}
