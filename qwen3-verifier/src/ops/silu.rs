use ark_bn254::Fr;
use ark_ff::One;
use itertools::Itertools;
use joltworks::transcripts::Transcript;
use qwen3_common::ops::silu::{
    SiluAdvice, SiluBooleanityOutput, SiluHammingWeightOutput, SiluOutput, SiluParams,
    SiluRaVirtualOutput, SiluReadOutput, SiluTensorOutput, SiluVerifierOutput, build_silu_tables,
    draw_silu_lookup_challenges, draw_silu_tensor_challenges,
};

use qwen3_common::{
    BitOpeningClaims, EvalClaim, FRAC_BITS, RaOpeningClaims, SCALE, append_eval_claim,
    verify_sumcheck_rounds,
};

use crate::utils::{eq_point_eval, eval_i32_mle_at_point};

pub fn verify_silu<Tr>(
    claim: EvalClaim,
    params: SiluParams,
    advice: SiluAdvice,
    proof: &SiluOutput,
    transcript: &mut Tr,
) -> Option<SiluVerifierOutput>
where
    Tr: Transcript,
{
    append_eval_claim(transcript, &claim);
    (claim.point.len() == params.shape.point_len()).then_some(())?;
    let tensor = verify_silu_tensor_sumcheck(
        claim,
        params,
        field_from_i64(advice.min_n),
        &proof.tensor,
        transcript,
    )?;

    // The tensor sumcheck reduced the output claim to index/base/slope claims.
    // These three values must be tied to the lookup table through the RA
    // one-hot checks below.
    append_eval_claim(transcript, &tensor.index);
    append_eval_claim(transcript, &tensor.base);
    append_eval_claim(transcript, &tensor.slope);
    let (read_challenges, _) = draw_silu_lookup_challenges(transcript)?;
    let read = verify_silu_read_sumcheck(
        tensor.point.clone(),
        tensor.index.value,
        tensor.base.value,
        tensor.slope.value,
        &advice,
        read_challenges,
        &proof.lookup.read,
        transcript,
    )?;
    let ra_virtual = verify_silu_ra_virtual_sumcheck(
        tensor.point.clone(),
        read.lookup_point.clone(),
        read.ra.value,
        &proof.lookup.ra_virtual,
        transcript,
    )?;
    let hamming_weight = verify_silu_hamming_weight_sumcheck(
        tensor.point.clone(),
        read.lookup_point.len(),
        &proof.lookup.hamming_weight,
        transcript,
    )?;
    let booleanity_address_point = transcript.challenge_vector::<Fr>(read.lookup_point.len());
    let booleanity = verify_silu_ra_booleanity_sumcheck(
        tensor.point.clone(),
        booleanity_address_point,
        &proof.lookup.booleanity,
        transcript,
    )?;
    Some(SiluVerifierOutput {
        input: tensor.input,
        input_bits: bit_opening_claims(&tensor.point, proof.tensor.input_bits),
        output_bits: bit_opening_claims(&tensor.point, proof.tensor.output_bits),
        ra: RaOpeningClaims {
            read: read.ra,
            virtual_claim: ra_virtual,
            hamming_weight,
            booleanity,
        },
    })
}

struct SiluTensorRelation {
    min_n: Fr,
    input_round_mix: Fr,
    input_bit_booleanity_challenges: [Fr; FRAC_BITS],
    output_bit_booleanity_challenges: [Fr; FRAC_BITS],
}

struct VerifiedTensor {
    point: Vec<Fr>,
    input: EvalClaim,
    index: EvalClaim,
    base: EvalClaim,
    slope: EvalClaim,
}

struct VerifiedRead {
    lookup_point: Vec<Fr>,
    ra: EvalClaim,
}

fn verify_silu_tensor_sumcheck<Tr>(
    claim: EvalClaim,
    params: SiluParams,
    min_n: Fr,
    proof: &SiluTensorOutput,
    transcript: &mut Tr,
) -> Option<VerifiedTensor>
where
    Tr: Transcript,
{
    let (input_round_mix, input_bit_booleanity_challenges, output_bit_booleanity_challenges) =
        draw_silu_tensor_challenges(transcript)?;
    let relation = SiluTensorRelation {
        min_n,
        input_round_mix,
        input_bit_booleanity_challenges,
        output_bit_booleanity_challenges,
    };
    // Tensor sumcheck for the SiLU approximation:
    //
    //   n        = min_n + index
    //   input    = 256*n + input_rem - 256*input_msb
    //   output   = round((base + (input - 256*n) * slope) / 256)
    //
    // The same relation also checks booleanity of input/output rounding bits.
    let sumcheck = verify_sumcheck_rounds(
        Fr::from(0_u64),
        &proof.rounds,
        params.shape.point_len(),
        transcript,
    )?;
    let point = sumcheck.point;
    (point.len() == params.shape.point_len()).then_some(())?;
    let input = EvalClaim::new(proof.input, point.clone());
    let index = EvalClaim::new(proof.index, point.clone());
    let base = EvalClaim::new(proof.base, point.clone());
    let slope = EvalClaim::new(proof.slope, point.clone());
    let final_relation = silu_tensor_final_relation(
        &relation,
        proof.input,
        proof.index,
        proof.base,
        proof.slope,
        proof.output,
        proof.input_bits,
        proof.output_bits,
    );
    (sumcheck.final_claim == eq_point_eval(&claim.point, &sumcheck.challenges)? * final_relation)
        .then_some(())?;
    Some(VerifiedTensor {
        point,
        input,
        index,
        base,
        slope,
    })
}

fn verify_silu_read_sumcheck<Tr>(
    tensor_point: Vec<Fr>,
    index: Fr,
    base: Fr,
    slope: Fr,
    advice: &SiluAdvice,
    read_challenges: [Fr; 4],
    proof: &SiluReadOutput,
    transcript: &mut Tr,
) -> Option<VerifiedRead>
where
    Tr: Transcript,
{
    let [one_challenge, id_challenge, base_challenge, slope_challenge] = read_challenges;
    let claim =
        one_challenge + id_challenge * index + base_challenge * base + slope_challenge * slope;
    // RA read sumcheck:
    //
    //   gamma_1 + gamma_id*index + gamma_base*base + gamma_slope*slope
    //     = Σ_addr ra(addr, x)
    //         * (gamma_1 + gamma_id*id(addr)
    //            + gamma_base*base_table(addr)
    //            + gamma_slope*slope_table(addr))
    //
    // It binds the tensor point x and returns an opening claim for ra(addr*, x).
    let lookup_vars = build_silu_tables(advice)?.id.len().ilog2() as usize;
    let sumcheck = verify_sumcheck_rounds(claim, &proof.rounds, lookup_vars, transcript)?;
    let lookup_point = sumcheck.point;
    let ra_point = [lookup_point.as_slice(), tensor_point.as_slice()].concat();
    let ra = EvalClaim::new(proof.ra, ra_point);
    let public_evals = build_public_silu_read_evals(advice, &lookup_point)?;
    let final_relation = proof.ra
        * (one_challenge
            + id_challenge * public_evals.id
            + base_challenge * public_evals.base
            + slope_challenge * public_evals.slope);
    (sumcheck.final_claim == final_relation).then_some(())?;
    Some(VerifiedRead {
        lookup_point: lookup_point.clone(),
        ra,
    })
}

struct SiluReadPublicEvals {
    id: Fr,
    base: Fr,
    slope: Fr,
}

fn build_public_silu_read_evals(
    advice: &SiluAdvice,
    lookup_point: &[Fr],
) -> Option<SiluReadPublicEvals> {
    let tables = build_silu_tables(advice)?;
    Some(SiluReadPublicEvals {
        id: eval_i32_mle_at_point(&tables.id, lookup_point)?,
        base: eval_i32_mle_at_point(&tables.base, lookup_point)?,
        slope: eval_i32_mle_at_point(&tables.slope, lookup_point)?,
    })
}

fn verify_silu_ra_virtual_sumcheck<Tr>(
    tensor_point: Vec<Fr>,
    lookup_point: Vec<Fr>,
    claim: Fr,
    proof: &SiluRaVirtualOutput,
    transcript: &mut Tr,
) -> Option<EvalClaim>
where
    Tr: Transcript,
{
    // RA virtual sumcheck selects the RA row for the tensor point:
    //
    //   ra(addr*, x_tensor)
    //     = Σ_x eq(x_tensor, x) * ra(addr*, x)
    //
    // This converts the read claim into the same committed RA domain used by
    // the opening phase.
    let sumcheck = verify_sumcheck_rounds(claim, &proof.rounds, tensor_point.len(), transcript)?;
    let row_point = sumcheck.point;
    let ra = EvalClaim::new(
        proof.ra,
        [lookup_point.as_slice(), row_point.as_slice()].concat(),
    );
    (sumcheck.final_claim == eq_point_eval(&tensor_point, &sumcheck.challenges)? * proof.ra)
        .then_some(())?;
    Some(ra)
}

fn verify_silu_hamming_weight_sumcheck<Tr>(
    tensor_point: Vec<Fr>,
    lookup_vars: usize,
    proof: &SiluHammingWeightOutput,
    transcript: &mut Tr,
) -> Option<EvalClaim>
where
    Tr: Transcript,
{
    // Hamming-weight sumcheck proves that a selected RA row has total mass one:
    //
    //   1 = Σ_addr ra(addr, x_tensor)
    //
    // Together with booleanity this makes the RA row one-hot.
    let sumcheck = verify_sumcheck_rounds(Fr::one(), &proof.rounds, lookup_vars, transcript)?;
    let lookup_point = sumcheck.point;
    (lookup_point.len() == lookup_vars).then_some(())?;
    let ra = EvalClaim::new(
        proof.ra,
        [lookup_point.as_slice(), tensor_point.as_slice()].concat(),
    );
    (sumcheck.final_claim == proof.ra).then_some(())?;
    Some(ra)
}

fn verify_silu_ra_booleanity_sumcheck<Tr>(
    tensor_point: Vec<Fr>,
    lookup_point: Vec<Fr>,
    proof: &SiluBooleanityOutput,
    transcript: &mut Tr,
) -> Option<EvalClaim>
where
    Tr: Transcript,
{
    // RA booleanity sumcheck checks one random RA cell:
    //
    //   0 = Σ_{addr,x} eq((addr*, x_tensor), (addr, x))
    //       * ra(addr, x) * (ra(addr, x) - 1)
    //
    // The address point is sampled independently from the read point, matching
    // the SHOUT-style read/virtual/hamming/booleanity split.
    let sumcheck = verify_sumcheck_rounds(
        Fr::from(0_u64),
        &proof.rounds,
        lookup_point.len() + tensor_point.len(),
        transcript,
    )?;
    let point = sumcheck.point;
    let ra = EvalClaim::new(proof.ra, point);
    let eq_point = [lookup_point.as_slice(), tensor_point.as_slice()].concat();
    (sumcheck.final_claim
        == eq_point_eval(&eq_point, &sumcheck.challenges)? * proof.ra * (proof.ra - Fr::one()))
    .then_some(())?;
    Some(ra)
}

fn bit_opening_claims(point: &[Fr], values: [Fr; FRAC_BITS]) -> BitOpeningClaims {
    values.map(|value| EvalClaim::new(value, point.to_vec()))
}

#[allow(clippy::too_many_arguments)]
fn silu_tensor_final_relation(
    relation: &SiluTensorRelation,
    input: Fr,
    index: Fr,
    base: Fr,
    slope: Fr,
    output: Fr,
    input_bits: [Fr; FRAC_BITS],
    output_bits: [Fr; FRAC_BITS],
) -> Fr {
    let n = relation.min_n + index;
    let scale = Fr::from(SCALE);
    let input_pairs = input_bits.map(|bit| (bit, bit));
    let output_pairs = output_bits.map(|bit| (bit, bit));
    let input_remainder = remainder_constant(&input_pairs);
    let output_remainder = remainder_constant(&output_pairs);
    let input_msb = input_bits[FRAC_BITS - 1];
    let output_msb = output_bits[FRAC_BITS - 1];

    let output_round =
        scale * output - base - (input - scale * n) * slope + output_remainder - scale * output_msb;
    let input_round = input + scale * input_msb - input_remainder - scale * n;
    let mut value = output_round + relation.input_round_mix * input_round;
    value += input_bits
        .iter()
        .zip_eq(relation.input_bit_booleanity_challenges)
        .map(|(bit, challenge)| challenge * *bit * (*bit - Fr::one()))
        .sum::<Fr>();
    value += output_bits
        .iter()
        .zip_eq(relation.output_bit_booleanity_challenges)
        .map(|(bit, challenge)| challenge * *bit * (*bit - Fr::one()))
        .sum::<Fr>();
    value
}

fn remainder_constant(bits: &[(Fr, Fr); FRAC_BITS]) -> Fr {
    bits.iter()
        .enumerate()
        .map(|(bit, (value, _))| Fr::from(1_u64 << bit) * *value)
        .sum()
}

fn field_from_i64(value: i64) -> Fr {
    if value >= 0 {
        Fr::from(value as u64)
    } else {
        -Fr::from((-value) as u64)
    }
}
