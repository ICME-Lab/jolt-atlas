use ark_bn254::{Bn254, Fr};
use ark_ff::{One, Zero};
use itertools::Itertools;
use joltworks::{
    field::JoltField,
    poly::{
        commitment::hyperkzg::{HyperKZG, HyperKZGCommitment, HyperKZGProof, HyperKZGProverKey},
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialEvaluation},
        split_eq_poly::GruenSplitEqPolynomial,
    },
    transcripts::Transcript,
};
use rayon::prelude::*;

use crate::{
    commitment::{ChunkedCommitments, LayerCommitments},
    layer::{
        BitOpeningClaims, BitOpeningWitnesses, EvalClaim, LayerOpeningClaims,
        LayerOpeningWitnesses, LayerShape, RaOpeningClaims,
    },
    profile,
    round_message::{RoundPolynomial, SumCheckRounds, append_round_statement},
};
pub use qwen3_common::{
    ChunkedLayerPcsOpeningProof, OpeningReductionOutput, OpeningReductionProof,
};

pub fn layer_opening_claim_count(claims: &LayerOpeningClaims) -> usize {
    flatten_layer_claims(claims).len()
}

pub fn prove_layer_opening_reduction_sumcheck<Tr>(
    claims: &LayerOpeningClaims,
    witnesses: &LayerOpeningWitnesses,
    shape: LayerShape,
    transcript: &mut Tr,
) -> Option<OpeningReductionOutput>
where
    Tr: Transcript,
{
    validate_layer_opening_claim_domains(claims, shape)?;
    let claims = flatten_layer_claims(claims);
    let tables = layer_opening_tables(witnesses, shape)?;
    (claims.len() == tables.len()).then_some(())?;

    let max_vars = claims.iter().map(|claim| claim.point.len()).max()?;
    for claim in &claims {
        transcript.append_scalar(&claim.value);
    }
    let gammas = transcript.challenge_vector::<Fr>(claims.len());

    let mut states = claims
        .iter()
        .zip_eq(tables)
        .map(|(claim, table)| OpeningTermState::new(claim, table, max_vars))
        .collect::<Option<Vec<_>>>()?;
    for (index, (claim, state)) in claims.iter().zip_eq(&states).enumerate() {
        let actual = state.current_claim();
        if claim.value != actual {
            opening_debug(|| {
                eprintln!(
                    "opening reduction initial claim mismatch: index={index}, point_vars={}, table_len={}, expected={:?}, actual={:?}",
                    claim.point.len(),
                    state.len(),
                    claim.value,
                    actual,
                );
            });
            return None;
        }
    }
    let mut claim_i = states
        .iter()
        .zip_eq(&gammas)
        .map(|(state, gamma)| *gamma * state.current_claim())
        .sum();

    let mut round_polys = Vec::with_capacity(max_vars);
    let mut reduction_point = Vec::with_capacity(max_vars);
    let mut reduction_challenges = Vec::with_capacity(max_vars);
    for round in 0..max_vars {
        let round_poly = opening_reduction_round_poly(&states, &gammas, round);
        (round_poly.eval(Fr::zero()) + round_poly.eval(Fr::one()) == claim_i).then_some(())?;
        append_round_statement(transcript, claim_i, &round_poly);
        let challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r = challenge.into();
        for state in &mut states {
            state.bind(round, challenge, r);
        }
        claim_i = round_poly.eval(r);
        reduction_point.push(r);
        reduction_challenges.push(challenge);
        round_polys.push(round_poly);
    }

    let mut evals_at_reduction_point = Vec::with_capacity(states.len());
    for (index, state) in states.into_iter().enumerate() {
        match state.final_claim() {
            Some(value) => evals_at_reduction_point.push(value),
            None => {
                opening_debug(|| {
                    eprintln!("opening reduction final claim missing: index={index}");
                });
                return None;
            }
        }
    }
    Some(OpeningReductionOutput {
        proof: OpeningReductionProof {
            rounds: SumCheckRounds {
                round_polys,
                final_claim: claim_i,
            },
            evals_at_reduction_point,
        },
        reduction_point,
        reduction_challenges,
        gamma_powers: gammas,
    })
}

pub fn prove_chunked_layer_pcs_opening<Tr>(
    claims: &LayerOpeningClaims,
    witnesses: &LayerOpeningWitnesses,
    shape: LayerShape,
    opening: &OpeningReductionOutput,
    commitments: &LayerCommitments<HyperKZGCommitment<Bn254>>,
    setup: &HyperKZGProverKey<Bn254>,
    transcript: &mut Tr,
) -> Option<ChunkedLayerPcsOpeningProof>
where
    Tr: Transcript,
{
    let prepared = match profile::measure("chunked_pcs_open.chunk_evals", || {
        prepare_joint_chunk_opening(
            claims,
            witnesses,
            shape,
            commitments,
            &opening.reduction_point,
            &opening.reduction_challenges,
            &opening.gamma_powers,
        )
    }) {
        Some(prepared) => prepared,
        None => {
            opening_debug(|| eprintln!("chunk pcs: failed to prepare joint chunk opening"));
            return None;
        }
    };
    let expected =
        direct_chunk_claim(&prepared.coefficients, &prepared.chunk_evals).or_else(|| {
            opening_debug(|| eprintln!("chunk pcs: failed to compute direct chunk claim"));
            None
        })?;
    if opening.proof.rounds.final_claim != expected {
        opening_debug(|| eprintln!("chunk pcs: direct chunk claim mismatch"));
        return None;
    }
    append_scalars(transcript, &prepared.chunk_evals);
    let pcs_gammas = transcript.challenge_vector::<Fr>(prepared.chunk_evals.len());
    let proofs = match profile::measure("chunked_pcs_open.hyperkzg_open", || {
        prove_joint_chunk_opening(setup, &prepared, &pcs_gammas, transcript)
    }) {
        Some(proofs) => proofs,
        None => {
            opening_debug(|| {
                eprintln!(
                    "chunk pcs: failed to prove joint chunk opening: chunks={}",
                    prepared.chunk_evals.len()
                )
            });
            return None;
        }
    };

    Some(ChunkedLayerPcsOpeningProof {
        chunk_evals: prepared.chunk_evals,
        proofs,
    })
}

pub fn pcs_joint_claim(
    claims: &LayerOpeningClaims,
    shape: LayerShape,
    r_sumcheck: &[Fr],
    gamma_powers: &[Fr],
) -> Option<Fr> {
    validate_layer_opening_claim_domains(claims, shape)?;
    let claims = flatten_layer_claims(claims);
    let max_vars = claims.iter().map(|claim| claim.point.len()).max()?;
    (r_sumcheck.len() == max_vars && gamma_powers.len() == claims.len()).then_some(())?;

    Some(
        claims
            .into_iter()
            .zip_eq(gamma_powers)
            .map(|(claim, gamma)| {
                *gamma * claim.value * zero_suffix_eval(&r_sumcheck[claim.point.len()..])
            })
            .sum(),
    )
}

pub fn pcs_joint_claim_from_values(
    original_claims: &[&EvalClaim],
    evals_at_reduction_point: &[Fr],
    r_sumcheck: &[Fr],
    gamma_powers: &[Fr],
) -> Option<Fr> {
    let max_vars = original_claims
        .iter()
        .map(|claim| claim.point.len())
        .max()?;
    (r_sumcheck.len() == max_vars
        && gamma_powers.len() == original_claims.len()
        && evals_at_reduction_point.len() == original_claims.len())
    .then_some(())?;

    Some(
        original_claims
            .iter()
            .zip_eq(evals_at_reduction_point)
            .zip_eq(gamma_powers)
            .map(|((claim, value), gamma)| {
                *gamma * *value * zero_suffix_eval(&r_sumcheck[claim.point.len()..])
            })
            .sum(),
    )
}

fn eq_eval(left: &[Fr], right: &[Fr]) -> Option<Fr> {
    (left.len() == right.len()).then_some(())?;
    Some(
        left.iter()
            .zip_eq(right)
            .map(|(left, right)| (Fr::one() - left) * (Fr::one() - right) + *left * *right)
            .product(),
    )
}

fn eq_index_eval(index: usize, reduction_challenges: &[Fr]) -> Option<Fr> {
    (index < (1_usize << reduction_challenges.len())).then_some(())?;
    Some(
        reduction_challenges
            .iter()
            .enumerate()
            .map(|(bit, reduction_challenges)| {
                if ((index >> bit) & 1) == 1 {
                    *reduction_challenges
                } else {
                    Fr::one() - reduction_challenges
                }
            })
            .product(),
    )
}

fn zero_suffix_eval(reduction_challenges: &[Fr]) -> Fr {
    reduction_challenges.iter().map(|r| Fr::one() - r).product()
}

fn reversed_challenges(
    reduction_challenges: &[<Fr as JoltField>::Challenge],
) -> Vec<<Fr as JoltField>::Challenge> {
    reduction_challenges.iter().rev().copied().collect()
}

fn append_scalars<Tr: Transcript>(transcript: &mut Tr, values: &[Fr]) {
    for value in values {
        transcript.append_scalar(value);
    }
}

fn flatten_layer_claims(claims: &LayerOpeningClaims) -> Vec<&EvalClaim> {
    let mut out = Vec::with_capacity(235);
    out.push(&claims.hidden_out);
    out.push(&claims.hidden_in_a);
    out.push(&claims.hidden_in_b);
    push_ra_claims(&mut out, &claims.silu_lookup_ra);
    push_ra_claims(&mut out, &claims.softmax_lookup_ra);
    push_bit_claims(&mut out, &claims.down_proj_output_frac_bits);
    push_bit_claims(&mut out, &claims.silu_up_output_frac_bits);
    push_bit_claims(&mut out, &claims.silu_input_frac_bits);
    push_bit_claims(&mut out, &claims.silu_output_frac_bits);
    push_bit_claims(&mut out, &claims.gate_proj_output_frac_bits);
    push_bit_claims(&mut out, &claims.up_proj_output_frac_bits);
    push_bit_claims(&mut out, &claims.rms_norm_mlp_norm_frac_bits);
    push_bit_claims(&mut out, &claims.rms_norm_mlp_output_frac_bits);
    push_bit_claims(&mut out, &claims.o_proj_output_frac_bits);
    push_bit_claims(&mut out, &claims.pv_matmul_output_frac_bits);
    push_bit_claims(&mut out, &claims.softmax_floor_frac_bits);
    push_bit_claims(&mut out, &claims.softmax_output_frac_bits);
    push_bit_claims(&mut out, &claims.softmax_exp_frac_bits);
    push_bit_claims(&mut out, &claims.qk_score_dot_output_frac_bits);
    push_bit_claims(&mut out, &claims.qk_score_output_frac_bits);
    push_bit_claims(&mut out, &claims.q_rope_first_half_output_frac_bits);
    push_bit_claims(&mut out, &claims.q_rope_second_half_output_frac_bits);
    push_bit_claims(&mut out, &claims.k_rope_first_half_output_frac_bits);
    push_bit_claims(&mut out, &claims.k_rope_second_half_output_frac_bits);
    push_bit_claims(&mut out, &claims.q_norm_norm_frac_bits);
    push_bit_claims(&mut out, &claims.q_norm_output_frac_bits);
    push_bit_claims(&mut out, &claims.k_norm_norm_frac_bits);
    push_bit_claims(&mut out, &claims.k_norm_output_frac_bits);
    push_bit_claims(&mut out, &claims.q_proj_output_frac_bits);
    push_bit_claims(&mut out, &claims.k_proj_output_frac_bits);
    push_bit_claims(&mut out, &claims.v_proj_output_frac_bits);
    push_bit_claims(&mut out, &claims.rms_norm_atten_norm_frac_bits);
    push_bit_claims(&mut out, &claims.rms_norm_atten_output_frac_bits);
    out
}

fn push_ra_claims<'a>(out: &mut Vec<&'a EvalClaim>, claims: &'a RaOpeningClaims) {
    out.push(&claims.read);
    out.push(&claims.virtual_claim);
    out.push(&claims.hamming_weight);
    out.push(&claims.booleanity);
}

fn push_bit_claims<'a>(out: &mut Vec<&'a EvalClaim>, claims: &'a BitOpeningClaims) {
    out.extend(claims.iter());
}

fn layer_opening_tables(
    witnesses: &LayerOpeningWitnesses,
    shape: LayerShape,
) -> Option<Vec<OpeningTable<'_>>> {
    let shape = shape.padded();
    let silu_tensor_len = shape.seq.checked_mul(shape.intermediate)?;
    let softmax_tensor_len = shape
        .q_heads
        .checked_mul(shape.seq)?
        .checked_mul(shape.seq)?;

    let mut out = Vec::with_capacity(235);
    out.push(OpeningTable::I32(&witnesses.hidden_out));
    out.push(OpeningTable::I32(&witnesses.hidden_in_a));
    out.push(OpeningTable::I32(&witnesses.hidden_in_b));
    push_ra_tables(&mut out, &witnesses.silu_lookup_ra, silu_tensor_len)?;
    push_ra_tables(&mut out, &witnesses.softmax_lookup_ra, softmax_tensor_len)?;
    push_bit_tables(&mut out, &witnesses.down_proj_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.silu_up_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.silu_input_frac_bits);
    push_bit_tables(&mut out, &witnesses.silu_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.gate_proj_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.up_proj_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.rms_norm_mlp_norm_frac_bits);
    push_bit_tables(&mut out, &witnesses.rms_norm_mlp_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.o_proj_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.pv_matmul_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.softmax_floor_frac_bits);
    push_bit_tables(&mut out, &witnesses.softmax_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.softmax_exp_frac_bits);
    push_bit_tables(&mut out, &witnesses.qk_score_dot_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.qk_score_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.q_rope_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.q_rope_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.k_rope_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.k_rope_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.q_norm_norm_frac_bits);
    push_bit_tables(&mut out, &witnesses.q_norm_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.k_norm_norm_frac_bits);
    push_bit_tables(&mut out, &witnesses.k_norm_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.q_proj_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.k_proj_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.v_proj_output_frac_bits);
    push_bit_tables(&mut out, &witnesses.rms_norm_atten_norm_frac_bits);
    push_bit_tables(&mut out, &witnesses.rms_norm_atten_output_frac_bits);
    Some(out)
}

fn layer_opening_table_shapes(
    commitments: &LayerCommitments<HyperKZGCommitment<Bn254>>,
    shape: LayerShape,
) -> Option<Vec<OpeningTableShape>> {
    let shape = shape.padded();
    let silu_tensor_len = shape.seq.checked_mul(shape.intermediate)?;
    let softmax_tensor_len = shape
        .q_heads
        .checked_mul(shape.seq)?
        .checked_mul(shape.seq)?;

    let mut out = Vec::with_capacity(235);
    out.push(dense_table_shape(&commitments.hidden_out));
    out.push(dense_table_shape(&commitments.hidden_in_a));
    out.push(dense_table_shape(&commitments.hidden_in_b));
    push_ra_table_shapes(&mut out, &commitments.silu_lookup_ra, silu_tensor_len)?;
    push_ra_table_shapes(&mut out, &commitments.softmax_lookup_ra, softmax_tensor_len)?;
    push_bit_table_shapes(&mut out, &commitments.bits.down_proj_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.silu_up_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.silu_input_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.silu_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.gate_proj_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.up_proj_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.rms_norm_mlp_norm_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.rms_norm_mlp_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.o_proj_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.pv_matmul_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.softmax_floor_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.softmax_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.softmax_exp_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.qk_score_dot_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.qk_score_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.q_rope_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.q_rope_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.k_rope_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.k_rope_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.q_norm_norm_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.q_norm_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.k_norm_norm_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.k_norm_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.q_proj_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.k_proj_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.v_proj_output_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.rms_norm_atten_norm_frac_bits);
    push_bit_table_shapes(&mut out, &commitments.bits.rms_norm_atten_output_frac_bits);
    Some(out)
}

fn prepare_joint_chunk_opening(
    claims: &LayerOpeningClaims,
    witnesses: &LayerOpeningWitnesses,
    shape: LayerShape,
    commitments: &LayerCommitments<HyperKZGCommitment<Bn254>>,
    reduction_point: &[Fr],
    reduction_challenges: &[<Fr as JoltField>::Challenge],
    gamma_powers: &[Fr],
) -> Option<PreparedChunkOpening> {
    let chunk_specs = chunk_opening_specs(
        claims,
        shape,
        commitments,
        reduction_point,
        reduction_challenges,
        gamma_powers,
    )?;
    let tables = layer_opening_tables(witnesses, shape)?;
    (chunk_specs.len() == tables.len()).then_some(())?;
    let mut coefficients = Vec::new();
    let mut values = Vec::new();
    let mut chunk_evals = Vec::new();
    let mut pcs_point = None;
    let mut eq_table_cache = Vec::new();
    for (claim_index, (spec, table)) in chunk_specs.into_iter().zip_eq(tables).enumerate() {
        let chunk_values = match chunk_opening_values(table, spec.chunk_size) {
            Some(values) => values,
            None => {
                opening_debug(|| {
                    eprintln!(
                        "chunk values failed: claim_index={claim_index}, chunk_size={}, table_len={}",
                        spec.chunk_size,
                        table.len(),
                    );
                });
                return None;
            }
        };
        for (chunk_index, chunk_value) in chunk_values {
            let coefficient = match eq_index_eval(chunk_index, spec.selector_point) {
                Some(selector) => spec.coefficient * selector,
                None => {
                    opening_debug(|| {
                        eprintln!(
                            "chunk selector failed: claim_index={claim_index}, chunk_index={chunk_index}, selector_vars={}",
                            spec.selector_point.len(),
                        );
                    });
                    return None;
                }
            };
            let chunk_eval = match chunk_eval_at_point(
                &spec.inner_point,
                &chunk_value,
                &mut eq_table_cache,
            ) {
                Some(value) => value,
                None => {
                    opening_debug(|| {
                        eprintln!(
                            "chunk reduction_point failed: claim_index={claim_index}, chunk_index={chunk_index}, inner_vars={}",
                            spec.inner_point.len(),
                        );
                    });
                    return None;
                }
            };
            if let Some(reduction_challenges) = &pcs_point {
                (reduction_challenges == &spec.inner_challenges).then_some(())?;
            } else {
                pcs_point = Some(spec.inner_challenges.clone());
            }
            coefficients.push(coefficient);
            values.push(chunk_value);
            chunk_evals.push(chunk_eval);
        }
    }
    Some(PreparedChunkOpening {
        coefficients,
        values,
        chunk_evals,
        pcs_point: pcs_point?,
    })
}

fn chunk_opening_specs<'a>(
    claims: &'a LayerOpeningClaims,
    shape: LayerShape,
    commitments: &'a LayerCommitments<HyperKZGCommitment<Bn254>>,
    reduction_point: &'a [Fr],
    reduction_challenges: &'a [<Fr as JoltField>::Challenge],
    gamma_powers: &'a [Fr],
) -> Option<Vec<ChunkOpeningSpec<'a>>> {
    validate_layer_opening_claim_domains(claims, shape)?;
    let claims = flatten_layer_claims(claims);
    let tables = layer_opening_table_shapes(commitments, shape)?;
    (claims.len() == tables.len() && gamma_powers.len() == claims.len()).then_some(())?;
    (reduction_point.len() == reduction_challenges.len()).then_some(())?;

    claims
        .into_iter()
        .zip_eq(tables)
        .zip_eq(gamma_powers)
        .enumerate()
        .map(|(claim_index, ((claim, table), gamma))| {
            let claim_vars = claim.point.len();
            (reduction_point.len() >= claim_vars).then_some(())?;
            let active_point = &reduction_point[..claim_vars];
            let active_challenges = &reduction_challenges[..claim_vars];
            let projection = match chunk_opening_projection(
                table,
                reduction_point,
                reduction_challenges,
                active_point,
                active_challenges,
            ) {
                Some(projection) => projection,
                None => {
                    opening_debug(|| {
                        eprintln!(
                            "chunk opening projection failed: claim_index={claim_index}, table={table:?}, joint_vars={}, claim_vars={claim_vars}",
                            reduction_point.len(),
                        )
                    });
                    return None;
                }
            };
            Some(ChunkOpeningSpec {
                coefficient: *gamma
                    * zero_suffix_eval(&reduction_point[projection.suffix_start..])
                    * eq_eval(&claim.point, active_point)?,
                inner_point: projection.inner_point,
                inner_challenges: projection.inner_challenges,
                selector_point: projection.selector_point,
                chunk_size: table.chunk_size,
            })
        })
        .collect()
}

struct ChunkOpeningProjection<'a> {
    inner_point: Vec<Fr>,
    inner_challenges: Vec<<Fr as JoltField>::Challenge>,
    selector_point: &'a [Fr],
    suffix_start: usize,
}

fn chunk_opening_projection<'a>(
    table: OpeningTableShape,
    reduction_point: &'a [Fr],
    reduction_challenges: &'a [<Fr as JoltField>::Challenge],
    active_point: &'a [Fr],
    active_challenges: &'a [<Fr as JoltField>::Challenge],
) -> Option<ChunkOpeningProjection<'a>> {
    match table.kind {
        OpeningTableKind::Dense => {
            let chunk_vars = table.chunk_size.ilog2() as usize;
            if active_point.len() >= chunk_vars {
                let (inner_point, selector_point) = active_point.split_at(chunk_vars);
                Some(ChunkOpeningProjection {
                    inner_point: inner_point.to_vec(),
                    inner_challenges: active_challenges[..chunk_vars].to_vec(),
                    selector_point,
                    suffix_start: active_point.len(),
                })
            } else {
                (reduction_point.len() >= chunk_vars && reduction_challenges.len() >= chunk_vars)
                    .then_some(())?;
                Some(ChunkOpeningProjection {
                    inner_point: reduction_point[..chunk_vars].to_vec(),
                    inner_challenges: reduction_challenges[..chunk_vars].to_vec(),
                    selector_point: &active_point[0..0],
                    suffix_start: chunk_vars,
                })
            }
        }
        OpeningTableKind::Ra {
            tensor_len,
            address_space,
        } => {
            let tensor_vars = tensor_len.ilog2() as usize;
            let address_vars = address_space.ilog2() as usize;
            let chunk_tensor_vars = table.chunk_size.ilog2() as usize;
            (active_point.len() >= address_vars + tensor_vars).then_some(())?;
            let (address_point, tensor_point) = active_point.split_at(address_vars);
            let (address_challenges, tensor_challenges) = active_challenges.split_at(address_vars);
            (tensor_point.len() == tensor_vars).then_some(())?;
            if tensor_vars < chunk_tensor_vars {
                let inner_vars = address_vars + chunk_tensor_vars;
                (reduction_point.len() >= inner_vars && reduction_challenges.len() >= inner_vars)
                    .then_some(())?;
                return Some(ChunkOpeningProjection {
                    inner_point: reduction_point[..inner_vars].to_vec(),
                    inner_challenges: reduction_challenges[..inner_vars].to_vec(),
                    selector_point: &active_point[0..0],
                    suffix_start: inner_vars,
                });
            }
            let (inner_tensor_point, selector_point) = tensor_point.split_at(chunk_tensor_vars);
            let inner_tensor_challenges = &tensor_challenges[..chunk_tensor_vars];
            Some(ChunkOpeningProjection {
                inner_point: [address_point, inner_tensor_point].concat(),
                inner_challenges: [address_challenges, inner_tensor_challenges].concat(),
                selector_point,
                suffix_start: active_point.len(),
            })
        }
    }
}

fn chunk_opening_values<'a>(
    table: OpeningTable<'a>,
    chunk_size: usize,
) -> Option<Vec<(usize, ChunkValues)>> {
    match table {
        OpeningTable::I32(values) => Some(
            values
                .chunks(chunk_size)
                .enumerate()
                .map(|(chunk_index, chunk)| {
                    let mut values = chunk
                        .iter()
                        .map(|value| Fr::from_i32(*value))
                        .collect::<Vec<_>>();
                    values.resize(chunk_size, Fr::zero());
                    (chunk_index, ChunkValues::Field(values))
                })
                .collect(),
        ),
        OpeningTable::Bool(values) => Some(
            values
                .chunks(chunk_size)
                .enumerate()
                .map(|(chunk_index, chunk)| {
                    let mut values = chunk.to_vec();
                    values.resize(chunk_size, false);
                    (chunk_index, ChunkValues::Bool(BoolState::OwnedBits(values)))
                })
                .collect(),
        ),
        OpeningTable::Ra {
            table,
            tensor_len,
            address_space,
        } => {
            let selected = selected_entries_from_dense_ra(table, tensor_len)?;
            Some(
                selected
                    .chunks(chunk_size)
                    .enumerate()
                    .map(|(chunk_index, chunk)| {
                        let mut selected = chunk.to_vec();
                        selected.resize(chunk_size, usize::MAX);
                        (
                            chunk_index,
                            ChunkValues::Ra(RaState::Selected {
                                selected,
                                layout: RaLayout::PcsCoeffs {
                                    cycle_len: chunk_size,
                                },
                                len: address_space * chunk_size,
                            }),
                        )
                    })
                    .collect(),
            )
        }
    }
}

fn selected_entries_from_dense_ra(table: &[u8], tensor_len: usize) -> Option<Vec<usize>> {
    let entries = ra_address_space(table, tensor_len)?;
    let mut selected = vec![usize::MAX; tensor_len];
    for entry in 0..entries {
        let offset = entry * tensor_len;
        for tensor_index in 0..tensor_len {
            match table[offset + tensor_index] {
                0 => {}
                1 if selected[tensor_index] == usize::MAX => selected[tensor_index] = entry,
                _ => return None,
            }
        }
    }
    selected
        .iter()
        .all(|entry| *entry != usize::MAX)
        .then_some(selected)
}

fn ra_address_space(table: &[u8], tensor_len: usize) -> Option<usize> {
    (tensor_len > 0 && table.len().is_multiple_of(tensor_len)).then_some(())?;
    let address_space = table.len() / tensor_len;
    (address_space > 0 && address_space.is_power_of_two()).then_some(address_space)
}

fn materialized_chunk_rlc(
    chunk_values: &[ChunkValues],
    gammas: &[Fr],
) -> Option<(MultilinearPolynomial<Fr>, bool)> {
    (chunk_values.len() == gammas.len()).then_some(())?;
    let len = chunk_values.iter().map(ChunkValues::len).max()?;
    let mut rlc = vec![Fr::zero(); len];
    for (values, gamma) in chunk_values.iter().zip_eq(gammas) {
        values.add_scaled_to(&mut rlc, *gamma)?;
    }
    let is_zero = rlc.iter().all(Zero::is_zero);
    Some((MultilinearPolynomial::from(rlc), is_zero))
}

fn prove_joint_chunk_opening<Tr>(
    setup: &HyperKZGProverKey<Bn254>,
    prepared: &PreparedChunkOpening,
    gammas: &[Fr],
    transcript: &mut Tr,
) -> Option<Vec<HyperKZGProof<Bn254>>>
where
    Tr: Transcript,
{
    (prepared.values.len() == prepared.chunk_evals.len()
        && prepared.chunk_evals.len() == gammas.len())
    .then_some(())?;
    let (rlc, is_zero) = profile::measure("chunked_pcs_open.materialize_rlc", || {
        materialized_chunk_rlc(&prepared.values, gammas)
    })?;
    let pcs_point = reversed_challenges(&prepared.pcs_point);
    let actual = rlc.evaluate(&pcs_point);
    let expected = prepared
        .chunk_evals
        .iter()
        .zip_eq(gammas)
        .map(|(value, gamma)| *gamma * *value)
        .sum::<Fr>();
    if actual != expected {
        opening_debug(|| {
            let kinds = prepared
                .values
                .iter()
                .take(8)
                .map(ChunkValues::kind)
                .collect::<Vec<_>>();
            eprintln!(
                "chunk pcs: joint rlc reduction_point mismatch: chunks={}, point_vars={}, first_kinds={:?}",
                prepared.values.len(),
                prepared.pcs_point.len(),
                kinds,
            )
        });
        return None;
    }
    if is_zero {
        opening_debug(|| eprintln!("chunk pcs: joint rlc is zero, skip open"));
        return Some(Vec::new());
    }
    HyperKZG::open(setup, &rlc, &pcs_point, transcript)
        .map(|proof| vec![proof])
        .map_err(|err| {
            opening_debug(|| eprintln!("chunk pcs: hyperkzg open failed: err={err:?}"));
            err
        })
        .ok()
}

fn direct_chunk_claim(coefficients: &[Fr], chunk_evals: &[Fr]) -> Option<Fr> {
    (coefficients.len() == chunk_evals.len()).then_some(())?;
    coefficients
        .iter()
        .zip_eq(chunk_evals)
        .map(|(coefficient, value)| Some(*coefficient * *value))
        .sum()
}

fn push_ra_tables<'a>(
    out: &mut Vec<OpeningTable<'a>>,
    table: &'a [u8],
    tensor_len: usize,
) -> Option<()> {
    let address_space = ra_address_space(table, tensor_len)?;
    out.extend(std::iter::repeat_n(
        OpeningTable::Ra {
            table,
            tensor_len,
            address_space,
        },
        4,
    ));
    Some(())
}

fn push_bit_tables<'a>(out: &mut Vec<OpeningTable<'a>>, tables: &'a BitOpeningWitnesses) {
    out.extend(tables.iter().map(|table| OpeningTable::Bool(table)));
}

fn dense_table_shape(
    commitments: &ChunkedCommitments<HyperKZGCommitment<Bn254>>,
) -> OpeningTableShape {
    OpeningTableShape {
        kind: OpeningTableKind::Dense,
        chunk_size: commitments.chunk_size,
    }
}

fn ra_table_shape(
    commitments: &ChunkedCommitments<HyperKZGCommitment<Bn254>>,
    tensor_len: usize,
) -> Option<OpeningTableShape> {
    let address_space = commitments.address_space?;
    Some(OpeningTableShape {
        kind: OpeningTableKind::Ra {
            tensor_len,
            address_space,
        },
        chunk_size: commitments.chunk_size,
    })
}

fn push_ra_table_shapes(
    out: &mut Vec<OpeningTableShape>,
    commitments: &ChunkedCommitments<HyperKZGCommitment<Bn254>>,
    tensor_len: usize,
) -> Option<()> {
    let shape = ra_table_shape(commitments, tensor_len)?;
    out.extend(std::iter::repeat_n(shape, 4));
    Some(())
}

fn push_bit_table_shapes(
    out: &mut Vec<OpeningTableShape>,
    commitments: &[ChunkedCommitments<HyperKZGCommitment<Bn254>>; 8],
) {
    out.extend(commitments.iter().map(dense_table_shape));
}

fn validate_layer_opening_claim_domains(
    claims: &LayerOpeningClaims,
    shape: LayerShape,
) -> Option<()> {
    let shape = shape.padded();
    let hidden = vars(shape.seq.checked_mul(shape.hidden)?)?;
    let intermediate = vars(shape.seq.checked_mul(shape.intermediate)?)?;
    let q_tensor = vars(
        shape
            .seq
            .checked_mul(shape.q_heads)?
            .checked_mul(shape.head_dim)?,
    )?;
    let kv_tensor = vars(
        shape
            .seq
            .checked_mul(shape.kv_heads)?
            .checked_mul(shape.head_dim)?,
    )?;
    let score = vars(
        shape
            .q_heads
            .checked_mul(shape.seq)?
            .checked_mul(shape.seq)?,
    )?;
    let context = q_tensor;
    let q_rope = q_tensor;
    let k_rope = kv_tensor;

    claim_vars(&claims.hidden_out, hidden)?;
    claim_vars(&claims.hidden_in_a, hidden)?;
    claim_vars(&claims.hidden_in_b, hidden)?;
    ra_claim_vars_at_least(&claims.silu_lookup_ra, intermediate)?;
    ra_claim_vars_at_least(&claims.softmax_lookup_ra, score)?;
    bit_claim_vars(&claims.down_proj_output_frac_bits, hidden)?;
    bit_claim_vars(&claims.silu_up_output_frac_bits, intermediate)?;
    bit_claim_vars(&claims.silu_input_frac_bits, intermediate)?;
    bit_claim_vars(&claims.silu_output_frac_bits, intermediate)?;
    bit_claim_vars(&claims.gate_proj_output_frac_bits, intermediate)?;
    bit_claim_vars(&claims.up_proj_output_frac_bits, intermediate)?;
    bit_claim_vars(&claims.rms_norm_mlp_norm_frac_bits, hidden)?;
    bit_claim_vars(&claims.rms_norm_mlp_output_frac_bits, hidden)?;
    bit_claim_vars(&claims.o_proj_output_frac_bits, hidden)?;
    bit_claim_vars(&claims.pv_matmul_output_frac_bits, context)?;
    bit_claim_vars(&claims.softmax_floor_frac_bits, score)?;
    bit_claim_vars(&claims.softmax_output_frac_bits, score)?;
    bit_claim_vars(&claims.softmax_exp_frac_bits, score)?;
    bit_claim_vars(&claims.qk_score_dot_output_frac_bits, score)?;
    bit_claim_vars(&claims.qk_score_output_frac_bits, score)?;
    bit_claim_vars(&claims.q_rope_first_half_output_frac_bits, q_rope)?;
    bit_claim_vars(&claims.q_rope_second_half_output_frac_bits, q_rope)?;
    bit_claim_vars(&claims.k_rope_first_half_output_frac_bits, k_rope)?;
    bit_claim_vars(&claims.k_rope_second_half_output_frac_bits, k_rope)?;
    bit_claim_vars(&claims.q_norm_norm_frac_bits, q_tensor)?;
    bit_claim_vars(&claims.q_norm_output_frac_bits, q_tensor)?;
    bit_claim_vars(&claims.k_norm_norm_frac_bits, kv_tensor)?;
    bit_claim_vars(&claims.k_norm_output_frac_bits, kv_tensor)?;
    bit_claim_vars(&claims.q_proj_output_frac_bits, q_tensor)?;
    bit_claim_vars(&claims.k_proj_output_frac_bits, kv_tensor)?;
    bit_claim_vars(&claims.v_proj_output_frac_bits, kv_tensor)?;
    bit_claim_vars(&claims.rms_norm_atten_norm_frac_bits, hidden)?;
    bit_claim_vars(&claims.rms_norm_atten_output_frac_bits, hidden)?;
    Some(())
}

fn vars(len: usize) -> Option<usize> {
    (len > 0 && len.is_power_of_two()).then_some(len.ilog2() as usize)
}

fn claim_vars(claim: &EvalClaim, expected: usize) -> Option<()> {
    (claim.point.len() == expected).then_some(())
}

fn bit_claim_vars(claims: &BitOpeningClaims, expected: usize) -> Option<()> {
    claims
        .iter()
        .all(|claim| claim.point.len() == expected)
        .then_some(())
}

fn ra_claim_vars_at_least(claims: &RaOpeningClaims, tensor_vars: usize) -> Option<()> {
    let vars = [
        claims.read.point.len(),
        claims.virtual_claim.point.len(),
        claims.hamming_weight.point.len(),
        claims.booleanity.point.len(),
    ];
    vars.into_iter()
        .all(|point_vars| point_vars >= tensor_vars)
        .then_some(())
}

#[derive(Clone, Copy)]
enum OpeningTable<'a> {
    I32(&'a [i32]),
    Ra {
        table: &'a [u8],
        tensor_len: usize,
        address_space: usize,
    },
    Bool(&'a [bool]),
}

impl OpeningTable<'_> {
    fn len(&self) -> usize {
        match self {
            Self::I32(values) => values.len(),
            Self::Ra { table, .. } => table.len(),
            Self::Bool(values) => values.len(),
        }
    }
}

struct OpeningTermState<'a> {
    point: &'a [Fr],
    max_vars: usize,
    suffix_bound: Fr,
    current_claim: Fr,
    values: OpeningValues<'a>,
    eq: GruenSplitEqPolynomial<Fr>,
}

impl<'a> OpeningTermState<'a> {
    fn new(claim: &'a EvalClaim, table: OpeningTable<'a>, max_vars: usize) -> Option<Self> {
        (table.len().is_power_of_two()).then_some(())?;
        (table.len().ilog2() as usize == claim.point.len()).then_some(())?;
        (claim.point.len() <= max_vars).then_some(())?;
        let values = OpeningValues::new(table)?;
        let split_eq_point = claim.point.iter().rev().copied().collect::<Vec<_>>();
        let eq = GruenSplitEqPolynomial::new(&split_eq_point, BindingOrder::LowToHigh);
        let mut state = Self {
            point: &claim.point,
            max_vars,
            suffix_bound: Fr::one(),
            current_claim: Fr::zero(),
            values,
            eq,
        };
        state.refresh_current_claim()?;
        Some(state)
    }

    fn vars(&self) -> usize {
        self.point.len()
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn current_claim(&self) -> Fr {
        self.suffix_bound * self.current_claim
    }

    fn refresh_current_claim(&mut self) -> Option<()> {
        self.current_claim = if self.values.len() == 1 && self.eq.len() == 1 {
            self.eq.get_current_scalar() * self.values.value_at(0)
        } else {
            let round = self.actual_round_poly()?;
            round.eval(Fr::zero()) + round.eval(Fr::one())
        };
        Some(())
    }

    fn round_poly(&self, round: usize) -> RoundPolynomial<3> {
        if round >= self.vars() {
            let claim = self.current_claim();
            return RoundPolynomial {
                coeffs: [claim, -claim, Fr::zero()],
            };
        }

        self.actual_round_poly()
            .map(|round| scale_round_poly(round, self.suffix_bound))
            .unwrap_or_else(|| RoundPolynomial {
                coeffs: [Fr::zero(); 3],
            })
    }

    fn actual_round_poly(&self) -> Option<RoundPolynomial<3>> {
        (self.eq.len() == self.values.len() && self.values.len() % 2 == 0).then_some(())?;
        let relation_coeffs = self.eq.par_fold_out_in_unreduced::<9, 2>(&|index| {
            let lower = self.values.value_at(2 * index);
            let upper = self.values.value_at(2 * index + 1);
            [lower, upper - lower]
        });
        Some(linear_relation_times_eq(
            relation_coeffs,
            self.eq.get_current_w(),
            self.eq.get_current_scalar(),
        ))
    }

    fn bind(&mut self, round: usize, challenge: <Fr as JoltField>::Challenge, r: Fr) {
        if round >= self.vars() {
            self.suffix_bound *= Fr::one() - r;
            return;
        }

        self.values.bind(r);
        self.eq.bind(challenge);
        self.refresh_current_claim()
            .expect("bound opening term remains well formed");
    }

    fn final_claim(self) -> Option<Fr> {
        if self.values.len() == 1 && self.eq.len() == 1 {
            Some(self.values.value_at(0))
        } else {
            opening_debug(|| {
                eprintln!(
                    "opening term was not fully bound: point_vars={}, max_vars={}, values_len={}, eq_len={}",
                    self.vars(),
                    self.max_vars,
                    self.values.len(),
                    self.eq.len(),
                );
            });
            None
        }
    }
}

struct PreparedChunkOpening {
    coefficients: Vec<Fr>,
    values: Vec<ChunkValues>,
    chunk_evals: Vec<Fr>,
    pcs_point: Vec<<Fr as JoltField>::Challenge>,
}

struct ChunkOpeningSpec<'a> {
    coefficient: Fr,
    inner_point: Vec<Fr>,
    inner_challenges: Vec<<Fr as JoltField>::Challenge>,
    selector_point: &'a [Fr],
    chunk_size: usize,
}

#[derive(Debug, Clone, Copy)]
struct OpeningTableShape {
    kind: OpeningTableKind,
    chunk_size: usize,
}

#[derive(Debug, Clone, Copy)]
enum OpeningTableKind {
    Dense,
    Ra {
        tensor_len: usize,
        address_space: usize,
    },
}

fn chunk_value_at_point(reduction_challenges: &[Fr], values: &ChunkValues) -> Option<Fr> {
    (values.len().is_power_of_two()).then_some(())?;
    (values.len().ilog2() as usize == reduction_challenges.len()).then_some(())?;
    if let Some(claim) = values.semantic_claim(reduction_challenges) {
        Some(claim)
    } else if values.len() == 1 {
        Some(values.value_at(0))
    } else {
        let split_eq_point = reduction_challenges
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        let eq = GruenSplitEqPolynomial::new(&split_eq_point, BindingOrder::LowToHigh);
        let round = chunk_values_round_poly(values, &eq)?;
        Some(round.eval(Fr::zero()) + round.eval(Fr::one()))
    }
}

fn chunk_eval_at_point(
    reduction_challenges: &[Fr],
    values: &ChunkValues,
    eq_table_cache: &mut Vec<(Vec<Fr>, Vec<Fr>)>,
) -> Option<Fr> {
    if let ChunkValues::Bool(values) = values {
        (values.len().is_power_of_two()).then_some(())?;
        (values.len().ilog2() as usize == reduction_challenges.len()).then_some(())?;
        let eq_index = cached_eq_table_index(eq_table_cache, reduction_challenges);
        return values.eval_with_eq_table(&eq_table_cache[eq_index].1);
    }
    chunk_value_at_point(reduction_challenges, values)
}

fn cached_eq_table_index(
    cache: &mut Vec<(Vec<Fr>, Vec<Fr>)>,
    reduction_challenges: &[Fr],
) -> usize {
    if let Some(index) = cache
        .iter()
        .position(|(cached_point, _)| cached_point == reduction_challenges)
    {
        index
    } else {
        cache.push((
            reduction_challenges.to_vec(),
            eq_table(reduction_challenges),
        ));
        cache.len() - 1
    }
}

fn eq_table(reduction_challenges: &[Fr]) -> Vec<Fr> {
    let mut table = vec![Fr::one()];
    for r in reduction_challenges {
        let lower_scale = Fr::one() - r;
        let upper_scale = *r;
        let len = table.len();
        table.resize(len * 2, Fr::zero());
        for index in (0..len).rev() {
            let value = table[index];
            table[index] = value * lower_scale;
            table[index + len] = value * upper_scale;
        }
    }
    table
}

fn chunk_values_round_poly(
    values: &ChunkValues,
    eq: &GruenSplitEqPolynomial<Fr>,
) -> Option<RoundPolynomial<3>> {
    (eq.len() == values.len() && values.len() % 2 == 0).then_some(())?;
    let relation_coeffs = eq.par_fold_out_in_unreduced::<9, 2>(&|index| {
        let lower = values.value_at(2 * index);
        let upper = values.value_at(2 * index + 1);
        [lower, upper - lower]
    });
    Some(linear_relation_times_eq(
        relation_coeffs,
        eq.get_current_w(),
        eq.get_current_scalar(),
    ))
}

enum OpeningValues<'a> {
    I32(Vec<Fr>),
    Bool(BoolState<'a>),
    Ra(RaState),
}

impl<'a> OpeningValues<'a> {
    fn new(table: OpeningTable<'a>) -> Option<Self> {
        match table {
            OpeningTable::I32(values) => Some(Self::I32(
                values.iter().map(|value| Fr::from_i32(*value)).collect(),
            )),
            OpeningTable::Bool(values) => Some(Self::Bool(BoolState::Bits(values))),
            OpeningTable::Ra {
                table, tensor_len, ..
            } => Some(Self::Ra(
                RaState::from_onehot(table, tensor_len)
                    .unwrap_or_else(|| RaState::from_u8_values(table)),
            )),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::I32(values) => values.len(),
            Self::Bool(values) => values.len(),
            Self::Ra(values) => values.len(),
        }
    }

    fn value_at(&self, index: usize) -> Fr {
        match self {
            Self::I32(values) => values[index],
            Self::Bool(values) => values.value_at(index),
            Self::Ra(values) => values.value_at(index),
        }
    }

    fn bind(&mut self, r: Fr) {
        match self {
            Self::I32(values) => bind_field_values(values, r),
            Self::Bool(values) => values.bind(r),
            Self::Ra(values) => values.bind(r),
        }
    }
}

#[derive(Clone)]
enum ChunkValues {
    Field(Vec<Fr>),
    Bool(BoolState<'static>),
    Ra(RaState),
}

impl ChunkValues {
    fn kind(&self) -> &'static str {
        match self {
            Self::Field(_) => "field",
            Self::Bool(_) => "bool",
            Self::Ra(_) => "ra",
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Field(values) => values.len(),
            Self::Bool(values) => values.len(),
            Self::Ra(values) => values.len(),
        }
    }

    fn value_at(&self, index: usize) -> Fr {
        match self {
            Self::Field(values) => values[index],
            Self::Bool(values) => values.value_at(index),
            Self::Ra(values) => values.value_at(index),
        }
    }

    fn semantic_claim(&self, reduction_challenges: &[Fr]) -> Option<Fr> {
        match self {
            Self::Ra(values) => values.semantic_claim(reduction_challenges),
            Self::Field(_) | Self::Bool(_) => None,
        }
    }

    fn add_scaled_to(&self, out: &mut [Fr], scale: Fr) -> Option<()> {
        (out.len() >= self.len()).then_some(())?;
        match self {
            Self::Field(values) => {
                for (out, value) in out.iter_mut().zip_eq(values) {
                    *out += scale * *value;
                }
            }
            Self::Bool(values) => values.add_scaled_to(out, scale)?,
            Self::Ra(values) => values.add_scaled_to(out, scale)?,
        }
        Some(())
    }
}

#[derive(Clone)]
enum BoolState<'a> {
    Bits(&'a [bool]),
    OwnedBits(Vec<bool>),
    AffineTags { tags: Vec<u8>, r: Fr },
    Affine2Tags { tags: Vec<u8>, values: [Fr; 16] },
    FieldValues(Vec<Fr>),
}

impl BoolState<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Bits(bits) => bits.len(),
            Self::OwnedBits(bits) => bits.len(),
            Self::AffineTags { tags, .. } => tags.len(),
            Self::Affine2Tags { tags, .. } => tags.len(),
            Self::FieldValues(values) => values.len(),
        }
    }

    fn value_at(&self, index: usize) -> Fr {
        match self {
            Self::Bits(bits) => bit_to_fr(bits[index]),
            Self::OwnedBits(bits) => bit_to_fr(bits[index]),
            Self::AffineTags { tags, r } => affine_tag_value(tags[index] as u16, *r),
            Self::Affine2Tags { tags, values } => values[tags[index] as usize],
            Self::FieldValues(values) => values[index],
        }
    }

    fn add_scaled_to(&self, out: &mut [Fr], scale: Fr) -> Option<()> {
        (out.len() >= self.len()).then_some(())?;
        match self {
            Self::Bits(bits) => {
                for (out, bit) in out.iter_mut().zip_eq(*bits) {
                    if *bit {
                        *out += scale;
                    }
                }
            }
            Self::OwnedBits(bits) => {
                for (out, bit) in out.iter_mut().zip_eq(bits) {
                    if *bit {
                        *out += scale;
                    }
                }
            }
            Self::AffineTags { .. } | Self::Affine2Tags { .. } | Self::FieldValues(_) => {
                for (index, out) in out.iter_mut().enumerate().take(self.len()) {
                    *out += scale * self.value_at(index);
                }
            }
        }
        Some(())
    }

    fn eval_with_eq_table(&self, eq: &[Fr]) -> Option<Fr> {
        (eq.len() == self.len()).then_some(())?;
        match self {
            Self::Bits(bits) => Some(
                bits.par_iter()
                    .zip_eq(eq.par_iter())
                    .filter_map(|(bit, eq)| bit.then_some(*eq))
                    .sum(),
            ),
            Self::OwnedBits(bits) => Some(
                bits.par_iter()
                    .zip_eq(eq.par_iter())
                    .filter_map(|(bit, eq)| bit.then_some(*eq))
                    .sum(),
            ),
            Self::AffineTags { .. } | Self::Affine2Tags { .. } | Self::FieldValues(_) => Some(
                (0..self.len())
                    .into_par_iter()
                    .map(|index| self.value_at(index) * eq[index])
                    .sum(),
            ),
        }
    }

    fn bind(&mut self, r: Fr) {
        match self {
            Self::Bits(bits) => {
                let tags = (0..bits.len() / 2)
                    .map(|index| affine_tag_from_bits(bits[2 * index], bits[2 * index + 1]))
                    .collect();
                *self = Self::AffineTags { tags, r };
            }
            Self::OwnedBits(bits) => {
                let tags = (0..bits.len() / 2)
                    .map(|index| affine_tag_from_bits(bits[2 * index], bits[2 * index + 1]))
                    .collect();
                *self = Self::AffineTags { tags, r };
            }
            Self::AffineTags { tags, r: r0 } => {
                let next = (0..tags.len() / 2)
                    .map(|index| {
                        affine2_tag_from_affine1_pair(tags[2 * index], tags[2 * index + 1])
                    })
                    .collect();
                *self = Self::Affine2Tags {
                    tags: next,
                    values: affine2_value_table(*r0, r),
                };
            }
            Self::Affine2Tags { tags, values } => {
                let bind_values = affine2_bind_table(values, r);
                let next = (0..tags.len() / 2)
                    .map(|index| {
                        bind_affine2_tag(tags[2 * index], tags[2 * index + 1], &bind_values)
                    })
                    .collect();
                *self = Self::FieldValues(next);
            }
            Self::FieldValues(values) => bind_field_values(values, r),
        }
    }
}

#[derive(Clone)]
enum RaState {
    Selected {
        selected: Vec<usize>,
        layout: RaLayout,
        len: usize,
    },
    AffineTags {
        tags: Vec<u8>,
        r: Fr,
    },
    Affine2Tags {
        tags: Vec<u8>,
        values: [Fr; 16],
    },
    FieldValues(Vec<Fr>),
}

#[derive(Clone, Copy)]
enum RaLayout {
    OpeningReduction { address_space: usize },
    PcsCoeffs { cycle_len: usize },
}

impl RaState {
    fn from_u8_values(table: &[u8]) -> Self {
        Self::FieldValues(table.iter().map(|value| Fr::from(*value)).collect())
    }

    fn from_onehot(table: &[u8], tensor_len: usize) -> Option<Self> {
        (tensor_len > 0 && table.len().is_multiple_of(tensor_len)).then_some(())?;
        let entries = table.len() / tensor_len;
        (entries > 0).then_some(())?;
        let mut selected = vec![usize::MAX; tensor_len];
        for entry in 0..entries {
            let offset = entry * tensor_len;
            for tensor_index in 0..tensor_len {
                match table[offset + tensor_index] {
                    0 => {}
                    1 if selected[tensor_index] == usize::MAX => {
                        selected[tensor_index] = entry;
                    }
                    _ => return None,
                }
            }
        }
        selected
            .iter()
            .all(|entry| *entry != usize::MAX)
            .then_some(())?;
        Some(Self::Selected {
            selected,
            layout: RaLayout::OpeningReduction {
                address_space: entries,
            },
            len: table.len(),
        })
    }

    fn len(&self) -> usize {
        match self {
            Self::Selected { len, .. } => *len,
            Self::AffineTags { tags, .. } => tags.len(),
            Self::Affine2Tags { tags, .. } => tags.len(),
            Self::FieldValues(values) => values.len(),
        }
    }

    fn value_at(&self, index: usize) -> Fr {
        match self {
            Self::Selected {
                selected, layout, ..
            } => {
                if selected_value(index, selected, *layout) {
                    Fr::one()
                } else {
                    Fr::zero()
                }
            }
            Self::AffineTags { tags, r } => affine_tag_value(tags[index] as u16, *r),
            Self::Affine2Tags { tags, values } => values[tags[index] as usize],
            Self::FieldValues(values) => values[index],
        }
    }

    fn add_scaled_to(&self, out: &mut [Fr], scale: Fr) -> Option<()> {
        (out.len() >= self.len()).then_some(())?;
        match self {
            Self::Selected {
                selected,
                layout,
                len,
            } => {
                if let RaLayout::PcsCoeffs { cycle_len } = layout {
                    let address_space = len.checked_div(*cycle_len)?;
                    (selected.len() == *cycle_len).then_some(())?;
                    for (tensor_index, entry) in selected.iter().copied().enumerate() {
                        if entry != usize::MAX {
                            (entry < address_space).then_some(())?;
                            out[tensor_index * address_space + entry] += scale;
                        }
                    }
                } else {
                    for (index, out) in out.iter_mut().enumerate().take(*len) {
                        *out += scale * self.value_at(index);
                    }
                }
            }
            Self::AffineTags { .. } | Self::Affine2Tags { .. } | Self::FieldValues(_) => {
                for (index, out) in out.iter_mut().enumerate().take(self.len()) {
                    *out += scale * self.value_at(index);
                }
            }
        }
        Some(())
    }

    fn bind(&mut self, r: Fr) {
        match self {
            Self::Selected {
                selected,
                layout,
                len,
            } => {
                let tags = (0..*len / 2)
                    .map(|index| {
                        affine_tag_from_bits(
                            selected_value(2 * index, selected, *layout),
                            selected_value(2 * index + 1, selected, *layout),
                        )
                    })
                    .collect();
                *self = Self::AffineTags { tags, r };
            }
            Self::AffineTags { tags, r: r0 } => {
                let next = (0..tags.len() / 2)
                    .map(|index| {
                        affine2_tag_from_affine1_pair(tags[2 * index], tags[2 * index + 1])
                    })
                    .collect();
                *self = Self::Affine2Tags {
                    tags: next,
                    values: affine2_value_table(*r0, r),
                };
            }
            Self::Affine2Tags { tags, values } => {
                let bind_values = affine2_bind_table(values, r);
                let next = (0..tags.len() / 2)
                    .map(|index| {
                        bind_affine2_tag(tags[2 * index], tags[2 * index + 1], &bind_values)
                    })
                    .collect();
                *self = Self::FieldValues(next);
            }
            Self::FieldValues(values) => bind_field_values(values, r),
        }
    }

    fn semantic_claim(&self, reduction_challenges: &[Fr]) -> Option<Fr> {
        let Self::Selected {
            selected,
            layout: RaLayout::PcsCoeffs { cycle_len },
            len,
        } = self
        else {
            return None;
        };
        (selected.len() == *cycle_len).then_some(())?;
        let address_space = len.checked_div(*cycle_len)?;
        let address_vars = address_space.ilog2() as usize;
        (reduction_challenges.len() == address_vars + cycle_len.ilog2() as usize).then_some(())?;
        let (address_point, cycle_point) = reduction_challenges.split_at(address_vars);
        selected
            .iter()
            .copied()
            .enumerate()
            .map(|(cycle, address)| {
                if address == usize::MAX {
                    Some(Fr::zero())
                } else {
                    Some(
                        eq_index_eval(address, address_point)? * eq_index_eval(cycle, cycle_point)?,
                    )
                }
            })
            .sum()
    }
}

fn selected_value(index: usize, selected: &[usize], layout: RaLayout) -> bool {
    match layout {
        RaLayout::OpeningReduction { address_space } => {
            let address = index % address_space;
            let cycle = index / address_space;
            selected[cycle] == address
        }
        RaLayout::PcsCoeffs { cycle_len } => {
            let address = index / cycle_len;
            let cycle = index % cycle_len;
            selected[cycle] == address
        }
    }
}

fn bit_to_fr(bit: bool) -> Fr {
    if bit { Fr::one() } else { Fr::zero() }
}

fn affine_tag_from_bits(lower: bool, upper: bool) -> u8 {
    match (lower, upper) {
        (false, false) => 0b00,
        (true, true) => 0b01,
        (false, true) => 0b10,
        (true, false) => 0b11,
    }
}

fn affine_tag_value(tag: u16, r: Fr) -> Fr {
    match tag {
        0b00 => Fr::zero(),
        0b01 => Fr::one(),
        0b10 => r,
        0b11 => Fr::one() - r,
        _ => unreachable!("affine tag is two bits"),
    }
}

fn affine2_tag_from_affine1_pair(lower: u8, upper: u8) -> u8 {
    lower | (upper << 2)
}

fn affine2_value_table(r1: Fr, r2: Fr) -> [Fr; 16] {
    std::array::from_fn(|tag| affine2_tag_value(tag as u16, r1, r2))
}

fn affine2_tag_value(tag: u16, r1: Fr, r2: Fr) -> Fr {
    let lower = affine_tag_value(tag & 0b11, r1);
    let upper = affine_tag_value((tag >> 2) & 0b11, r1);
    lower + r2 * (upper - lower)
}

fn affine2_bind_table(values: &[Fr; 16], r: Fr) -> [Fr; 256] {
    std::array::from_fn(|pair_tag| {
        let lower = values[pair_tag & 0b1111];
        let upper = values[pair_tag >> 4];
        lower + r * (upper - lower)
    })
}

fn bind_affine2_tag(lower: u8, upper: u8, bind_values: &[Fr; 256]) -> Fr {
    bind_values[lower as usize | ((upper as usize) << 4)]
}

fn bind_field_values(values: &mut Vec<Fr>, r: Fr) {
    for index in 0..values.len() / 2 {
        values[index] = line(values[2 * index], values[2 * index + 1], r);
    }
    values.truncate(values.len() / 2);
}

fn scale_round_poly(round: RoundPolynomial<3>, scale: Fr) -> RoundPolynomial<3> {
    RoundPolynomial {
        coeffs: round.coeffs.map(|coeff| scale * coeff),
    }
}

fn linear_relation_times_eq(
    relation_coeffs: [Fr; 2],
    current_w: Fr,
    current_scalar: Fr,
) -> RoundPolynomial<3> {
    let [constant, linear] = relation_coeffs;
    let eq_constant = Fr::one() - current_w;
    let eq_linear = current_w + current_w - Fr::one();
    RoundPolynomial {
        coeffs: [
            current_scalar * eq_constant * constant,
            current_scalar * (eq_constant * linear + eq_linear * constant),
            current_scalar * eq_linear * linear,
        ],
    }
}

fn opening_debug(log: impl FnOnce()) {
    if std::env::var_os("Q3_OPENING_DEBUG").is_some() {
        log();
    }
}

fn opening_reduction_round_poly(
    states: &[OpeningTermState],
    gammas: &[Fr],
    round_index: usize,
) -> RoundPolynomial<3> {
    let mut coeffs = [Fr::zero(); 3];
    for (state, gamma) in states.iter().zip_eq(gammas) {
        let round = state.round_poly(round_index);
        coeffs[0] += *gamma * round.coeffs[0];
        coeffs[1] += *gamma * round.coeffs[1];
        coeffs[2] += *gamma * round.coeffs[2];
    }
    RoundPolynomial { coeffs }
}

fn line(left: Fr, right: Fr, r: Fr) -> Fr {
    left + r * (right - left)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        commitment::{LayerBitCommitments, commit_bit_tables, commit_i32_chunks, commit_ra_chunks},
        layer::RaOpeningClaims,
    };
    use ark_bn254::Bn254;
    use ark_ff::Zero;
    use joltworks::poly::commitment::{commitment_scheme::CommitmentScheme, hyperkzg::HyperKZG};
    use joltworks::transcripts::Blake2bTranscript;
    use qwen3_verifier::{verify_layer_opening_reduction, verify_layer_pcs_opening};

    #[test]
    fn expected_claim_scales_short_domains_by_zero_suffix() {
        let shape = LayerShape {
            seq: 2,
            q_heads: 1,
            kv_heads: 1,
            head_dim: 2,
            hidden: 1,
            intermediate: 1,
        };
        let claim = EvalClaim::new(Fr::from(5_u64), vec![Fr::from(7_u64)]);
        let mut claims = layer_claims_with(shape, claim);
        claims.silu_lookup_ra = zero_ra_claims(9);
        claims.softmax_lookup_ra = zero_ra_claims(10);
        let r = vec![
            Fr::from(3_u64),
            Fr::from(11_u64),
            Fr::from(13_u64),
            Fr::from(17_u64),
            Fr::from(19_u64),
            Fr::from(23_u64),
            Fr::from(29_u64),
            Fr::from(31_u64),
            Fr::from(37_u64),
            Fr::from(41_u64),
        ];
        let gamma = vec![Fr::one(); layer_opening_claim_count(&claims)];

        let expected = pcs_joint_claim(&claims, shape, &r, &gamma).expect("expected claim");

        assert_eq!(
            expected,
            r[1..].iter().map(|r| Fr::one() - r).product::<Fr>() * Fr::from(5_u64)
        );
    }

    #[test]
    fn bool_chunk_eval_uses_same_point_order_as_eq_path() {
        let reduction_challenges = vec![Fr::from(3_u64), Fr::from(5_u64), Fr::from(7_u64)];
        let values = ChunkValues::Bool(BoolState::OwnedBits(vec![
            false, true, true, false, true, true, false, false,
        ]));

        let split_eq_point = reduction_challenges
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        let eq = GruenSplitEqPolynomial::new(&split_eq_point, BindingOrder::LowToHigh);
        let round = chunk_values_round_poly(&values, &eq).expect("old eq path evaluates");
        let expected = round.eval(Fr::zero()) + round.eval(Fr::one());

        assert_eq!(
            chunk_eval_at_point(&reduction_challenges, &values, &mut Vec::new())
                .expect("cached eq bool path evaluates"),
            expected
        );
    }

    #[test]
    fn proves_and_verifies_layer_opening_reduction_sumcheck() {
        let shape = test_shape();
        let claim = EvalClaim::new(Fr::from(5_u64), vec![Fr::one()]);
        let claims = layer_claims_with(shape, claim);
        let witnesses = layer_witnesses_with_hidden_out(vec![0, 5]);

        let mut prover_transcript = Blake2bTranscript::default();
        let output = prove_layer_opening_reduction_sumcheck(
            &claims,
            &witnesses,
            shape,
            &mut prover_transcript,
        )
        .expect("opening reduction proves");

        let mut verifier_transcript = Blake2bTranscript::default();
        let verified =
            verify_layer_opening_reduction(&claims, shape, &output.proof, &mut verifier_transcript)
                .expect("opening reduction verifies");

        assert_eq!(output.reduction_point, verified.sumcheck.point);
        assert_eq!(output.reduction_challenges, verified.sumcheck.challenges);
        assert_eq!(output.gamma_powers, verified.gamma_powers);
        pcs_joint_claim_from_values(
            &flatten_layer_claims(&claims),
            &output.proof.evals_at_reduction_point,
            &output.reduction_point,
            &output.gamma_powers,
        )
        .expect("joint claim");
    }

    #[test]
    #[ignore = "uses a small PCS domain; layer PCS opening now assumes the fixed 2^20 chunk domain"]
    fn proves_and_verifies_chunked_layer_pcs_opening() {
        type Pcs = HyperKZG<Bn254>;

        let shape = test_shape();
        let claim = EvalClaim::new(Fr::from(5_u64), vec![Fr::one()]);
        let claims = layer_claims_with(shape, claim);
        let witnesses = layer_witnesses_with_hidden_out(vec![0, 5]);

        let mut opening_transcript = Blake2bTranscript::default();
        let opening = prove_layer_opening_reduction_sumcheck(
            &claims,
            &witnesses,
            shape,
            &mut opening_transcript,
        )
        .expect("opening reduction proves");

        let setup = Pcs::setup_prover(4);
        let commitments = layer_commitments_from_witnesses(&witnesses, &setup);
        let verifier_setup = Pcs::setup_verifier(&setup);

        let mut pcs_transcript = Blake2bTranscript::default();
        let pcs = prove_chunked_layer_pcs_opening(
            &claims,
            &witnesses,
            shape,
            &opening,
            &commitments,
            &setup,
            &mut pcs_transcript,
        )
        .expect("pcs opening proves");

        let mut opening_verifier_transcript = Blake2bTranscript::default();
        let verified = verify_layer_opening_reduction(
            &claims,
            shape,
            &opening.proof,
            &mut opening_verifier_transcript,
        )
        .expect("opening reduction verifies");
        assert_eq!(opening.reduction_challenges, verified.sumcheck.challenges);

        let mut pcs_verifier_transcript = Blake2bTranscript::default();
        verify_layer_pcs_opening(
            &claims,
            shape,
            &verified,
            &commitments,
            &pcs,
            &verifier_setup,
            &mut pcs_verifier_transcript,
        )
        .expect("pcs opening verifies");
    }

    fn layer_claims_with(shape: LayerShape, claim: EvalClaim) -> LayerOpeningClaims {
        let hidden = zero_claim(vars(shape.seq * shape.hidden).unwrap());
        let intermediate = zero_claim(vars(shape.seq * shape.intermediate).unwrap());
        let tensor = zero_claim(vars(shape.seq * shape.q_heads * shape.head_dim).unwrap());
        let score = zero_claim(vars(shape.q_heads * shape.seq * shape.seq).unwrap());
        let hidden_bits = bit_claims(hidden.clone());
        let intermediate_bits = bit_claims(intermediate.clone());
        let tensor_bits = bit_claims(tensor.clone());
        let score_bits = bit_claims(score.clone());
        let silu_lookup_ra = RaOpeningClaims {
            read: one_claim(2),
            virtual_claim: one_claim(2),
            hamming_weight: one_claim(2),
            booleanity: one_claim(2),
        };
        let softmax_lookup_ra = RaOpeningClaims {
            read: one_claim(4),
            virtual_claim: one_claim(4),
            hamming_weight: one_claim(4),
            booleanity: one_claim(4),
        };
        LayerOpeningClaims {
            hidden_out: claim,
            hidden_in_a: hidden.clone(),
            hidden_in_b: hidden.clone(),
            silu_lookup_ra,
            softmax_lookup_ra,
            down_proj_output_frac_bits: hidden_bits.clone(),
            silu_up_output_frac_bits: intermediate_bits.clone(),
            silu_input_frac_bits: intermediate_bits.clone(),
            silu_output_frac_bits: intermediate_bits.clone(),
            gate_proj_output_frac_bits: intermediate_bits.clone(),
            up_proj_output_frac_bits: intermediate_bits.clone(),
            rms_norm_mlp_norm_frac_bits: hidden_bits.clone(),
            rms_norm_mlp_output_frac_bits: hidden_bits.clone(),
            o_proj_output_frac_bits: hidden_bits.clone(),
            pv_matmul_output_frac_bits: tensor_bits.clone(),
            softmax_floor_frac_bits: score_bits.clone(),
            softmax_output_frac_bits: score_bits.clone(),
            softmax_exp_frac_bits: score_bits.clone(),
            qk_score_dot_output_frac_bits: score_bits.clone(),
            qk_score_output_frac_bits: score_bits.clone(),
            q_rope_first_half_output_frac_bits: tensor_bits.clone(),
            q_rope_second_half_output_frac_bits: tensor_bits.clone(),
            k_rope_first_half_output_frac_bits: tensor_bits.clone(),
            k_rope_second_half_output_frac_bits: tensor_bits.clone(),
            q_norm_norm_frac_bits: tensor_bits.clone(),
            q_norm_output_frac_bits: tensor_bits.clone(),
            k_norm_norm_frac_bits: tensor_bits.clone(),
            k_norm_output_frac_bits: tensor_bits.clone(),
            q_proj_output_frac_bits: tensor_bits.clone(),
            k_proj_output_frac_bits: tensor_bits.clone(),
            v_proj_output_frac_bits: tensor_bits,
            rms_norm_atten_norm_frac_bits: hidden_bits.clone(),
            rms_norm_atten_output_frac_bits: hidden_bits,
        }
    }

    fn zero_claim(vars: usize) -> EvalClaim {
        EvalClaim::new(Fr::zero(), vec![Fr::zero(); vars])
    }

    fn one_claim(vars: usize) -> EvalClaim {
        EvalClaim::new(Fr::one(), vec![Fr::zero(); vars])
    }

    fn bit_claims(claim: EvalClaim) -> BitOpeningClaims {
        std::array::from_fn(|_| claim.clone())
    }

    fn zero_ra_claims(vars: usize) -> RaOpeningClaims {
        RaOpeningClaims {
            read: zero_claim(vars),
            virtual_claim: zero_claim(vars),
            hamming_weight: zero_claim(vars),
            booleanity: zero_claim(vars),
        }
    }

    fn test_shape() -> LayerShape {
        LayerShape {
            seq: 2,
            q_heads: 1,
            kv_heads: 1,
            head_dim: 2,
            hidden: 1,
            intermediate: 1,
        }
    }

    fn layer_witnesses_with_hidden_out(hidden_out: Vec<i32>) -> LayerOpeningWitnesses {
        let hidden = vec![0_i32; 2];
        let intermediate_bits = zero_bits(2);
        let hidden_bits = zero_bits(2);
        let tensor_bits = zero_bits(4);
        let score_bits = zero_bits(4);
        LayerOpeningWitnesses {
            hidden_out,
            hidden_in_a: hidden.clone(),
            hidden_in_b: hidden,
            silu_lookup_ra: dense_ra_with_selected_zero(2, 2),
            softmax_lookup_ra: dense_ra_with_selected_zero(4, 4),
            down_proj_output_frac_bits: hidden_bits.clone(),
            silu_up_output_frac_bits: intermediate_bits.clone(),
            silu_input_frac_bits: intermediate_bits.clone(),
            silu_output_frac_bits: intermediate_bits.clone(),
            gate_proj_output_frac_bits: intermediate_bits.clone(),
            up_proj_output_frac_bits: intermediate_bits,
            rms_norm_mlp_norm_frac_bits: hidden_bits.clone(),
            rms_norm_mlp_output_frac_bits: hidden_bits.clone(),
            o_proj_output_frac_bits: hidden_bits.clone(),
            pv_matmul_output_frac_bits: tensor_bits.clone(),
            softmax_floor_frac_bits: score_bits.clone(),
            softmax_output_frac_bits: score_bits.clone(),
            softmax_exp_frac_bits: score_bits.clone(),
            qk_score_dot_output_frac_bits: score_bits.clone(),
            qk_score_output_frac_bits: score_bits,
            q_rope_output_frac_bits: tensor_bits.clone(),
            k_rope_output_frac_bits: tensor_bits.clone(),
            q_norm_norm_frac_bits: tensor_bits.clone(),
            q_norm_output_frac_bits: tensor_bits.clone(),
            k_norm_norm_frac_bits: tensor_bits.clone(),
            k_norm_output_frac_bits: tensor_bits.clone(),
            q_proj_output_frac_bits: tensor_bits.clone(),
            k_proj_output_frac_bits: tensor_bits.clone(),
            v_proj_output_frac_bits: tensor_bits,
            rms_norm_atten_norm_frac_bits: hidden_bits.clone(),
            rms_norm_atten_output_frac_bits: hidden_bits,
        }
    }

    fn zero_bits(len: usize) -> BitOpeningWitnesses {
        std::array::from_fn(|_| vec![false; len])
    }

    fn dense_ra_with_selected_zero(tensor_len: usize, address_space: usize) -> Vec<u8> {
        let mut table = vec![0_u8; tensor_len * address_space];
        for tensor_index in 0..tensor_len {
            table[tensor_index] = 1;
        }
        table
    }

    fn layer_commitments_from_witnesses(
        witnesses: &LayerOpeningWitnesses,
        setup: &<Pcs as CommitmentScheme>::ProverSetup,
    ) -> LayerCommitments<HyperKZGCommitment<Bn254>> {
        let bit_commitments = LayerBitCommitments {
            down_proj_output_frac_bits: commit_bool_bits(
                &witnesses.down_proj_output_frac_bits,
                setup,
            ),
            silu_up_output_frac_bits: commit_bool_bits(&witnesses.silu_up_output_frac_bits, setup),
            silu_input_frac_bits: commit_bool_bits(&witnesses.silu_input_frac_bits, setup),
            silu_output_frac_bits: commit_bool_bits(&witnesses.silu_output_frac_bits, setup),
            gate_proj_output_frac_bits: commit_bool_bits(
                &witnesses.gate_proj_output_frac_bits,
                setup,
            ),
            up_proj_output_frac_bits: commit_bool_bits(&witnesses.up_proj_output_frac_bits, setup),
            rms_norm_mlp_norm_frac_bits: commit_bool_bits(
                &witnesses.rms_norm_mlp_norm_frac_bits,
                setup,
            ),
            rms_norm_mlp_output_frac_bits: commit_bool_bits(
                &witnesses.rms_norm_mlp_output_frac_bits,
                setup,
            ),
            o_proj_output_frac_bits: commit_bool_bits(&witnesses.o_proj_output_frac_bits, setup),
            pv_matmul_output_frac_bits: commit_bool_bits(
                &witnesses.pv_matmul_output_frac_bits,
                setup,
            ),
            softmax_floor_frac_bits: commit_bool_bits(&witnesses.softmax_floor_frac_bits, setup),
            softmax_output_frac_bits: commit_bool_bits(&witnesses.softmax_output_frac_bits, setup),
            softmax_exp_frac_bits: commit_bool_bits(&witnesses.softmax_exp_frac_bits, setup),
            qk_score_dot_output_frac_bits: commit_bool_bits(
                &witnesses.qk_score_dot_output_frac_bits,
                setup,
            ),
            qk_score_output_frac_bits: commit_bool_bits(
                &witnesses.qk_score_output_frac_bits,
                setup,
            ),
            q_rope_output_frac_bits: commit_bool_bits(&witnesses.q_rope_output_frac_bits, setup),
            k_rope_output_frac_bits: commit_bool_bits(&witnesses.k_rope_output_frac_bits, setup),
            q_norm_norm_frac_bits: commit_bool_bits(&witnesses.q_norm_norm_frac_bits, setup),
            q_norm_output_frac_bits: commit_bool_bits(&witnesses.q_norm_output_frac_bits, setup),
            k_norm_norm_frac_bits: commit_bool_bits(&witnesses.k_norm_norm_frac_bits, setup),
            k_norm_output_frac_bits: commit_bool_bits(&witnesses.k_norm_output_frac_bits, setup),
            q_proj_output_frac_bits: commit_bool_bits(&witnesses.q_proj_output_frac_bits, setup),
            k_proj_output_frac_bits: commit_bool_bits(&witnesses.k_proj_output_frac_bits, setup),
            v_proj_output_frac_bits: commit_bool_bits(&witnesses.v_proj_output_frac_bits, setup),
            rms_norm_atten_norm_frac_bits: commit_bool_bits(
                &witnesses.rms_norm_atten_norm_frac_bits,
                setup,
            ),
            rms_norm_atten_output_frac_bits: commit_bool_bits(
                &witnesses.rms_norm_atten_output_frac_bits,
                setup,
            ),
        };
        LayerCommitments {
            hidden_out: commit_i32_chunks::<Pcs>(&witnesses.hidden_out, 1 << 4, setup).unwrap(),
            hidden_in_a: commit_i32_chunks::<Pcs>(&witnesses.hidden_in_a, 1 << 4, setup).unwrap(),
            hidden_in_b: commit_i32_chunks::<Pcs>(&witnesses.hidden_in_b, 1 << 4, setup).unwrap(),
            silu_lookup_ra: commit_ra_chunks(
                &selected_entries_as_u16(&witnesses.silu_lookup_ra, 2),
                1 << 4,
                2,
                setup,
            )
            .unwrap(),
            softmax_lookup_ra: commit_ra_chunks(
                &selected_entries_as_u16(&witnesses.softmax_lookup_ra, 4),
                1 << 4,
                4,
                setup,
            )
            .unwrap(),
            bits: bit_commitments,
        }
    }

    type Pcs = HyperKZG<Bn254>;

    fn commit_bool_bits(
        bits: &BitOpeningWitnesses,
        setup: &<Pcs as CommitmentScheme>::ProverSetup,
    ) -> [ChunkedCommitments<HyperKZGCommitment<Bn254>>; 8] {
        let u8_bits =
            std::array::from_fn(|bit| bits[bit].iter().copied().map(u8::from).collect::<Vec<_>>());
        commit_bit_tables::<Pcs>(&u8_bits, 1 << 4, setup).unwrap()
    }

    fn selected_entries_as_u16(table: &[u8], tensor_len: usize) -> Vec<u16> {
        selected_entries_from_dense_ra(table, tensor_len)
            .unwrap()
            .into_iter()
            .map(|entry| entry as u16)
            .collect()
    }
}
