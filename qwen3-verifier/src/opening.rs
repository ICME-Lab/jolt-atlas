use ark_bn254::{Bn254, Fr, G1Affine};
use ark_ff::{One, Zero};
use itertools::Itertools;
use joltworks::{
    field::JoltField,
    poly::commitment::{
        commitment_scheme::CommitmentScheme,
        hyperkzg::{HyperKZG, HyperKZGCommitment, HyperKZGProof, HyperKZGVerifierKey},
    },
    transcripts::Transcript,
};
use qwen3_common::{
    BitOpeningClaims, ChunkedCommitments, ChunkedLayerPcsOpeningProof, EvalClaim, FRAC_BITS,
    LAYER_OPENING_CHUNK_VARS, LayerCommitments, LayerOpeningClaims, LayerShape,
    OpeningReductionProof, RaOpeningClaims, VerifiedOpeningReduction, verify_sumcheck_rounds,
};

pub fn verify_layer_opening_reduction<Tr>(
    claims: &LayerOpeningClaims,
    shape: LayerShape,
    proof: &OpeningReductionProof,
    transcript: &mut Tr,
) -> Option<VerifiedOpeningReduction>
where
    Tr: Transcript,
{
    // Batch all layer opening claims into one sumcheck:
    //
    //   Σ_i gamma_i * v_i
    //     = Σ_x Σ_i gamma_i * eq(z_i, x) * f_i(x)
    //
    // Claims with smaller domains are lifted to the maximum domain by forcing
    // the unused suffix variables to zero.
    validate_layer_opening_claim_domains(claims, shape)?;
    let claims = flatten_layer_claims(claims);
    let max_vars = claims.iter().map(|claim| claim.point.len()).max()?;
    let gammas = draw_opening_reduction_gammas(&claims, transcript);
    let claim = claims
        .iter()
        .zip_eq(&gammas)
        .map(|(claim, gamma)| *gamma * claim.value)
        .sum();
    let sumcheck = verify_sumcheck_rounds(claim, &proof.rounds, max_vars, transcript)?;
    let input_evals = opening_reduction_input_evals(proof)?;
    let public_evals = build_public_opening_reduction_evals(&claims, &sumcheck.point, &gammas)?;
    (sumcheck.final_claim == opening_reduction_relation(&input_evals, &public_evals))
        .then_some(())?;

    Some(VerifiedOpeningReduction {
        sumcheck,
        gamma_powers: gammas,
    })
}

pub fn verify_layer_pcs_opening<Tr>(
    claims: &LayerOpeningClaims,
    shape: LayerShape,
    verified: &VerifiedOpeningReduction,
    commitments: &LayerCommitments<HyperKZGCommitment<Bn254>>,
    proof: &ChunkedLayerPcsOpeningProof,
    setup: &HyperKZGVerifierKey<Bn254>,
    transcript: &mut Tr,
) -> Option<()>
where
    Tr: Transcript,
{
    // Connect the opening-reduction sumcheck to the actual chunk commitments.
    // The prover sends each chunk evaluation.  We first check that those values
    // reconstruct the verified sumcheck final claim, then batch all chunk
    // openings into one HyperKZG verification.
    let chunk_eval_coeffs = build_chunk_eval_coeffs(
        claims,
        shape,
        &verified.sumcheck.point,
        &verified.gamma_powers,
    )?;
    let chunk_opening_point = layer_chunk_opening_point(&verified.sumcheck.challenges)?;
    verify_chunk_evals_match_opening_reduction(
        verified.sumcheck.final_claim,
        &chunk_eval_coeffs,
        &proof.chunk_evals,
    )?;

    append_scalars(transcript, &proof.chunk_evals);
    let pcs_gammas = transcript.challenge_vector::<Fr>(chunk_eval_coeffs.len());
    let chunk_commitments = flatten_layer_chunk_commitments(commitments);
    verify_layer_pcs_opening_proof(
        setup,
        &chunk_commitments,
        &proof.chunk_evals,
        &pcs_gammas,
        &chunk_opening_point,
        &proof.proofs,
        transcript,
    )
}

struct OpeningReductionInputEvals<'a> {
    values_at_reduction_point: &'a [Fr],
}

struct OpeningReductionPublicEvals {
    eq_terms: Vec<Fr>,
    zero_suffix_terms: Vec<Fr>,
    gammas: Vec<Fr>,
}

fn draw_opening_reduction_gammas<Tr>(claims: &[&EvalClaim], transcript: &mut Tr) -> Vec<Fr>
where
    Tr: Transcript,
{
    for claim in claims {
        transcript.append_scalar(&claim.value);
    }
    transcript.challenge_vector::<Fr>(claims.len())
}

fn opening_reduction_input_evals(
    proof: &OpeningReductionProof,
) -> Option<OpeningReductionInputEvals<'_>> {
    Some(OpeningReductionInputEvals {
        values_at_reduction_point: &proof.evals_at_reduction_point,
    })
}

fn build_public_opening_reduction_evals(
    claims: &[&EvalClaim],
    reduction_point: &[Fr],
    gammas: &[Fr],
) -> Option<OpeningReductionPublicEvals> {
    let max_vars = claims.iter().map(|claim| claim.point.len()).max()?;
    (reduction_point.len() == max_vars && gammas.len() == claims.len()).then_some(())?;
    let eq_terms = claims
        .iter()
        .map(|claim| eq_eval(&claim.point, &reduction_point[..claim.point.len()]))
        .collect::<Option<Vec<_>>>()?;
    let zero_suffix_terms = claims
        .iter()
        .map(|claim| zero_suffix_eval(&reduction_point[claim.point.len()..]))
        .collect();
    Some(OpeningReductionPublicEvals {
        eq_terms,
        zero_suffix_terms,
        gammas: gammas.to_vec(),
    })
}

fn opening_reduction_relation(
    input: &OpeningReductionInputEvals<'_>,
    public: &OpeningReductionPublicEvals,
) -> Fr {
    input
        .values_at_reduction_point
        .iter()
        .zip_eq(&public.eq_terms)
        .zip_eq(&public.zero_suffix_terms)
        .zip_eq(&public.gammas)
        .map(|(((value, eq), zero_suffix), gamma)| *gamma * *zero_suffix * *value * *eq)
        .sum()
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
    // Canonical order for layer openings.  This order must match:
    // - LayerOpeningWitnesses in the prover,
    // - flatten_layer_chunk_commitments below,
    // - build_chunk_eval_coeffs for the PCS batching coefficients.
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

fn layer_opening_chunk_layouts(
    claims: &[&EvalClaim],
    shape: LayerShape,
) -> Option<Vec<OpeningChunkLayout>> {
    // Compute chunk layout from claim domains and layer shape only.  The chunk
    // size is fixed by the layer PCS domain, so commitments are not needed here.
    let shape = shape.padded();
    let silu_tensor_len = shape.seq.checked_mul(shape.intermediate)?;
    let softmax_tensor_len = shape
        .q_heads
        .checked_mul(shape.seq)?
        .checked_mul(shape.seq)?;
    let silu_address_space = ra_address_space(claims[3], silu_tensor_len)?;
    let softmax_address_space = ra_address_space(claims[7], softmax_tensor_len)?;

    let mut out = Vec::with_capacity(219);
    for claim in &claims[..3] {
        out.push(dense_chunk_layout(claim)?);
    }
    push_ra_chunk_layouts(&mut out, silu_tensor_len, silu_address_space)?;
    push_ra_chunk_layouts(&mut out, softmax_tensor_len, softmax_address_space)?;
    for claim in &claims[11..] {
        out.push(dense_chunk_layout(claim)?);
    }
    Some(out)
}

fn flatten_layer_chunk_commitments(
    commitments: &LayerCommitments<HyperKZGCommitment<Bn254>>,
) -> Vec<HyperKZGCommitment<Bn254>> {
    // Flatten chunk commitments in the same order as flatten_layer_claims.
    // RA has four logical opening claims over the same one-hot commitment, so
    // each RA chunk commitment is repeated four times.  RoPE has first-half and
    // second-half bit claims over the same full output-bit commitment, so those
    // bit commitments are repeated for the two half claims.
    let mut out = Vec::new();
    push_chunk_commitments(&mut out, &commitments.hidden_out);
    push_chunk_commitments(&mut out, &commitments.hidden_in_a);
    push_chunk_commitments(&mut out, &commitments.hidden_in_b);
    push_ra_chunk_commitments(&mut out, &commitments.silu_lookup_ra);
    push_ra_chunk_commitments(&mut out, &commitments.softmax_lookup_ra);
    push_bit_chunk_commitments(&mut out, &commitments.bits.down_proj_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.silu_up_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.silu_input_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.silu_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.gate_proj_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.up_proj_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.rms_norm_mlp_norm_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.rms_norm_mlp_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.o_proj_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.pv_matmul_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.softmax_floor_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.softmax_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.softmax_exp_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.qk_score_dot_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.qk_score_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.q_rope_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.q_rope_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.k_rope_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.k_rope_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.q_norm_norm_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.q_norm_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.k_norm_norm_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.k_norm_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.q_proj_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.k_proj_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.v_proj_output_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.rms_norm_atten_norm_frac_bits);
    push_bit_chunk_commitments(&mut out, &commitments.bits.rms_norm_atten_output_frac_bits);
    out
}

fn build_chunk_eval_coeffs(
    claims: &LayerOpeningClaims,
    shape: LayerShape,
    reduction_point: &[Fr],
    gamma_powers: &[Fr],
) -> Option<Vec<Fr>> {
    // Expand each full-domain opening claim into coefficients for the concrete
    // chunk evaluations:
    //
    //   coeff_c = gamma_i * eq(z_i, active_point) * zero_suffix
    //                     * eq(chunk_index, chunk_selector_point)
    //
    // Then Σ_c coeff_c * chunk_eval_c must equal the opening-reduction final
    // claim.
    let chunked_claims = chunked_opening_claims(claims, shape, reduction_point, gamma_powers)?;
    let mut chunk_eval_coeffs = Vec::new();
    for claim in chunked_claims {
        for chunk_index in 0..claim.num_chunks {
            let selector_eval = eq_index_eval(chunk_index, claim.chunk_selector_point)?;
            chunk_eval_coeffs.push(claim.coeff * selector_eval);
        }
    }
    Some(chunk_eval_coeffs)
}

fn chunked_opening_claims<'a>(
    claims: &'a LayerOpeningClaims,
    shape: LayerShape,
    reduction_point: &'a [Fr],
    gamma_powers: &'a [Fr],
) -> Option<Vec<ChunkedOpeningClaim<'a>>> {
    // One ChunkedOpeningClaim still corresponds to one original EvalClaim.  It
    // is expanded across its chunks by build_chunk_eval_coeffs.
    validate_layer_opening_claim_domains(claims, shape)?;
    let claims = flatten_layer_claims(claims);
    let layouts = layer_opening_chunk_layouts(&claims, shape)?;
    (claims.len() == layouts.len() && gamma_powers.len() == claims.len()).then_some(())?;

    claims
        .into_iter()
        .zip_eq(layouts)
        .zip_eq(gamma_powers)
        .map(|((claim, layout), gamma)| {
            let claim_vars = claim.point.len();
            (reduction_point.len() >= claim_vars).then_some(())?;
            let active_point = &reduction_point[..claim_vars];
            let selector = chunk_selector(layout, reduction_point, active_point)?;
            Some(ChunkedOpeningClaim {
                coeff: *gamma
                    * zero_suffix_eval(&reduction_point[selector.suffix_start..])
                    * eq_eval(&claim.point, active_point)?,
                chunk_selector_point: selector.chunk_selector_point,
                num_chunks: layout.num_chunks,
            })
        })
        .collect()
}

struct ChunkSelector<'a> {
    chunk_selector_point: &'a [Fr],
    suffix_start: usize,
}

fn chunk_selector<'a>(
    layout: OpeningChunkLayout,
    reduction_point: &'a [Fr],
    active_point: &'a [Fr],
) -> Option<ChunkSelector<'a>> {
    // Split the reduction point into the fixed inner PCS point and the suffix
    // that selects which chunk contributes.  RA chunks are [address, tensor
    // chunk], so only the tensor suffix selects among chunks.
    match layout.kind {
        OpeningChunkKind::Dense => {
            if active_point.len() >= LAYER_OPENING_CHUNK_VARS {
                let (_, chunk_selector_point) = active_point.split_at(LAYER_OPENING_CHUNK_VARS);
                Some(ChunkSelector {
                    chunk_selector_point,
                    suffix_start: active_point.len(),
                })
            } else {
                (reduction_point.len() >= LAYER_OPENING_CHUNK_VARS).then_some(())?;
                Some(ChunkSelector {
                    chunk_selector_point: &active_point[0..0],
                    suffix_start: LAYER_OPENING_CHUNK_VARS,
                })
            }
        }
        OpeningChunkKind::Ra {
            tensor_len,
            address_space,
        } => {
            let tensor_vars = tensor_len.ilog2() as usize;
            let address_vars = address_space.ilog2() as usize;
            let chunk_tensor_vars = LAYER_OPENING_CHUNK_VARS.checked_sub(address_vars)?;
            (active_point.len() >= address_vars + tensor_vars).then_some(())?;
            let (_, tensor_point) = active_point.split_at(address_vars);
            (tensor_point.len() == tensor_vars).then_some(())?;
            if tensor_vars < chunk_tensor_vars {
                let inner_vars = address_vars + chunk_tensor_vars;
                (reduction_point.len() >= inner_vars).then_some(())?;
                return Some(ChunkSelector {
                    chunk_selector_point: &active_point[0..0],
                    suffix_start: inner_vars,
                });
            }
            let (_, chunk_selector_point) = tensor_point.split_at(chunk_tensor_vars);
            Some(ChunkSelector {
                chunk_selector_point,
                suffix_start: active_point.len(),
            })
        }
    }
}

fn layer_chunk_opening_point(
    reduction_challenges: &[<Fr as JoltField>::Challenge],
) -> Option<Vec<<Fr as JoltField>::Challenge>> {
    // All layer chunks use the fixed 2^20 PCS domain.
    (reduction_challenges.len() >= LAYER_OPENING_CHUNK_VARS).then_some(())?;
    Some(reduction_challenges[..LAYER_OPENING_CHUNK_VARS].to_vec())
}

fn verify_layer_pcs_opening_proof<Tr>(
    setup: &HyperKZGVerifierKey<Bn254>,
    commitments: &[HyperKZGCommitment<Bn254>],
    chunk_evals: &[Fr],
    gammas: &[Fr],
    chunk_opening_point: &[<Fr as JoltField>::Challenge],
    proofs: &[HyperKZGProof<Bn254>],
    transcript: &mut Tr,
) -> Option<()>
where
    Tr: Transcript,
{
    // Batch all chunk openings into one PCS check:
    //
    //   C = Σ_j rho_j * C_j
    //   v = Σ_j rho_j * chunk_eval_j
    //
    // and verify that C opens to v at chunk_opening_point.
    (commitments.len() == chunk_evals.len() && gammas.len() == chunk_evals.len()).then_some(())?;
    let joint_commitment = HyperKZG::<Bn254>::combine_commitments(commitments, gammas);
    let pcs_claim = chunk_evals
        .iter()
        .zip_eq(gammas)
        .map(|(value, gamma)| *gamma * *value)
        .sum::<Fr>();
    if joint_commitment.0 == G1Affine::default() {
        (pcs_claim == Fr::zero()).then_some(())?;
        return (proofs.is_empty()).then_some(());
    }
    let [proof] = proofs else {
        return None;
    };
    HyperKZG::verify(
        proof,
        setup,
        transcript,
        &reversed_challenges(chunk_opening_point),
        &pcs_claim,
        &joint_commitment,
    )
    .ok()
}

fn verify_chunk_evals_match_opening_reduction(
    final_claim: Fr,
    chunk_eval_coeffs: &[Fr],
    chunk_evals: &[Fr],
) -> Option<()> {
    // This is the algebraic bridge from the sumcheck final claim to the chunk
    // evaluations that will be checked by PCS.
    (chunk_eval_coeffs.len() == chunk_evals.len()).then_some(())?;
    let claim = chunk_eval_coeffs
        .iter()
        .zip_eq(chunk_evals)
        .map(|(coefficient, value)| *coefficient * *value)
        .sum::<Fr>();
    (final_claim == claim).then_some(())
}

fn dense_chunk_layout(claim: &EvalClaim) -> Option<OpeningChunkLayout> {
    Some(OpeningChunkLayout {
        kind: OpeningChunkKind::Dense,
        num_chunks: num_chunks(claim.point.len())?,
    })
}

fn ra_address_space(claim: &EvalClaim, tensor_len: usize) -> Option<usize> {
    let tensor_vars = tensor_len.ilog2() as usize;
    (claim.point.len() >= tensor_vars).then_some(())?;
    let address_vars = claim.point.len() - tensor_vars;
    let address_space = 1_usize.checked_shl(address_vars as u32)?;
    (address_space > 0 && address_space.is_power_of_two()).then_some(address_space)
}

fn ra_chunk_layout(tensor_len: usize, address_space: usize) -> Option<OpeningChunkLayout> {
    let address_vars = address_space.ilog2() as usize;
    let chunk_tensor_vars = LAYER_OPENING_CHUNK_VARS.checked_sub(address_vars)?;
    let tensor_vars = tensor_len.ilog2() as usize;
    Some(OpeningChunkLayout {
        kind: OpeningChunkKind::Ra {
            tensor_len,
            address_space,
        },
        num_chunks: num_chunks_for_vars(tensor_vars, chunk_tensor_vars)?,
    })
}

fn push_ra_chunk_layouts(
    out: &mut Vec<OpeningChunkLayout>,
    tensor_len: usize,
    address_space: usize,
) -> Option<()> {
    let layout = ra_chunk_layout(tensor_len, address_space)?;
    out.extend(std::iter::repeat_n(layout, 4));
    Some(())
}

fn num_chunks(domain_vars: usize) -> Option<usize> {
    num_chunks_for_vars(domain_vars, LAYER_OPENING_CHUNK_VARS)
}

fn num_chunks_for_vars(domain_vars: usize, chunk_vars: usize) -> Option<usize> {
    if domain_vars <= chunk_vars {
        Some(1)
    } else {
        1_usize.checked_shl((domain_vars - chunk_vars) as u32)
    }
}

fn push_chunk_commitments(
    out: &mut Vec<HyperKZGCommitment<Bn254>>,
    commitments: &ChunkedCommitments<HyperKZGCommitment<Bn254>>,
) {
    out.extend(commitments.commitments.iter().cloned());
}

fn push_ra_chunk_commitments(
    out: &mut Vec<HyperKZGCommitment<Bn254>>,
    commitments: &ChunkedCommitments<HyperKZGCommitment<Bn254>>,
) {
    for _ in 0..4 {
        push_chunk_commitments(out, commitments);
    }
}

fn push_bit_chunk_commitments(
    out: &mut Vec<HyperKZGCommitment<Bn254>>,
    commitments: &[ChunkedCommitments<HyperKZGCommitment<Bn254>>; FRAC_BITS],
) {
    for bit_commitments in commitments {
        push_chunk_commitments(out, bit_commitments);
    }
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

struct ChunkedOpeningClaim<'a> {
    coeff: Fr,
    chunk_selector_point: &'a [Fr],
    num_chunks: usize,
}

#[derive(Debug, Clone, Copy)]
struct OpeningChunkLayout {
    kind: OpeningChunkKind,
    num_chunks: usize,
}

#[derive(Debug, Clone, Copy)]
enum OpeningChunkKind {
    Dense,
    Ra {
        tensor_len: usize,
        address_space: usize,
    },
}
