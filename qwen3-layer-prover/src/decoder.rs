use joltworks::{field::JoltField, transcripts::Transcript, utils::errors::ProofVerifyError};

use crate::{
    claim::Claim,
    error::Result,
    layer::{
        LayerClaims, LayerProof, LayerShape, LayerTensorIds, LayerWeights, LayerWitness,
        prove_layer, verify_layer,
    },
    ops::round::ROUND_FRAC_BITS,
    proof::ProveResult,
};

// Proves the repeated Qwen decoder layers.
// This does not include token embedding, final RMSNorm, lm_head, or sampling.
//
// Qwen3-0.6B has a public, fixed decoder depth.  The layer count must not be
// inferred from the witness.
pub const QWEN3_DECODER_LAYERS: usize = 28;

#[derive(Debug, Clone)]
pub struct DecoderWitness {
    pub layers: Vec<LayerWitness>,
}

#[derive(Debug, Clone)]
pub struct DecoderWeights {
    pub layers: Vec<LayerWeights>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderClaims<F> {
    pub hidden_in_a: Claim<F>,
    pub hidden_in_b: Claim<F>,
    pub layers: Vec<LayerAuxClaims<F>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerAuxClaims<F> {
    pub context_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub o_proj_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub softmax_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub softmax_floor_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub softmax_input_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub softmax_ra: Claim<F>,
    pub qk_score_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub q_rope_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub k_rope_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub q_norm_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub k_norm_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub q_proj_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub k_proj_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub v_proj_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub rms_norm_atten_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub rms_norm_mlp_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub gate_proj_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub silu_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub silu_ra: Claim<F>,
    pub silu_out_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub silu_up_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub up_proj_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
    pub down_proj_frac_bits: [Claim<F>; ROUND_FRAC_BITS],
}

#[derive(Debug, Clone)]
pub struct DecoderProof<F: JoltField, T: Transcript> {
    pub layers: Vec<LayerProof<F, T>>,
}

pub fn prove_decoder<F, T>(
    hidden_out_a: Claim<F>,
    hidden_out_b: Claim<F>,
    witness: &DecoderWitness,
    weights: &DecoderWeights,
    shape: &LayerShape,
    tensors: &[LayerTensorIds],
    transcript: &mut T,
) -> Result<ProveResult<DecoderClaims<F>, DecoderProof<F, T>>>
where
    F: JoltField,
    T: Transcript,
{
    validate_layer_count(witness.layers.len(), weights.layers.len(), tensors.len())?;

    let mut hidden_a = hidden_out_a;
    let mut hidden_b = hidden_out_b;
    let mut layer_claims = Vec::with_capacity(QWEN3_DECODER_LAYERS);
    let mut layer_proofs = Vec::with_capacity(QWEN3_DECODER_LAYERS);

    for idx in (0..QWEN3_DECODER_LAYERS).rev() {
        let result = prove_layer(
            hidden_a,
            hidden_b,
            &witness.layers[idx],
            &weights.layers[idx],
            shape,
            &tensors[idx],
            transcript,
        )?;
        let (next_hidden_a, next_hidden_b, aux_claims) = split_layer_claims(result.claims);
        hidden_a = next_hidden_a;
        hidden_b = next_hidden_b;
        layer_claims.push(aux_claims);
        layer_proofs.push(result.proof);
    }

    layer_claims.reverse();
    layer_proofs.reverse();

    Ok(ProveResult::new(
        DecoderClaims {
            hidden_in_a: hidden_a,
            hidden_in_b: hidden_b,
            layers: layer_claims,
        },
        DecoderProof {
            layers: layer_proofs,
        },
    ))
}

pub fn verify_decoder<F, T>(
    hidden_out_a: Claim<F>,
    hidden_out_b: Claim<F>,
    proof: &DecoderProof<F, T>,
    weights: &DecoderWeights,
    shape: &LayerShape,
    tensors: &[LayerTensorIds],
    transcript: &mut T,
) -> std::result::Result<DecoderClaims<F>, ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    verify_layer_count(proof.layers.len(), weights.layers.len(), tensors.len())?;

    let mut hidden_a = hidden_out_a;
    let mut hidden_b = hidden_out_b;
    let mut layer_claims = Vec::with_capacity(QWEN3_DECODER_LAYERS);

    for idx in (0..QWEN3_DECODER_LAYERS).rev() {
        let claims = verify_layer(
            hidden_a,
            hidden_b,
            &proof.layers[idx],
            &weights.layers[idx],
            shape,
            &tensors[idx],
            transcript,
        )?;
        let (next_hidden_a, next_hidden_b, aux_claims) = split_layer_claims(claims);
        hidden_a = next_hidden_a;
        hidden_b = next_hidden_b;
        layer_claims.push(aux_claims);
    }

    layer_claims.reverse();

    Ok(DecoderClaims {
        hidden_in_a: hidden_a,
        hidden_in_b: hidden_b,
        layers: layer_claims,
    })
}

fn validate_layer_count(
    witness_layers: usize,
    weight_layers: usize,
    tensor_layers: usize,
) -> Result<()> {
    if witness_layers != QWEN3_DECODER_LAYERS {
        return Err(crate::ProverError::InvalidClaimCount {
            name: "decoder witness layers",
            expected: QWEN3_DECODER_LAYERS,
            actual: witness_layers,
        });
    }
    if weight_layers != QWEN3_DECODER_LAYERS {
        return Err(crate::ProverError::InvalidClaimCount {
            name: "decoder weight layers",
            expected: QWEN3_DECODER_LAYERS,
            actual: weight_layers,
        });
    }
    if tensor_layers != QWEN3_DECODER_LAYERS {
        return Err(crate::ProverError::InvalidClaimCount {
            name: "decoder tensor layers",
            expected: QWEN3_DECODER_LAYERS,
            actual: tensor_layers,
        });
    }
    Ok(())
}

fn verify_layer_count(
    proof_layers: usize,
    weight_layers: usize,
    tensor_layers: usize,
) -> std::result::Result<(), ProofVerifyError> {
    if proof_layers != QWEN3_DECODER_LAYERS {
        return Err(ProofVerifyError::InvalidInputLength(
            QWEN3_DECODER_LAYERS,
            proof_layers,
        ));
    }
    if weight_layers != QWEN3_DECODER_LAYERS {
        return Err(ProofVerifyError::InvalidInputLength(
            QWEN3_DECODER_LAYERS,
            weight_layers,
        ));
    }
    if tensor_layers != QWEN3_DECODER_LAYERS {
        return Err(ProofVerifyError::InvalidInputLength(
            QWEN3_DECODER_LAYERS,
            tensor_layers,
        ));
    }
    Ok(())
}

fn split_layer_claims<F>(claims: LayerClaims<F>) -> (Claim<F>, Claim<F>, LayerAuxClaims<F>) {
    (
        claims.hidden_in_a,
        claims.hidden_in_b,
        LayerAuxClaims {
            context_frac_bits: claims.context_frac_bits,
            o_proj_frac_bits: claims.o_proj_frac_bits,
            softmax_frac_bits: claims.softmax_frac_bits,
            softmax_floor_frac_bits: claims.softmax_floor_frac_bits,
            softmax_input_frac_bits: claims.softmax_input_frac_bits,
            softmax_ra: claims.softmax_ra,
            qk_score_frac_bits: claims.qk_score_frac_bits,
            q_rope_frac_bits: claims.q_rope_frac_bits,
            k_rope_frac_bits: claims.k_rope_frac_bits,
            q_norm_frac_bits: claims.q_norm_frac_bits,
            k_norm_frac_bits: claims.k_norm_frac_bits,
            q_proj_frac_bits: claims.q_proj_frac_bits,
            k_proj_frac_bits: claims.k_proj_frac_bits,
            v_proj_frac_bits: claims.v_proj_frac_bits,
            rms_norm_atten_frac_bits: claims.rms_norm_atten_frac_bits,
            rms_norm_mlp_frac_bits: claims.rms_norm_mlp_frac_bits,
            gate_proj_frac_bits: claims.gate_proj_frac_bits,
            silu_frac_bits: claims.silu_frac_bits,
            silu_ra: claims.silu_ra,
            silu_out_frac_bits: claims.silu_out_frac_bits,
            silu_up_frac_bits: claims.silu_up_frac_bits,
            up_proj_frac_bits: claims.up_proj_frac_bits,
            down_proj_frac_bits: claims.down_proj_frac_bits,
        },
    )
}
