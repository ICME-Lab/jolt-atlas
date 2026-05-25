use joltworks::{field::JoltField, transcripts::Transcript, utils::errors::ProofVerifyError};

use crate::{
    claim::{Claim, Shape},
    error::Result,
    ops::{
        hadamard_mul::{prove_hadamard_round, verify_hadamard_round},
        matadd::{prove_matadd, verify_matadd},
        matmul::{prove_matmul_round, verify_matmul_round},
        pv_matmul::{prove_pv_matmul_round, verify_pv_matmul_round},
        qk_score::{prove_qk_score_round, verify_qk_score_round},
        rms_norm::{prove_rmsnorm_round, verify_rmsnorm_round},
        rope::{prove_rope_round, verify_rope_round},
        silu::{prove_silu_round, verify_silu_round},
        softmax::{prove_softmax_round, verify_softmax_round},
    },
    proof::ProveResult,
};

use super::{
    tensors::LayerTensorIds,
    types::{LayerClaims, LayerIopProof, LayerShape, LayerWeights},
    witness::LayerWitness,
};

// Layer IOP only. No trace loading, no commitments, no PCS accumulator.
// It consumes output claims and walks the Qwen block backward to input claims.

pub(crate) fn prove_layer_iop<F, T>(
    hidden_out: Claim<F>,
    witness: &LayerWitness,
    weights: &LayerWeights,
    shape: &LayerShape,
    tensors: &LayerTensorIds,
    transcript: &mut T,
) -> Result<ProveResult<LayerClaims<F>, LayerIopProof<F, T>>>
where
    F: JoltField,
    T: Transcript,
{
    macro_rules! timed {
        ($name:literal, $expr:expr) => {{ $expr }};
    }

    // hidden_out = residual_add_attn + down_proj
    let (residual_add_mlp_proof, residual_add_attn_a, down_proj) = timed!(
        "residual_add_mlp",
        prove_matadd(
            vec![hidden_out],
            &witness.residual_add_attn_a,
            &witness.down_proj,
            &tensors.residual_add_mlp_params(shape),
            transcript,
        )
    )?;

    // down_proj = silu_up @ W_down
    let (down_proj_proof, silu_up, down_proj_round_ra) = timed!(
        "down_proj",
        prove_matmul_round(
            down_proj,
            &witness.down_proj_witness(),
            &weights.down_proj,
            &tensors.down_proj_params(shape),
            transcript,
        )
    )?;

    // silu_up = silu * up_proj
    let (silu_up_proof, silu, up_proj, silu_up_round_ra) = timed!(
        "silu_up",
        prove_hadamard_round(
            silu_up,
            &witness.silu_up_witness(),
            &tensors.silu_up_params(shape),
            transcript,
        )
    )?;

    // silu = SiLU(gate_proj, frac_bits, ra)
    let (silu_proof, gate_proj, silu_gate_round_ra, silu_ra, silu_out_round_ra) = timed!(
        "silu",
        prove_silu_round(
            silu,
            &witness.silu_witness(),
            &tensors.silu_params(shape),
            transcript,
        )
    )?;

    // gate_proj = rms_norm_mlp @ W_gate
    let (gate_proj_proof, rms_norm_mlp_a, gate_proj_round_ra) = timed!(
        "gate_proj",
        prove_matmul_round(
            gate_proj,
            &witness.gate_proj_witness(),
            &weights.gate_proj,
            &tensors.gate_proj_params(shape),
            transcript,
        )
    )?;

    // up_proj = rms_norm_mlp @ W_up
    let (up_proj_proof, rms_norm_mlp_b, up_proj_round_ra) = timed!(
        "up_proj",
        prove_matmul_round(
            up_proj,
            &witness.up_proj_witness(),
            &weights.up_proj,
            &tensors.up_proj_params(shape),
            transcript,
        )
    )?;

    // rms_norm_mlp = RMSNorm(residual_add_attn)
    let (rms_norm_mlp_proof, residual_add_attn_b, rms_norm_mlp_round_ra) = timed!(
        "rms_norm_mlp",
        prove_rmsnorm_round(
            vec![rms_norm_mlp_a, rms_norm_mlp_b],
            &witness.rms_norm_mlp_witness(),
            &weights.rms_norm_mlp,
            &tensors.rms_norm_mlp_params(shape),
            transcript,
        )
    )?;

    // residual_add_attn = hidden_in_a + o_proj
    let (residual_add_attn_proof, hidden_in_a, o_proj) = timed!(
        "residual_add_attn",
        prove_matadd(
            vec![residual_add_attn_a, residual_add_attn_b],
            &witness.hidden_in,
            &witness.o_proj,
            &tensors.residual_add_attn_params(shape),
            transcript,
        )
    )?;

    // o_proj = context @ W_o
    let (o_proj_proof, context, o_proj_round_ra) = timed!(
        "o_proj",
        prove_matmul_round(
            o_proj,
            &witness.o_proj_witness(),
            &weights.o_proj,
            &tensors.o_proj_params(shape),
            transcript,
        )
    )?;
    let context = reshape_claim(
        context,
        Shape::new(vec![shape.seq, shape.q_heads, shape.head_dim]),
    );

    // context = softmax @ v_proj
    let (pv_matmul_proof, softmax, v_proj, context_round_ra) = timed!(
        "pv_matmul",
        prove_pv_matmul_round(
            context,
            &witness.pv_matmul_witness(),
            &tensors.pv_matmul_params(shape),
            transcript,
        )
    )?;
    let v_proj = reshape_claim(
        v_proj,
        Shape::new(vec![shape.seq, shape.kv_heads * shape.head_dim]),
    );

    // softmax = softmax(qk_score)
    let (
        softmax_proof,
        qk_score,
        softmax_round_ra,
        softmax_floor_round_ra,
        softmax_input_remainder_ra,
        softmax_ra,
    ) = timed!(
        "softmax",
        prove_softmax_round(
            vec![softmax],
            &witness.softmax_witness(),
            &tensors.softmax_params(shape),
            transcript,
        )
    )?;

    // qk_score = round(round(q_rope @ k_rope) * inv_sqrt(head_dim))
    let (qk_score_proof, q_rope, k_rope, qk_score_round_ra, qk_score_dot_round_ra) = timed!(
        "qk_score",
        prove_qk_score_round(
            qk_score,
            &witness.qk_score_witness(),
            &tensors.qk_score_params(shape),
            transcript,
        )
    )?;

    // q_rope = RoPE(q_norm)
    let (q_rope_proof, q_norm_a, q_norm_b, q_rope_round_ra) = timed!(
        "q_rope",
        prove_rope_round(
            q_rope,
            &witness.q_rope_witness(),
            &weights.rope_cos,
            &weights.rope_sin,
            &tensors.q_rope_params(shape),
            transcript,
        )
    )?;

    // k_rope = RoPE(k_norm)
    let (k_rope_proof, k_norm_a, k_norm_b, k_rope_round_ra) = timed!(
        "k_rope",
        prove_rope_round(
            k_rope,
            &witness.k_rope_witness(),
            &weights.rope_cos,
            &weights.rope_sin,
            &tensors.k_rope_params(shape),
            transcript,
        )
    )?;

    // q_norm = RMSNorm(q_proj)
    let (q_norm_proof, q_proj, q_norm_round_ra) = timed!(
        "q_norm",
        prove_rmsnorm_round(
            vec![q_norm_a, q_norm_b],
            &witness.q_norm_witness(),
            &weights.q_norm,
            &tensors.q_norm_params(shape),
            transcript,
        )
    )?;
    let q_proj = reshape_claim(q_proj, Shape::new(vec![shape.seq, shape.attention_width()]));

    // k_norm = RMSNorm(k_proj)
    let (k_norm_proof, k_proj, k_norm_round_ra) = timed!(
        "k_norm",
        prove_rmsnorm_round(
            vec![k_norm_a, k_norm_b],
            &witness.k_norm_witness(),
            &weights.k_norm,
            &tensors.k_norm_params(shape),
            transcript,
        )
    )?;
    let k_proj = reshape_claim(
        k_proj,
        Shape::new(vec![shape.seq, shape.kv_heads * shape.head_dim]),
    );

    // q_proj = rms_norm_atten @ W_q
    let (q_proj_proof, rms_norm_atten_a, q_proj_round_ra) = timed!(
        "q_proj",
        prove_matmul_round(
            q_proj,
            &witness.q_proj_witness(),
            &weights.q_proj,
            &tensors.q_proj_params(shape),
            transcript,
        )
    )?;

    // k_proj = rms_norm_atten @ W_k
    let (k_proj_proof, rms_norm_atten_b, k_proj_round_ra) = timed!(
        "k_proj",
        prove_matmul_round(
            k_proj,
            &witness.k_proj_witness(),
            &weights.k_proj,
            &tensors.k_proj_params(shape),
            transcript,
        )
    )?;

    // v_proj = rms_norm_atten @ W_v
    let (v_proj_proof, rms_norm_atten_c, v_proj_round_ra) = timed!(
        "v_proj",
        prove_matmul_round(
            v_proj,
            &witness.v_proj_witness(),
            &weights.v_proj,
            &tensors.v_proj_params(shape),
            transcript,
        )
    )?;

    // rms_norm_atten = RMSNorm(hidden_in)
    let (rms_norm_atten_proof, hidden_in_b, rms_norm_atten_round_ra) = timed!(
        "rms_norm_atten",
        prove_rmsnorm_round(
            vec![rms_norm_atten_a, rms_norm_atten_b, rms_norm_atten_c],
            &witness.rms_norm_atten_witness(),
            &weights.rms_norm_atten,
            &tensors.rms_norm_atten_params(shape),
            transcript,
        )
    )?;

    Ok(ProveResult::new(
        LayerClaims {
            hidden_in_a,
            hidden_in_b,
            context_round_ra,
            o_proj_round_ra,
            softmax_round_ra,
            softmax_floor_round_ra,
            softmax_input_remainder_ra,
            softmax_ra,
            qk_score_round_ra,
            qk_score_dot_round_ra,
            q_rope_round_ra,
            k_rope_round_ra,
            q_norm_round_ra,
            k_norm_round_ra,
            q_proj_round_ra,
            k_proj_round_ra,
            v_proj_round_ra,
            rms_norm_atten_round_ra,
            rms_norm_mlp_round_ra,
            gate_proj_round_ra,
            silu_gate_round_ra,
            silu_ra,
            silu_out_round_ra,
            silu_up_round_ra,
            up_proj_round_ra,
            down_proj_round_ra,
        },
        LayerIopProof {
            residual_add_mlp_proof,
            down_proj_proof,
            silu_up_proof,
            silu_proof,
            gate_proj_proof,
            up_proj_proof,
            rms_norm_mlp_proof,
            residual_add_attn_proof,
            o_proj_proof,
            pv_matmul_proof,
            softmax_proof,
            qk_score_proof,
            q_rope_proof,
            k_rope_proof,
            q_norm_proof,
            k_norm_proof,
            q_proj_proof,
            k_proj_proof,
            v_proj_proof,
            rms_norm_atten_proof,
        },
    ))
}
pub(crate) fn verify_layer_iop<F, T>(
    hidden_out: Claim<F>,
    proof: &LayerIopProof<F, T>,
    weights: &LayerWeights,
    shape: &LayerShape,
    tensors: &LayerTensorIds,
    transcript: &mut T,
) -> std::result::Result<LayerClaims<F>, ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    macro_rules! timed {
        ($name:literal, $expr:expr) => {{ $expr }};
    }

    // hidden_out = residual_add_attn + down_proj
    let (residual_add_attn_a, down_proj) = timed!(
        "residual_add_mlp",
        verify_matadd(
            vec![hidden_out],
            &proof.residual_add_mlp_proof,
            &tensors.residual_add_mlp_params(shape),
            transcript,
        )
    )?;

    // down_proj = silu_up @ W_down
    let (silu_up, down_proj_round_ra) = timed!(
        "down_proj",
        verify_matmul_round(
            down_proj,
            &proof.down_proj_proof,
            &weights.down_proj,
            &tensors.down_proj_params(shape),
            transcript,
        )
    )?;

    // silu_up = silu * up_proj
    let (silu, up_proj, silu_up_round_ra) = timed!(
        "silu_up",
        verify_hadamard_round(
            silu_up,
            &proof.silu_up_proof,
            &tensors.silu_up_params(shape),
            transcript,
        )
    )?;

    // silu = SiLU(gate_proj, frac_bits, ra)
    let (gate_proj, silu_gate_round_ra, silu_ra, silu_out_round_ra) = timed!(
        "silu",
        verify_silu_round(
            silu,
            &proof.silu_proof,
            &tensors.silu_params(shape),
            transcript,
        )
    )?;

    // gate_proj = rms_norm_mlp @ W_gate
    let (rms_norm_mlp_a, gate_proj_round_ra) = timed!(
        "gate_proj",
        verify_matmul_round(
            gate_proj,
            &proof.gate_proj_proof,
            &weights.gate_proj,
            &tensors.gate_proj_params(shape),
            transcript,
        )
    )?;

    // up_proj = rms_norm_mlp @ W_up
    let (rms_norm_mlp_b, up_proj_round_ra) = timed!(
        "up_proj",
        verify_matmul_round(
            up_proj,
            &proof.up_proj_proof,
            &weights.up_proj,
            &tensors.up_proj_params(shape),
            transcript,
        )
    )?;

    // rms_norm_mlp = RMSNorm(residual_add_attn)
    let (residual_add_attn_b, rms_norm_mlp_round_ra) = timed!(
        "rms_norm_mlp",
        verify_rmsnorm_round(
            vec![rms_norm_mlp_a, rms_norm_mlp_b],
            &proof.rms_norm_mlp_proof,
            &weights.rms_norm_mlp,
            &tensors.rms_norm_mlp_params(shape),
            transcript,
        )
    )?;

    // residual_add_attn = hidden_in_a + o_proj
    let (hidden_in_a, o_proj) = timed!(
        "residual_add_attn",
        verify_matadd(
            vec![residual_add_attn_a, residual_add_attn_b],
            &proof.residual_add_attn_proof,
            &tensors.residual_add_attn_params(shape),
            transcript,
        )
    )?;

    // o_proj = context @ W_o
    let (context, o_proj_round_ra) = timed!(
        "o_proj",
        verify_matmul_round(
            o_proj,
            &proof.o_proj_proof,
            &weights.o_proj,
            &tensors.o_proj_params(shape),
            transcript,
        )
    )?;
    let context = reshape_claim(
        context,
        Shape::new(vec![shape.seq, shape.q_heads, shape.head_dim]),
    );

    // context = softmax @ v_proj
    let (softmax, v_proj, context_round_ra) = timed!(
        "pv_matmul",
        verify_pv_matmul_round(
            context,
            &proof.pv_matmul_proof,
            &tensors.pv_matmul_params(shape),
            transcript,
        )
    )?;
    let v_proj = reshape_claim(
        v_proj,
        Shape::new(vec![shape.seq, shape.kv_heads * shape.head_dim]),
    );

    // softmax = softmax(qk_score)
    let (
        qk_score,
        softmax_round_ra,
        softmax_floor_round_ra,
        softmax_input_remainder_ra,
        softmax_ra,
    ) = timed!(
        "softmax",
        verify_softmax_round(
            vec![softmax],
            &proof.softmax_proof,
            &tensors.softmax_params(shape),
            transcript,
        )
    )?;

    // qk_score = round(round(q_rope @ k_rope) * inv_sqrt(head_dim))
    let (q_rope, k_rope, qk_score_round_ra, qk_score_dot_round_ra) = timed!(
        "qk_score",
        verify_qk_score_round(
            qk_score,
            &proof.qk_score_proof,
            &tensors.qk_score_params(shape),
            transcript,
        )
    )?;

    // q_rope = RoPE(q_norm)
    let (q_norm_a, q_norm_b, q_rope_round_ra) = timed!(
        "q_rope",
        verify_rope_round(
            q_rope,
            &proof.q_rope_proof,
            &weights.rope_cos,
            &weights.rope_sin,
            &tensors.q_rope_params(shape),
            transcript,
        )
    )?;

    // k_rope = RoPE(k_norm)
    let (k_norm_a, k_norm_b, k_rope_round_ra) = timed!(
        "k_rope",
        verify_rope_round(
            k_rope,
            &proof.k_rope_proof,
            &weights.rope_cos,
            &weights.rope_sin,
            &tensors.k_rope_params(shape),
            transcript,
        )
    )?;

    // q_norm = RMSNorm(q_proj)
    let (q_proj, q_norm_round_ra) = timed!(
        "q_norm",
        verify_rmsnorm_round(
            vec![q_norm_a, q_norm_b],
            &proof.q_norm_proof,
            &weights.q_norm,
            &tensors.q_norm_params(shape),
            transcript,
        )
    )?;
    let q_proj = reshape_claim(q_proj, Shape::new(vec![shape.seq, shape.attention_width()]));

    // k_norm = RMSNorm(k_proj)
    let (k_proj, k_norm_round_ra) = timed!(
        "k_norm",
        verify_rmsnorm_round(
            vec![k_norm_a, k_norm_b],
            &proof.k_norm_proof,
            &weights.k_norm,
            &tensors.k_norm_params(shape),
            transcript,
        )
    )?;
    let k_proj = reshape_claim(
        k_proj,
        Shape::new(vec![shape.seq, shape.kv_heads * shape.head_dim]),
    );

    // q_proj = rms_norm_atten @ W_q
    let (rms_norm_atten_a, q_proj_round_ra) = timed!(
        "q_proj",
        verify_matmul_round(
            q_proj,
            &proof.q_proj_proof,
            &weights.q_proj,
            &tensors.q_proj_params(shape),
            transcript,
        )
    )?;

    // k_proj = rms_norm_atten @ W_k
    let (rms_norm_atten_b, k_proj_round_ra) = timed!(
        "k_proj",
        verify_matmul_round(
            k_proj,
            &proof.k_proj_proof,
            &weights.k_proj,
            &tensors.k_proj_params(shape),
            transcript,
        )
    )?;

    // v_proj = rms_norm_atten @ W_v
    let (rms_norm_atten_c, v_proj_round_ra) = timed!(
        "v_proj",
        verify_matmul_round(
            v_proj,
            &proof.v_proj_proof,
            &weights.v_proj,
            &tensors.v_proj_params(shape),
            transcript,
        )
    )?;

    // rms_norm_atten = RMSNorm(hidden_in)
    let (hidden_in_b, rms_norm_atten_round_ra) = timed!(
        "rms_norm_atten",
        verify_rmsnorm_round(
            vec![rms_norm_atten_a, rms_norm_atten_b, rms_norm_atten_c],
            &proof.rms_norm_atten_proof,
            &weights.rms_norm_atten,
            &tensors.rms_norm_atten_params(shape),
            transcript,
        )
    )?;

    Ok(LayerClaims {
        hidden_in_a,
        hidden_in_b,
        context_round_ra,
        o_proj_round_ra,
        softmax_round_ra,
        softmax_floor_round_ra,
        softmax_input_remainder_ra,
        softmax_ra,
        qk_score_round_ra,
        qk_score_dot_round_ra,
        q_rope_round_ra,
        k_rope_round_ra,
        q_norm_round_ra,
        k_norm_round_ra,
        q_proj_round_ra,
        k_proj_round_ra,
        v_proj_round_ra,
        rms_norm_atten_round_ra,
        rms_norm_mlp_round_ra,
        gate_proj_round_ra,
        silu_gate_round_ra,
        silu_ra,
        silu_out_round_ra,
        silu_up_round_ra,
        up_proj_round_ra,
        down_proj_round_ra,
    })
}

// === Shape bridge ==========================================================

fn reshape_claim<F>(mut claim: Claim<F>, logical_shape: Shape) -> Claim<F> {
    // Projection ops use flattened `[seq, heads * head_dim]` tensors while
    // attention ops use `[seq, heads, head_dim]`.  For Qwen these split
    // dimensions are powers of two, so the row-major MLE point is the same
    // concatenation of bits and this shape bridge needs no sumcheck.
    claim.logical_shape = logical_shape.clone();
    claim.domain_shape = logical_shape.padded_power_of_two();
    claim
}
