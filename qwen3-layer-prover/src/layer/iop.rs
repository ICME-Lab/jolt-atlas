use joltworks::{field::JoltField, transcripts::Transcript};

use crate::{
    claim::Claim,
    error::Result,
    ops::{
        hadamard_mul::prove_hadamard_round, matadd::prove_matadd, matmul::prove_matmul_round,
        pv_matmul::prove_pv_matmul_round, qk_score::prove_qk_score_round,
        rms_norm::prove_rmsnorm_round, rope::prove_rope_round, silu::prove_silu_round,
        softmax::prove_softmax_round,
    },
    proof::ProveResult,
};

use super::{
    polys::LayerPolys,
    tensors::LayerTensorIds,
    types::{LayerClaims, LayerIopProof, LayerShape},
};

/// Pure layer IOP.
///
/// Every op returns a tuple. This file deliberately names every tuple element at
/// the call site so the claim flow stays visible and no hidden aggregate claim
/// bucket can swallow a missing RA claim.
pub(crate) fn prove_layer_iop<F, T, C>(
    hidden_out: Claim<F, C>,
    polys: LayerPolys<F, C>,
    shape: &LayerShape,
    tensors: &LayerTensorIds,
    transcript: &mut T,
) -> Result<ProveResult<LayerClaims<F, C>, LayerIopProof<F, T>>>
where
    F: JoltField,
    T: Transcript,
    C: Clone,
{
    let hidden_in_for_residual = polys.hidden_in.clone();
    let rope_cos_for_q = polys.rope_cos.clone();
    let rope_sin_for_q = polys.rope_sin.clone();

    // residual_add_mlp = residual_add_attn + down_proj
    let (residual_add_mlp_proof, residual_add_attn_a, down_proj) = prove_matadd(
        vec![hidden_out],
        polys.residual_add_attn_a,
        polys.down_proj,
        &tensors.residual_add_mlp_params(shape),
        transcript,
    )?;

    // down_proj = silu_up @ W_down
    let (down_proj_proof, silu_up, w_down_proj, down_proj_round_ra) = prove_matmul_round(
        down_proj,
        polys.silu_up,
        polys.w_down_proj,
        polys.down_proj_round_ra,
        &tensors.down_proj_params(shape),
        transcript,
    )?;

    // silu_up = silu * up_proj
    let (silu_up_proof, silu, up_proj, silu_up_round_ra) = prove_hadamard_round(
        silu_up,
        polys.silu,
        polys.up_proj,
        polys.silu_up_round_ra,
        &tensors.silu_up_params(shape),
        transcript,
    )?;

    // silu = SiLU(gate_proj)
    let (silu_proof, gate_proj, silu_gate_round_ra, silu_ra, silu_round_ra) = prove_silu_round(
        silu,
        polys.gate_proj,
        polys.silu_gate_round_ra,
        polys.silu_ra,
        polys.silu_round_ra,
        &tensors.silu_params(shape),
        transcript,
    )?;

    // gate_proj = rms_norm_mlp @ W_gate
    let (gate_proj_proof, rms_norm_mlp_a, w_gate_proj, gate_proj_round_ra) = prove_matmul_round(
        gate_proj,
        polys.rms_norm_mlp_a,
        polys.w_gate_proj,
        polys.gate_proj_round_ra,
        &tensors.gate_proj_params(shape),
        transcript,
    )?;

    // up_proj = rms_norm_mlp @ W_up
    let (up_proj_proof, rms_norm_mlp_b, w_up_proj, up_proj_round_ra) = prove_matmul_round(
        up_proj,
        polys.rms_norm_mlp_b,
        polys.w_up_proj,
        polys.up_proj_round_ra,
        &tensors.up_proj_params(shape),
        transcript,
    )?;

    // rms_norm_mlp = RMSNorm(residual_add_attn)
    let (
        rms_norm_mlp_proof,
        residual_add_attn_b,
        w_rms_norm_mlp,
        rms_norm_mlp_norm_round_ra,
        rms_norm_mlp_round_ra,
    ) = prove_rmsnorm_round(
        vec![rms_norm_mlp_a, rms_norm_mlp_b],
        polys.residual_add_attn_b,
        polys.w_rms_norm_mlp,
        polys.rms_norm_mlp_norm_round_ra,
        polys.rms_norm_mlp_round_ra,
        &tensors.rms_norm_mlp_params(shape),
        transcript,
    )?;

    // residual_add_attn = hidden_in + o_proj
    let (residual_add_attn_proof, hidden_in_a, o_proj) = prove_matadd(
        vec![residual_add_attn_a, residual_add_attn_b],
        hidden_in_for_residual,
        polys.o_proj,
        &tensors.residual_add_attn_params(shape),
        transcript,
    )?;

    // o_proj = context @ W_o
    let (o_proj_proof, context, w_o_proj, o_proj_round_ra) = prove_matmul_round(
        o_proj,
        polys.context,
        polys.w_o_proj,
        polys.o_proj_round_ra,
        &tensors.o_proj_params(shape),
        transcript,
    )?;

    // context = softmax @ v_proj
    let (pv_matmul_proof, softmax, v_proj, pv_matmul_round_ra) = prove_pv_matmul_round(
        context,
        polys.softmax,
        polys.v_proj,
        polys.pv_matmul_round_ra,
        &tensors.pv_matmul_params(shape),
        transcript,
    )?;

    // softmax = softmax(qk_score)
    let (
        softmax_proof,
        qk_score,
        softmax_round_ra,
        softmax_floor_round_ra,
        softmax_exp_round_ra,
        softmax_ra,
    ) = prove_softmax_round(
        vec![softmax],
        polys.qk_score,
        polys.softmax_round_ra,
        polys.softmax_floor_round_ra,
        polys.softmax_exp_round_ra,
        polys.softmax_ra,
        &tensors.softmax_params(shape),
        transcript,
    )?;

    // qk_score = q_rope @ k_rope^T
    let (qk_score_proof, q_rope, k_rope, qk_score_round_ra, qk_score_dot_round_ra) =
        prove_qk_score_round(
            qk_score,
            polys.q_rope,
            polys.k_rope,
            polys.qk_score_round_ra,
            polys.qk_score_dot_round_ra,
            &tensors.qk_score_params(shape),
            transcript,
        )?;

    // q_rope = RoPE(q_norm)
    let (q_rope_proof, q_norm_a, q_norm_b, q_rope_round_ra) = prove_rope_round(
        q_rope,
        polys.q_norm,
        rope_cos_for_q,
        rope_sin_for_q,
        polys.q_rope_round_ra,
        &tensors.q_rope_params(shape),
        transcript,
    )?;

    // k_rope = RoPE(k_norm)
    let (k_rope_proof, k_norm_a, k_norm_b, k_rope_round_ra) = prove_rope_round(
        k_rope,
        polys.k_norm,
        polys.rope_cos,
        polys.rope_sin,
        polys.k_rope_round_ra,
        &tensors.k_rope_params(shape),
        transcript,
    )?;

    // q_norm = RMSNorm(q_proj)
    let (q_norm_proof, q_proj, w_q_norm, q_norm_norm_round_ra, q_norm_round_ra) =
        prove_rmsnorm_round(
            vec![q_norm_a, q_norm_b],
            polys.q_proj,
            polys.w_q_norm,
            polys.q_norm_norm_round_ra,
            polys.q_norm_round_ra,
            &tensors.q_norm_params(shape),
            transcript,
        )?;

    // k_norm = RMSNorm(k_proj)
    let (k_norm_proof, k_proj, w_k_norm, k_norm_norm_round_ra, k_norm_round_ra) =
        prove_rmsnorm_round(
            vec![k_norm_a, k_norm_b],
            polys.k_proj,
            polys.w_k_norm,
            polys.k_norm_norm_round_ra,
            polys.k_norm_round_ra,
            &tensors.k_norm_params(shape),
            transcript,
        )?;

    // q_proj = rms_norm_atten @ W_q
    let (q_proj_proof, rms_norm_atten_a, w_q_proj, q_proj_round_ra) = prove_matmul_round(
        q_proj,
        polys.rms_norm_atten_a,
        polys.w_q_proj,
        polys.q_proj_round_ra,
        &tensors.q_proj_params(shape),
        transcript,
    )?;

    // k_proj = rms_norm_atten @ W_k
    let (k_proj_proof, rms_norm_atten_b, w_k_proj, k_proj_round_ra) = prove_matmul_round(
        k_proj,
        polys.rms_norm_atten_b,
        polys.w_k_proj,
        polys.k_proj_round_ra,
        &tensors.k_proj_params(shape),
        transcript,
    )?;

    // v_proj = rms_norm_atten @ W_v
    let (v_proj_proof, rms_norm_atten_c, w_v_proj, v_proj_round_ra) = prove_matmul_round(
        v_proj,
        polys.rms_norm_atten_c,
        polys.w_v_proj,
        polys.v_proj_round_ra,
        &tensors.v_proj_params(shape),
        transcript,
    )?;

    // rms_norm_atten = RMSNorm(hidden_in)
    let (
        rms_norm_atten_proof,
        hidden_in_b,
        w_rms_norm_atten,
        rms_norm_atten_norm_round_ra,
        rms_norm_atten_round_ra,
    ) = prove_rmsnorm_round(
        vec![rms_norm_atten_a, rms_norm_atten_b, rms_norm_atten_c],
        polys.hidden_in,
        polys.w_rms_norm_atten,
        polys.rms_norm_atten_norm_round_ra,
        polys.rms_norm_atten_round_ra,
        &tensors.rms_norm_atten_params(shape),
        transcript,
    )?;

    Ok(ProveResult::new(
        LayerClaims {
            hidden_in_a,
            hidden_in_b,
            direct_eval_claims: vec![
                w_down_proj,
                w_gate_proj,
                w_up_proj,
                w_rms_norm_mlp,
                w_o_proj,
                w_q_norm,
                w_k_norm,
                w_q_proj,
                w_k_proj,
                w_v_proj,
                w_rms_norm_atten,
            ],
            down_proj_round_ra,
            silu_up_round_ra,
            silu_gate_round_ra,
            silu_ra,
            silu_round_ra,
            gate_proj_round_ra,
            up_proj_round_ra,
            rms_norm_mlp_round_ra,
            rms_norm_mlp_norm_round_ra,
            o_proj_round_ra,
            pv_matmul_round_ra,
            softmax_round_ra,
            softmax_floor_round_ra,
            softmax_exp_round_ra,
            softmax_ra,
            qk_score_round_ra,
            qk_score_dot_round_ra,
            q_rope_round_ra,
            k_rope_round_ra,
            q_norm_round_ra,
            q_norm_norm_round_ra,
            k_norm_round_ra,
            k_norm_norm_round_ra,
            q_proj_round_ra,
            k_proj_round_ra,
            v_proj_round_ra,
            rms_norm_atten_round_ra,
            rms_norm_atten_norm_round_ra,
        },
        LayerIopProof {
            residual_add_mlp: residual_add_mlp_proof,
            down_proj: down_proj_proof,
            silu_up: silu_up_proof,
            silu: silu_proof,
            gate_proj: gate_proj_proof,
            up_proj: up_proj_proof,
            rms_norm_mlp: rms_norm_mlp_proof,
            residual_add_attn: residual_add_attn_proof,
            o_proj: o_proj_proof,
            pv_matmul: pv_matmul_proof,
            softmax: softmax_proof,
            qk_score: qk_score_proof,
            q_rope: q_rope_proof,
            k_rope: k_rope_proof,
            q_norm: q_norm_proof,
            k_norm: k_norm_proof,
            q_proj: q_proj_proof,
            k_proj: k_proj_proof,
            v_proj: v_proj_proof,
            rms_norm_atten: rms_norm_atten_proof,
        },
    ))
}
