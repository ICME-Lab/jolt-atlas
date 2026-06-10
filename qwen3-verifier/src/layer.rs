use ark_bn254::Bn254;
use joltworks::{
    poly::commitment::hyperkzg::{HyperKZGCommitment, HyperKZGVerifierKey},
    transcripts::Transcript,
};
use qwen3_common::{
    ChunkedLayerPcsOpeningProof, IopLayerProof, LayerCommitments, LayerOpeningClaims, LayerShape,
    LayerVerifierPublicInput, OpeningReductionProof, draw_hidden_out_claim,
    ops::{
        matmul::{MatMulParams, MatMulVerifierInput},
        pv_matmul::PvMatmulParams,
        qk_score::QkScoreParams,
        rms_norm::RmsNormParams,
        rope::RopeParams,
        silu::SiluParams,
        softmax::SoftmaxParams,
    },
    qwen3_layer_shape,
};

use crate::commitment::append_layer_commitments;
use crate::opening::{verify_layer_opening_reduction, verify_layer_pcs_opening};
use crate::ops::{
    verify_add, verify_add_claims, verify_matmul, verify_mul, verify_pv_matmul, verify_qk_score,
    verify_rms_norm, verify_rope, verify_silu, verify_softmax,
};

pub struct VerifiedIopLayer {
    pub opening_claims: LayerOpeningClaims,
}

struct QwenLayerParams {
    down_proj: MatMulParams,
    silu: SiluParams,
    gate_proj: MatMulParams,
    up_proj: MatMulParams,
    rms_norm_mlp: RmsNormParams,
    o_proj: MatMulParams,
    pv_matmul: PvMatmulParams,
    softmax: SoftmaxParams,
    qk_score: QkScoreParams,
    q_rope: RopeParams,
    k_rope: RopeParams,
    q_norm: RmsNormParams,
    k_norm: RmsNormParams,
    q_proj: MatMulParams,
    k_proj: MatMulParams,
    v_proj: MatMulParams,
    rms_norm_atten: RmsNormParams,
}

impl QwenLayerParams {
    fn new(shape: LayerShape) -> Option<Self> {
        let attention = shape.q_heads.checked_mul(shape.head_dim)?;
        let kv = shape.kv_heads.checked_mul(shape.head_dim)?;
        Some(Self {
            down_proj: MatMulParams::new(shape.seq, shape.hidden, shape.intermediate)?,
            silu: SiluParams::new(shape.seq, shape.intermediate)?,
            gate_proj: MatMulParams::new(shape.seq, shape.intermediate, shape.hidden)?,
            up_proj: MatMulParams::new(shape.seq, shape.intermediate, shape.hidden)?,
            rms_norm_mlp: RmsNormParams::new(shape.seq, shape.hidden)?,
            o_proj: MatMulParams::new(shape.seq, shape.hidden, attention)?,
            pv_matmul: PvMatmulParams::new(
                shape.seq,
                shape.q_heads,
                shape.kv_heads,
                shape.head_dim,
            )?,
            softmax: SoftmaxParams::new(shape.q_heads * shape.seq, shape.seq)?,
            qk_score: QkScoreParams::new(shape.seq, shape.q_heads, shape.kv_heads, shape.head_dim)?,
            q_rope: RopeParams::new(shape.seq, shape.q_heads, shape.head_dim)?,
            k_rope: RopeParams::new(shape.seq, shape.kv_heads, shape.head_dim)?,
            q_norm: RmsNormParams::new(shape.seq * shape.q_heads, shape.head_dim)?,
            k_norm: RmsNormParams::new(shape.seq * shape.kv_heads, shape.head_dim)?,
            q_proj: MatMulParams::new(shape.seq, attention, shape.hidden)?,
            k_proj: MatMulParams::new(shape.seq, kv, shape.hidden)?,
            v_proj: MatMulParams::new(shape.seq, kv, shape.hidden)?,
            rms_norm_atten: RmsNormParams::new(shape.seq, shape.hidden)?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_layer<T>(
    public: LayerVerifierPublicInput,
    commitments: &LayerCommitments<HyperKZGCommitment<Bn254>>,
    iop: IopLayerProof,
    opening: &OpeningReductionProof,
    pcs_opening: &ChunkedLayerPcsOpeningProof,
    setup: &HyperKZGVerifierKey<Bn254>,
    transcript: &mut T,
) -> Option<VerifiedIopLayer>
where
    T: Transcript,
{
    // The verifier transcript starts from the committed layer witnesses.  The
    // IOP then reduces the public output claim into named opening claims, and
    // the two opening phases prove that those claimed evaluations match the
    // committed hidden/RA/bit polynomials.
    append_layer_commitments(transcript, commitments);
    let shape = qwen3_layer_shape(public.seq)?;
    let iop = verify_iop_layer(public, iop, transcript)?;
    let verified_opening =
        verify_layer_opening_reduction(&iop.opening_claims, shape, opening, transcript)?;
    verify_layer_pcs_opening(
        &iop.opening_claims,
        shape,
        &verified_opening,
        commitments,
        pcs_opening,
        setup,
        transcript,
    )?;
    Some(iop)
}

pub fn verify_iop_layer<T>(
    public: LayerVerifierPublicInput,
    proof: IopLayerProof,
    transcript: &mut T,
) -> Option<VerifiedIopLayer>
where
    T: Transcript,
{
    // Verify the layer IOP as a backward claim-reduction chain.  Each op
    // consumes an output EvalClaim and returns the input EvalClaims that must be
    // opened later.  Public weights/tables are evaluated inside each verifier.
    let shape = qwen3_layer_shape(public.seq)?;
    let params = QwenLayerParams::new(shape)?;
    let hidden_out = draw_hidden_out_claim(transcript, proof.hidden_out, shape)?;

    // residual_add_mlp = residual_add_attn + down_proj
    let residual_add_mlp = verify_add(hidden_out.clone(), &proof.residual_add_mlp, transcript)?;

    // down_proj = silu_up @ W_down
    let down_proj = verify_matmul(
        residual_add_mlp.rhs_claim,
        MatMulVerifierInput {
            params: params.down_proj,
            weight: public.down_proj_weight,
        },
        &proof.down_proj,
        transcript,
    )?;

    // silu_up = silu * up_proj
    let silu_up = verify_mul(down_proj.lhs, &proof.silu_up, transcript)?;

    // silu = SiLU(gate_proj)
    let silu = verify_silu(
        silu_up.lhs,
        params.silu,
        public.silu.advice,
        &proof.silu,
        transcript,
    )?;

    // gate_proj = rms_norm_mlp @ W_gate
    let gate_proj = verify_matmul(
        silu.input,
        MatMulVerifierInput {
            params: params.gate_proj,
            weight: public.gate_proj_weight,
        },
        &proof.gate_proj,
        transcript,
    )?;

    // up_proj = rms_norm_mlp @ W_up
    let up_proj = verify_matmul(
        silu_up.rhs,
        MatMulVerifierInput {
            params: params.up_proj,
            weight: public.up_proj_weight,
        },
        &proof.up_proj,
        transcript,
    )?;

    // rms_norm_mlp = RMSNorm(residual_add_attn)
    let rms_norm_mlp = verify_rms_norm(
        vec![gate_proj.lhs, up_proj.lhs],
        params.rms_norm_mlp,
        public.rms_norm_mlp,
        &proof.rms_norm_mlp,
        transcript,
    )?;

    // residual_add_attn = hidden_in + o_proj, with residual fanout absorbed inside add.
    let residual_add_attn = verify_add_claims(
        vec![residual_add_mlp.lhs_claim, rms_norm_mlp.input],
        &proof.residual_add_attn,
        transcript,
    )?;

    // o_proj = context @ W_o
    let o_proj = verify_matmul(
        residual_add_attn.rhs_claim,
        MatMulVerifierInput {
            params: params.o_proj,
            weight: public.o_proj_weight,
        },
        &proof.o_proj,
        transcript,
    )?;

    // context = softmax @ v_proj
    let pv_matmul = verify_pv_matmul(o_proj.lhs, params.pv_matmul, &proof.pv_matmul, transcript)?;

    // softmax = softmax(qk_score)
    let softmax = verify_softmax(
        pv_matmul.p,
        params.softmax,
        public.softmax.advice,
        &proof.softmax,
        transcript,
    )?;

    // qk_score = q_rope @ k_rope^T
    let qk_score = verify_qk_score(softmax.input, params.qk_score, &proof.qk_score, transcript)?;

    // q_rope = RoPE(q_norm)
    let q_rope = verify_rope(
        qk_score.dot.q,
        params.q_rope,
        public.q_rope,
        &proof.q_rope,
        transcript,
    )?;

    // k_rope = RoPE(k_norm)
    let k_rope = verify_rope(
        qk_score.dot.k,
        params.k_rope,
        public.k_rope,
        &proof.k_rope,
        transcript,
    )?;

    // q_norm = RMSNorm(q_proj)
    let q_norm = verify_rms_norm(
        vec![q_rope.input_first_half, q_rope.input_second_half],
        params.q_norm,
        public.q_norm,
        &proof.q_norm,
        transcript,
    )?;

    // k_norm = RMSNorm(k_proj)
    let k_norm = verify_rms_norm(
        vec![k_rope.input_first_half, k_rope.input_second_half],
        params.k_norm,
        public.k_norm,
        &proof.k_norm,
        transcript,
    )?;

    // q_proj = rms_norm_atten @ W_q
    let q_proj = verify_matmul(
        q_norm.input,
        MatMulVerifierInput {
            params: params.q_proj,
            weight: public.q_proj_weight,
        },
        &proof.q_proj,
        transcript,
    )?;

    // k_proj = rms_norm_atten @ W_k
    let k_proj = verify_matmul(
        k_norm.input,
        MatMulVerifierInput {
            params: params.k_proj,
            weight: public.k_proj_weight,
        },
        &proof.k_proj,
        transcript,
    )?;

    // v_proj = rms_norm_atten @ W_v
    let v_proj = verify_matmul(
        pv_matmul.v,
        MatMulVerifierInput {
            params: params.v_proj,
            weight: public.v_proj_weight,
        },
        &proof.v_proj,
        transcript,
    )?;

    // rms_norm_atten = RMSNorm(hidden_in)
    let rms_norm_atten = verify_rms_norm(
        vec![q_proj.lhs, k_proj.lhs, v_proj.lhs],
        params.rms_norm_atten,
        public.rms_norm_atten,
        &proof.rms_norm_atten,
        transcript,
    )?;

    let opening_claims = LayerOpeningClaims {
        hidden_out,
        hidden_in_a: residual_add_attn.lhs_claim,
        hidden_in_b: rms_norm_atten.input,
        silu_lookup_ra: silu.ra,
        softmax_lookup_ra: softmax.ra,
        down_proj_output_frac_bits: down_proj.rounding_bits,
        silu_up_output_frac_bits: silu_up.bits,
        silu_input_frac_bits: silu.input_bits,
        silu_output_frac_bits: silu.output_bits,
        gate_proj_output_frac_bits: gate_proj.rounding_bits,
        up_proj_output_frac_bits: up_proj.rounding_bits,
        rms_norm_mlp_norm_frac_bits: rms_norm_mlp.norm_bits,
        rms_norm_mlp_output_frac_bits: rms_norm_mlp.output_bits,
        o_proj_output_frac_bits: o_proj.rounding_bits,
        pv_matmul_output_frac_bits: pv_matmul.context_remainder_bits,
        softmax_floor_frac_bits: softmax.floor_bits,
        softmax_output_frac_bits: softmax.output_bits,
        softmax_exp_frac_bits: softmax.exp_bits,
        qk_score_dot_output_frac_bits: qk_score.dot_remainder_bits,
        qk_score_output_frac_bits: qk_score.score_remainder_bits,
        q_rope_first_half_output_frac_bits: q_rope.first_half_bits,
        q_rope_second_half_output_frac_bits: q_rope.second_half_bits,
        k_rope_first_half_output_frac_bits: k_rope.first_half_bits,
        k_rope_second_half_output_frac_bits: k_rope.second_half_bits,
        q_norm_norm_frac_bits: q_norm.norm_bits,
        q_norm_output_frac_bits: q_norm.output_bits,
        k_norm_norm_frac_bits: k_norm.norm_bits,
        k_norm_output_frac_bits: k_norm.output_bits,
        q_proj_output_frac_bits: q_proj.rounding_bits,
        k_proj_output_frac_bits: k_proj.rounding_bits,
        v_proj_output_frac_bits: v_proj.rounding_bits,
        rms_norm_atten_norm_frac_bits: rms_norm_atten.norm_bits,
        rms_norm_atten_output_frac_bits: rms_norm_atten.output_bits,
    };
    Some(VerifiedIopLayer { opening_claims })
}
