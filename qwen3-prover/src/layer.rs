use ark_bn254::{Bn254, Fr};
use ark_ff::One;
use joltworks::{
    field::JoltField,
    poly::commitment::hyperkzg::{HyperKZGCommitment, HyperKZGProverKey},
    transcripts::Transcript,
};
pub use qwen3_common::{
    BitOpeningClaims, EvalClaim, FRAC_BITS, IopLayerProof, LayerLookupRanges, LayerOpeningClaims,
    LayerOpeningDomainLengths, LayerOpeningDomainMax, LayerShape, RaOpeningClaims,
    append_eval_claim, draw_hidden_out_claim,
};

use crate::{
    commitment::{
        CommitLayerParams, LayerCommitments, LayerHiddenCommitments, append_layer_commitments,
        commit_layer_openings_with_hidden,
    },
    opening::{
        ChunkedLayerPcsOpeningProof, OpeningReductionOutput, prove_chunked_layer_pcs_opening,
        prove_layer_opening_reduction_sumcheck,
    },
    ops::{
        add::AddProverInput, matmul::MatMulProverInput, mul::MulProverInput, prove_add,
        prove_add_claims, prove_matmul, prove_mul, prove_pv_matmul, prove_qk_score, prove_rms_norm,
        prove_rope, prove_silu, prove_softmax, pv_matmul::PvMatmulProverInput,
        qk_score::QkScoreProverInput, rms_norm::RmsNormProverInput, rope::RopeProverInput,
        silu::SiluProverInput, softmax::SoftmaxProverInput,
    },
    profile,
};

pub struct LayerProverInput {
    pub shape: LayerShape,
    pub opening_witnesses: LayerOpeningWitnesses,
    pub residual_add_mlp: AddProverInput,
    pub down_proj: MatMulProverInput,
    pub silu_up: MulProverInput,
    pub silu: SiluProverInput,
    pub gate_proj: MatMulProverInput,
    pub up_proj: MatMulProverInput,
    pub rms_norm_mlp: RmsNormProverInput,
    pub residual_add_attn: AddProverInput,
    pub o_proj: MatMulProverInput,
    pub pv_matmul: PvMatmulProverInput,
    pub softmax: SoftmaxProverInput,
    pub qk_score: QkScoreProverInput,
    pub q_rope: RopeProverInput,
    pub k_rope: RopeProverInput,
    pub q_norm: RmsNormProverInput,
    pub k_norm: RmsNormProverInput,
    pub q_proj: MatMulProverInput,
    pub k_proj: MatMulProverInput,
    pub v_proj: MatMulProverInput,
    pub rms_norm_atten: RmsNormProverInput,
}

impl LayerProverInput {
    pub fn shape(&self) -> Option<LayerShape> {
        self.shape.validate().then_some(self.shape)
    }
}

pub type BitOpeningWitnesses = [Vec<bool>; FRAC_BITS];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerOpeningWitnesses {
    pub hidden_out: Vec<i32>,
    pub hidden_in_a: Vec<i32>,
    pub hidden_in_b: Vec<i32>,
    pub silu_lookup_ra: Vec<u8>,
    pub softmax_lookup_ra: Vec<u8>,
    pub down_proj_output_frac_bits: BitOpeningWitnesses,
    pub silu_up_output_frac_bits: BitOpeningWitnesses,
    pub silu_input_frac_bits: BitOpeningWitnesses,
    pub silu_output_frac_bits: BitOpeningWitnesses,
    pub gate_proj_output_frac_bits: BitOpeningWitnesses,
    pub up_proj_output_frac_bits: BitOpeningWitnesses,
    pub rms_norm_mlp_norm_frac_bits: BitOpeningWitnesses,
    pub rms_norm_mlp_output_frac_bits: BitOpeningWitnesses,
    pub o_proj_output_frac_bits: BitOpeningWitnesses,
    pub pv_matmul_output_frac_bits: BitOpeningWitnesses,
    pub softmax_floor_frac_bits: BitOpeningWitnesses,
    pub softmax_output_frac_bits: BitOpeningWitnesses,
    pub softmax_exp_frac_bits: BitOpeningWitnesses,
    pub qk_score_dot_output_frac_bits: BitOpeningWitnesses,
    pub qk_score_output_frac_bits: BitOpeningWitnesses,
    pub q_rope_output_frac_bits: BitOpeningWitnesses,
    pub k_rope_output_frac_bits: BitOpeningWitnesses,
    pub q_norm_norm_frac_bits: BitOpeningWitnesses,
    pub q_norm_output_frac_bits: BitOpeningWitnesses,
    pub k_norm_norm_frac_bits: BitOpeningWitnesses,
    pub k_norm_output_frac_bits: BitOpeningWitnesses,
    pub q_proj_output_frac_bits: BitOpeningWitnesses,
    pub k_proj_output_frac_bits: BitOpeningWitnesses,
    pub v_proj_output_frac_bits: BitOpeningWitnesses,
    pub rms_norm_atten_norm_frac_bits: BitOpeningWitnesses,
    pub rms_norm_atten_output_frac_bits: BitOpeningWitnesses,
}

pub struct LayerOutput {
    pub commitments: LayerCommitments<HyperKZGCommitment<Bn254>>,
    pub iop: IopLayerOutput,
    pub opening: OpeningReductionOutput,
    pub pcs_opening: ChunkedLayerPcsOpeningProof,
}

pub struct IopLayerOutput {
    pub proof: IopLayerProof,
    pub opening_claims: LayerOpeningClaims,
    pub opening_witnesses: LayerOpeningWitnesses,
}

pub fn prove_layer<T>(
    input: LayerProverInput,
    hidden_commitments: LayerHiddenCommitments<HyperKZGCommitment<Bn254>>,
    commit_params: CommitLayerParams,
    setup: &HyperKZGProverKey<Bn254>,
    transcript: &mut T,
) -> Option<LayerOutput>
where
    T: Transcript,
{
    let shape = input.shape()?;
    // A layer proof has three phases after witness commitment:
    //
    // 1. absorb pre-committed hidden state and commit RA/bit witnesses,
    // 2. run the op-by-op IOP claim-reduction chain,
    // 3. reduce all produced EvalClaims to one opening claim and prove the
    //    corresponding chunked PCS openings.
    let commitments = profile::measure("commit_layer", || {
        commit_layer_openings_with_hidden(
            hidden_commitments,
            &input.opening_witnesses,
            shape,
            commit_params,
            setup,
        )
        .ok()
    })?;
    append_layer_commitments(transcript, &commitments);

    let hidden_out = draw_hidden_out_claim_from_witness(
        transcript,
        &input.opening_witnesses.hidden_out,
        shape,
    )?;
    let iop = prove_iop_layer_from_claim(hidden_out, input, transcript)?;
    let opening = profile::measure("opening_reduction", || {
        prove_layer_opening_reduction_sumcheck(
            &iop.opening_claims,
            &iop.opening_witnesses,
            shape,
            transcript,
        )
    })?;
    let pcs_opening = profile::measure("chunked_pcs_open", || {
        prove_chunked_layer_pcs_opening(
            &iop.opening_claims,
            &iop.opening_witnesses,
            shape,
            &opening,
            &commitments,
            setup,
            transcript,
        )
    })?;

    Some(LayerOutput {
        commitments,
        iop,
        opening,
        pcs_opening,
    })
}

pub fn prove_iop_layer<T>(
    hidden_out_value: Fr,
    input: LayerProverInput,
    transcript: &mut T,
) -> Option<IopLayerOutput>
where
    T: Transcript,
{
    let shape = input.shape()?;
    let hidden_out = draw_hidden_out_claim(transcript, hidden_out_value, shape)?;
    prove_iop_layer_from_claim(hidden_out, input, transcript)
}

fn draw_hidden_out_claim_from_witness<T>(
    transcript: &mut T,
    hidden_out: &[i32],
    shape: LayerShape,
) -> Option<EvalClaim>
where
    T: Transcript,
{
    let shape = shape.padded();
    let len = shape.seq.checked_mul(shape.hidden)?;
    (hidden_out.len() == len && len.is_power_of_two()).then_some(())?;
    let point = transcript.challenge_vector::<Fr>(len.ilog2() as usize);
    let value = eval_i32_table(hidden_out, &point)?;
    transcript.append_scalar(&value);
    Some(EvalClaim::new(value, point))
}

fn eval_i32_table(values: &[i32], point: &[Fr]) -> Option<Fr> {
    (values.len() == (1_usize << point.len())).then_some(())?;
    Some(
        values
            .iter()
            .enumerate()
            .map(|(index, value)| eq_eval(index, point) * Fr::from_i32(*value))
            .sum(),
    )
}

fn eq_eval(index: usize, point: &[Fr]) -> Fr {
    point
        .iter()
        .enumerate()
        .map(|(bit, value)| {
            if ((index >> bit) & 1) == 1 {
                *value
            } else {
                Fr::one() - value
            }
        })
        .product()
}

pub fn prove_iop_layer_from_claim<T>(
    hidden_out: EvalClaim,
    input: LayerProverInput,
    transcript: &mut T,
) -> Option<IopLayerOutput>
where
    T: Transcript,
{
    let _shape = input.shape()?;
    let hidden_out_value = hidden_out.value;
    let opening_witnesses = input.opening_witnesses.clone();

    // The IOP runs backward from hidden_out.  Each op proves the local relation
    // with its own sumcheck(s), then returns the EvalClaims needed by earlier
    // tensors.  The final LayerOpeningClaims list is the PCS opening workload.

    // residual_add_mlp = residual_add_attn + down_proj
    let residual_add_mlp = profile::measure("layer.op.residual_add_mlp", || {
        prove_add(hidden_out.clone(), input.residual_add_mlp, transcript)
    })?;

    // down_proj = silu_up @ W_down
    let down_proj = profile::measure("layer.op.down_proj", || {
        prove_matmul(residual_add_mlp.rhs_claim, input.down_proj, transcript)
    })?;

    // silu_up = silu * up_proj
    let silu_up = profile::measure("layer.op.silu_up", || {
        prove_mul(down_proj.lhs, input.silu_up, transcript)
    })?;

    // silu = SiLU(gate_proj)
    let silu = profile::measure("layer.op.silu", || {
        prove_silu(silu_up.lhs, input.silu, transcript)
    })?;

    // gate_proj = rms_norm_mlp @ W_gate
    let gate_proj = profile::measure("layer.op.gate_proj", || {
        prove_matmul(silu.input, input.gate_proj, transcript)
    })?;

    // up_proj = rms_norm_mlp @ W_up
    let up_proj = profile::measure("layer.op.up_proj", || {
        prove_matmul(silu_up.rhs, input.up_proj, transcript)
    })?;

    // rms_norm_mlp = RMSNorm(residual_add_attn)
    let rms_norm_mlp = profile::measure("layer.op.rms_norm_mlp", || {
        prove_rms_norm(
            vec![gate_proj.lhs, up_proj.lhs],
            input.rms_norm_mlp,
            transcript,
        )
    })?;

    // residual_add_attn = hidden_in + o_proj, with residual fanout absorbed inside add.
    let residual_add_attn = profile::measure("layer.op.residual_add_attn", || {
        prove_add_claims(
            vec![residual_add_mlp.lhs_claim.clone(), rms_norm_mlp.input],
            input.residual_add_attn,
            transcript,
        )
    })?;

    // o_proj = context @ W_o
    let o_proj = profile::measure("layer.op.o_proj", || {
        prove_matmul(residual_add_attn.rhs_claim, input.o_proj, transcript)
    })?;

    // context = softmax @ v_proj
    let pv_matmul = profile::measure("layer.op.pv_matmul", || {
        prove_pv_matmul(o_proj.lhs, input.pv_matmul, transcript)
    })?;

    // softmax = softmax(qk_score)
    let softmax = profile::measure("layer.op.softmax", || {
        prove_softmax(pv_matmul.p, input.softmax, transcript)
    })?;

    // qk_score = q_rope @ k_rope^T
    let qk_score = profile::measure("layer.op.qk_score", || {
        prove_qk_score(softmax.input, input.qk_score, transcript)
    })?;

    // q_rope = RoPE(q_norm)
    let q_rope = profile::measure("layer.op.q_rope", || {
        prove_rope(qk_score.dot.q, input.q_rope, transcript)
    })?;

    // k_rope = RoPE(k_norm)
    let k_rope = profile::measure("layer.op.k_rope", || {
        prove_rope(qk_score.dot.k, input.k_rope, transcript)
    })?;

    // q_norm = RMSNorm(q_proj)
    let q_norm = profile::measure("layer.op.q_norm", || {
        prove_rms_norm(
            vec![q_rope.input_first_half, q_rope.input_second_half],
            input.q_norm,
            transcript,
        )
    })?;

    // k_norm = RMSNorm(k_proj)
    let k_norm = profile::measure("layer.op.k_norm", || {
        prove_rms_norm(
            vec![k_rope.input_first_half, k_rope.input_second_half],
            input.k_norm,
            transcript,
        )
    })?;

    // q_proj = rms_norm_atten @ W_q
    let q_proj = profile::measure("layer.op.q_proj", || {
        prove_matmul(q_norm.input, input.q_proj, transcript)
    })?;

    // k_proj = rms_norm_atten @ W_k
    let k_proj = profile::measure("layer.op.k_proj", || {
        prove_matmul(k_norm.input, input.k_proj, transcript)
    })?;

    // v_proj = rms_norm_atten @ W_v
    let v_proj = profile::measure("layer.op.v_proj", || {
        prove_matmul(pv_matmul.v, input.v_proj, transcript)
    })?;

    // rms_norm_atten = RMSNorm(hidden_in)
    let rms_norm_atten = profile::measure("layer.op.rms_norm_atten", || {
        prove_rms_norm(
            vec![q_proj.lhs, k_proj.lhs, v_proj.lhs],
            input.rms_norm_atten,
            transcript,
        )
    })?;

    let hidden_in_a = residual_add_attn.lhs_claim;
    let hidden_in_b = rms_norm_atten.input;

    let opening_claims = LayerOpeningClaims {
        hidden_out,
        hidden_in_a,
        hidden_in_b,
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
    Some(IopLayerOutput {
        opening_claims,
        opening_witnesses,
        proof: IopLayerProof {
            hidden_out: hidden_out_value,
            residual_add_mlp: residual_add_mlp.proof,
            down_proj: down_proj.proof,
            silu_up: silu_up.proof,
            silu: silu.proof,
            gate_proj: gate_proj.proof,
            up_proj: up_proj.proof,
            rms_norm_mlp: rms_norm_mlp.proof,
            residual_add_attn: residual_add_attn.proof,
            o_proj: o_proj.proof,
            pv_matmul: pv_matmul.proof,
            softmax: softmax.proof,
            qk_score: qk_score.proof,
            q_rope: q_rope.proof,
            k_rope: k_rope.proof,
            q_norm: q_norm.proof,
            k_norm: k_norm.proof,
            q_proj: q_proj.proof,
            k_proj: k_proj.proof,
            v_proj: v_proj.proof,
            rms_norm_atten: rms_norm_atten.proof,
        },
    })
}
