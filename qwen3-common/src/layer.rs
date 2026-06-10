use ark_bn254::Fr;
use joltworks::transcripts::Transcript;

use crate::EvalClaim;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerShape {
    pub seq: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub hidden: usize,
    pub intermediate: usize,
}

pub const QWEN3_HIDDEN: usize = 1024;
pub const QWEN3_INTERMEDIATE: usize = 3072;
pub const QWEN3_Q_HEADS: usize = 16;
pub const QWEN3_KV_HEADS: usize = 8;
pub const QWEN3_HEAD_DIM: usize = 128;

pub fn qwen3_layer_shape(seq: usize) -> Option<LayerShape> {
    let shape = LayerShape {
        seq,
        q_heads: QWEN3_Q_HEADS,
        kv_heads: QWEN3_KV_HEADS,
        head_dim: QWEN3_HEAD_DIM,
        hidden: QWEN3_HIDDEN,
        intermediate: QWEN3_INTERMEDIATE,
    }
    .padded();
    shape.validate().then_some(shape)
}

pub fn draw_hidden_out_claim<T>(
    transcript: &mut T,
    value: Fr,
    shape: LayerShape,
) -> Option<EvalClaim>
where
    T: Transcript,
{
    let shape = shape.padded();
    let len = shape.seq.checked_mul(shape.hidden)?;
    len.is_power_of_two().then_some(())?;
    let point = transcript.challenge_vector::<Fr>(len.ilog2() as usize);
    transcript.append_scalar(&value);
    Some(EvalClaim::new(value, point))
}

impl LayerShape {
    pub fn validate(&self) -> bool {
        self.seq.is_power_of_two()
            && self.q_heads.is_power_of_two()
            && self.kv_heads.is_power_of_two()
            && self.head_dim.is_power_of_two()
            && self.hidden.is_power_of_two()
            && self.intermediate.is_power_of_two()
    }

    pub fn padded(&self) -> Self {
        Self {
            seq: self.seq.next_power_of_two(),
            q_heads: self.q_heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            hidden: self.hidden,
            intermediate: self.intermediate.next_power_of_two(),
        }
    }

    pub fn opening_domain_lengths(
        &self,
        ranges: LayerLookupRanges,
    ) -> Option<LayerOpeningDomainLengths> {
        let shape = self.padded();
        let hidden = shape.seq.checked_mul(shape.hidden)?;
        let intermediate = shape.seq.checked_mul(shape.intermediate)?;
        let q_tensor = shape
            .seq
            .checked_mul(shape.q_heads)?
            .checked_mul(shape.head_dim)?;
        let kv_tensor = shape
            .seq
            .checked_mul(shape.kv_heads)?
            .checked_mul(shape.head_dim)?;
        let score = shape
            .q_heads
            .checked_mul(shape.seq)?
            .checked_mul(shape.seq)?;
        let context = shape
            .seq
            .checked_mul(shape.q_heads)?
            .checked_mul(shape.head_dim)?;
        let silu_lookup_ra =
            intermediate.checked_mul(silu_padded_lut_len(ranges.silu_entries()?))?;
        let softmax_lookup_ra =
            score.checked_mul(softmax_padded_lut_len(ranges.softmax_entries()?))?;
        Some(LayerOpeningDomainLengths {
            hidden_out: hidden,
            hidden_in_a: hidden,
            hidden_in_b: hidden,
            silu_lookup_ra,
            softmax_lookup_ra,
            down_proj_output_frac_bits: hidden,
            silu_up_output_frac_bits: intermediate,
            silu_input_frac_bits: intermediate,
            silu_output_frac_bits: intermediate,
            gate_proj_output_frac_bits: intermediate,
            up_proj_output_frac_bits: intermediate,
            rms_norm_mlp_norm_frac_bits: hidden,
            rms_norm_mlp_output_frac_bits: hidden,
            o_proj_output_frac_bits: hidden,
            pv_matmul_output_frac_bits: context,
            softmax_floor_frac_bits: score,
            softmax_output_frac_bits: score,
            softmax_exp_frac_bits: score,
            qk_score_dot_output_frac_bits: score,
            qk_score_output_frac_bits: score,
            q_rope_output_frac_bits: q_tensor,
            k_rope_output_frac_bits: kv_tensor,
            q_norm_norm_frac_bits: q_tensor,
            q_norm_output_frac_bits: q_tensor,
            k_norm_norm_frac_bits: kv_tensor,
            k_norm_output_frac_bits: kv_tensor,
            q_proj_output_frac_bits: q_tensor,
            k_proj_output_frac_bits: kv_tensor,
            v_proj_output_frac_bits: kv_tensor,
            rms_norm_atten_norm_frac_bits: hidden,
            rms_norm_atten_output_frac_bits: hidden,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerLookupRanges {
    pub silu_min_n: i64,
    pub silu_max_n: i64,
    pub softmax_min_diff: i64,
    pub softmax_max_diff: i64,
}

impl LayerLookupRanges {
    fn silu_entries(&self) -> Option<usize> {
        entries(self.silu_min_n, self.silu_max_n)
    }

    fn softmax_entries(&self) -> Option<usize> {
        entries(self.softmax_min_diff, self.softmax_max_diff)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerOpeningDomainMax {
    pub name: &'static str,
    pub len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerOpeningDomainLengths {
    pub hidden_out: usize,
    pub hidden_in_a: usize,
    pub hidden_in_b: usize,
    pub silu_lookup_ra: usize,
    pub softmax_lookup_ra: usize,
    pub down_proj_output_frac_bits: usize,
    pub silu_up_output_frac_bits: usize,
    pub silu_input_frac_bits: usize,
    pub silu_output_frac_bits: usize,
    pub gate_proj_output_frac_bits: usize,
    pub up_proj_output_frac_bits: usize,
    pub rms_norm_mlp_norm_frac_bits: usize,
    pub rms_norm_mlp_output_frac_bits: usize,
    pub o_proj_output_frac_bits: usize,
    pub pv_matmul_output_frac_bits: usize,
    pub softmax_floor_frac_bits: usize,
    pub softmax_output_frac_bits: usize,
    pub softmax_exp_frac_bits: usize,
    pub qk_score_dot_output_frac_bits: usize,
    pub qk_score_output_frac_bits: usize,
    pub q_rope_output_frac_bits: usize,
    pub k_rope_output_frac_bits: usize,
    pub q_norm_norm_frac_bits: usize,
    pub q_norm_output_frac_bits: usize,
    pub k_norm_norm_frac_bits: usize,
    pub k_norm_output_frac_bits: usize,
    pub q_proj_output_frac_bits: usize,
    pub k_proj_output_frac_bits: usize,
    pub v_proj_output_frac_bits: usize,
    pub rms_norm_atten_norm_frac_bits: usize,
    pub rms_norm_atten_output_frac_bits: usize,
}

impl LayerOpeningDomainLengths {
    pub fn max(&self) -> LayerOpeningDomainMax {
        let mut max = LayerOpeningDomainMax {
            name: "hidden_out",
            len: self.hidden_out,
        };
        for candidate in [
            ("hidden_in_a", self.hidden_in_a),
            ("hidden_in_b", self.hidden_in_b),
            ("silu_lookup_ra", self.silu_lookup_ra),
            ("softmax_lookup_ra", self.softmax_lookup_ra),
            (
                "down_proj_output_frac_bits",
                self.down_proj_output_frac_bits,
            ),
            ("silu_up_output_frac_bits", self.silu_up_output_frac_bits),
            ("silu_input_frac_bits", self.silu_input_frac_bits),
            ("silu_output_frac_bits", self.silu_output_frac_bits),
            (
                "gate_proj_output_frac_bits",
                self.gate_proj_output_frac_bits,
            ),
            ("up_proj_output_frac_bits", self.up_proj_output_frac_bits),
            (
                "rms_norm_mlp_norm_frac_bits",
                self.rms_norm_mlp_norm_frac_bits,
            ),
            (
                "rms_norm_mlp_output_frac_bits",
                self.rms_norm_mlp_output_frac_bits,
            ),
            ("o_proj_output_frac_bits", self.o_proj_output_frac_bits),
            (
                "pv_matmul_output_frac_bits",
                self.pv_matmul_output_frac_bits,
            ),
            ("softmax_floor_frac_bits", self.softmax_floor_frac_bits),
            ("softmax_output_frac_bits", self.softmax_output_frac_bits),
            ("softmax_exp_frac_bits", self.softmax_exp_frac_bits),
            (
                "qk_score_dot_output_frac_bits",
                self.qk_score_dot_output_frac_bits,
            ),
            ("qk_score_output_frac_bits", self.qk_score_output_frac_bits),
            ("q_rope_output_frac_bits", self.q_rope_output_frac_bits),
            ("k_rope_output_frac_bits", self.k_rope_output_frac_bits),
            ("q_norm_norm_frac_bits", self.q_norm_norm_frac_bits),
            ("q_norm_output_frac_bits", self.q_norm_output_frac_bits),
            ("k_norm_norm_frac_bits", self.k_norm_norm_frac_bits),
            ("k_norm_output_frac_bits", self.k_norm_output_frac_bits),
            ("q_proj_output_frac_bits", self.q_proj_output_frac_bits),
            ("k_proj_output_frac_bits", self.k_proj_output_frac_bits),
            ("v_proj_output_frac_bits", self.v_proj_output_frac_bits),
            (
                "rms_norm_atten_norm_frac_bits",
                self.rms_norm_atten_norm_frac_bits,
            ),
            (
                "rms_norm_atten_output_frac_bits",
                self.rms_norm_atten_output_frac_bits,
            ),
        ] {
            if candidate.1 > max.len {
                max = LayerOpeningDomainMax {
                    name: candidate.0,
                    len: candidate.1,
                };
            }
        }
        max
    }
}

fn entries(min: i64, max: i64) -> Option<usize> {
    (max >= min)
        .then_some(max - min + 1)
        .and_then(|entries| usize::try_from(entries).ok())
}

fn silu_padded_lut_len(entries: usize) -> usize {
    (entries + 1).next_power_of_two().max(16)
}

fn softmax_padded_lut_len(entries: usize) -> usize {
    entries.next_power_of_two().max(2)
}

pub type BitOpeningClaims = [crate::EvalClaim; 8];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaOpeningClaims {
    pub read: crate::EvalClaim,
    pub virtual_claim: crate::EvalClaim,
    pub hamming_weight: crate::EvalClaim,
    pub booleanity: crate::EvalClaim,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerOpeningClaims {
    pub hidden_out: crate::EvalClaim,
    pub hidden_in_a: crate::EvalClaim,
    pub hidden_in_b: crate::EvalClaim,
    pub silu_lookup_ra: RaOpeningClaims,
    pub softmax_lookup_ra: RaOpeningClaims,
    pub down_proj_output_frac_bits: BitOpeningClaims,
    pub silu_up_output_frac_bits: BitOpeningClaims,
    pub silu_input_frac_bits: BitOpeningClaims,
    pub silu_output_frac_bits: BitOpeningClaims,
    pub gate_proj_output_frac_bits: BitOpeningClaims,
    pub up_proj_output_frac_bits: BitOpeningClaims,
    pub rms_norm_mlp_norm_frac_bits: BitOpeningClaims,
    pub rms_norm_mlp_output_frac_bits: BitOpeningClaims,
    pub o_proj_output_frac_bits: BitOpeningClaims,
    pub pv_matmul_output_frac_bits: BitOpeningClaims,
    pub softmax_floor_frac_bits: BitOpeningClaims,
    pub softmax_output_frac_bits: BitOpeningClaims,
    pub softmax_exp_frac_bits: BitOpeningClaims,
    pub qk_score_dot_output_frac_bits: BitOpeningClaims,
    pub qk_score_output_frac_bits: BitOpeningClaims,
    pub q_rope_first_half_output_frac_bits: BitOpeningClaims,
    pub q_rope_second_half_output_frac_bits: BitOpeningClaims,
    pub k_rope_first_half_output_frac_bits: BitOpeningClaims,
    pub k_rope_second_half_output_frac_bits: BitOpeningClaims,
    pub q_norm_norm_frac_bits: BitOpeningClaims,
    pub q_norm_output_frac_bits: BitOpeningClaims,
    pub k_norm_norm_frac_bits: BitOpeningClaims,
    pub k_norm_output_frac_bits: BitOpeningClaims,
    pub q_proj_output_frac_bits: BitOpeningClaims,
    pub k_proj_output_frac_bits: BitOpeningClaims,
    pub v_proj_output_frac_bits: BitOpeningClaims,
    pub rms_norm_atten_norm_frac_bits: BitOpeningClaims,
    pub rms_norm_atten_output_frac_bits: BitOpeningClaims,
}

pub struct IopLayerProof {
    pub hidden_out: Fr,
    pub residual_add_mlp: crate::ops::add::AddOutput,
    pub down_proj: crate::ops::matmul::MatMulOutput,
    pub silu_up: crate::ops::mul::MulOutput,
    pub silu: crate::ops::silu::SiluOutput,
    pub gate_proj: crate::ops::matmul::MatMulOutput,
    pub up_proj: crate::ops::matmul::MatMulOutput,
    pub rms_norm_mlp: crate::ops::rms_norm::RmsNormOutput,
    pub residual_add_attn: crate::ops::add::AddOutput,
    pub o_proj: crate::ops::matmul::MatMulOutput,
    pub pv_matmul: crate::ops::pv_matmul::PvMatmulOutput,
    pub softmax: crate::ops::softmax::SoftmaxOutput,
    pub qk_score: crate::ops::qk_score::QkScoreOutput,
    pub q_rope: crate::ops::rope::RopeOutput,
    pub k_rope: crate::ops::rope::RopeOutput,
    pub q_norm: crate::ops::rms_norm::RmsNormOutput,
    pub k_norm: crate::ops::rms_norm::RmsNormOutput,
    pub q_proj: crate::ops::matmul::MatMulOutput,
    pub k_proj: crate::ops::matmul::MatMulOutput,
    pub v_proj: crate::ops::matmul::MatMulOutput,
    pub rms_norm_atten: crate::ops::rms_norm::RmsNormOutput,
}

pub struct LayerVerifierPublicInput {
    pub seq: usize,
    pub down_proj_weight: Vec<i32>,
    pub silu: LayerSiluVerifierInput,
    pub gate_proj_weight: Vec<i32>,
    pub up_proj_weight: Vec<i32>,
    pub rms_norm_mlp: LayerRmsNormVerifierInput,
    pub o_proj_weight: Vec<i32>,
    pub softmax: LayerSoftmaxVerifierInput,
    pub q_rope: LayerRopeVerifierInput,
    pub k_rope: LayerRopeVerifierInput,
    pub q_norm: LayerRmsNormVerifierInput,
    pub k_norm: LayerRmsNormVerifierInput,
    pub q_proj_weight: Vec<i32>,
    pub k_proj_weight: Vec<i32>,
    pub v_proj_weight: Vec<i32>,
    pub rms_norm_atten: LayerRmsNormVerifierInput,
}

pub struct LayerSiluVerifierInput {
    pub advice: crate::ops::silu::SiluAdvice,
}

pub struct LayerSoftmaxVerifierInput {
    pub advice: crate::ops::softmax::SoftmaxAdvice,
}

pub struct LayerRmsNormVerifierInput {
    pub advice: crate::ops::rms_norm::RmsNormAdvice,
    pub weight: Vec<i32>,
}

pub struct LayerRopeVerifierInput {
    pub cos: Vec<i32>,
    pub sin: Vec<i32>,
}
