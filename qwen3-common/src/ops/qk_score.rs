use ark_bn254::Fr;
use joltworks::transcripts::Transcript;

use crate::{BitOpeningClaims, EvalClaim, SumCheckRounds};

use crate::FRAC_BITS;
const QWEN3_GQA_GROUP_SIZE: usize = 2;

pub struct QkScoreOutput {
    pub scale: QkScoreScaleOutput,
    pub dot: QkScoreDotOutput,
}

pub struct QkScoreScaleOutput {
    pub rounds: SumCheckRounds<4>,
    pub dot: Fr,
    pub score_remainder_bits: [Fr; FRAC_BITS],
}

pub struct QkScoreDotOutput {
    pub rounding_bits: QkScoreDotRoundingBits,
    pub k_reduction: QkScoreDotReduction,
    pub rem: Fr,
    pub msb: Fr,
    pub q: Fr,
    pub k: Fr,
}

pub struct QkScoreDotReduction {
    pub rounds: SumCheckRounds<3>,
}

pub struct QkScoreDotRoundingBits {
    pub rounds: SumCheckRounds<4>,
    pub dot_remainder_bits: [Fr; FRAC_BITS],
}

pub struct QkScoreDotClaims {
    pub q: EvalClaim,
    pub k: EvalClaim,
}

pub struct QkScoreVerifierOutput {
    pub dot: QkScoreDotClaims,
    pub score_remainder_bits: BitOpeningClaims,
    pub dot_remainder_bits: BitOpeningClaims,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QkScoreParams {
    pub seq: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl QkScoreParams {
    pub fn new(seq: usize, q_heads: usize, kv_heads: usize, head_dim: usize) -> Option<Self> {
        let params = Self {
            seq,
            q_heads,
            kv_heads,
            head_dim,
        };
        params.validate().then_some(params)
    }

    pub fn score_vars(&self) -> usize {
        log2(self.q_heads) + log2(self.seq) + log2(self.seq)
    }

    pub fn head_vars(&self) -> usize {
        log2(self.q_heads)
    }

    pub fn validate(&self) -> bool {
        self.seq.is_power_of_two()
            && self.q_heads.is_power_of_two()
            && self.kv_heads.is_power_of_two()
            && self.head_dim.is_power_of_two()
            && self.q_heads == self.kv_heads * QWEN3_GQA_GROUP_SIZE
    }
}

pub fn draw_qk_score_scale_challenges<T>(transcript: &mut T) -> Option<[Fr; FRAC_BITS]>
where
    T: Transcript,
{
    transcript.append_message(b"q3/qk_score/scale/v1");
    transcript.challenge_vector::<Fr>(FRAC_BITS).try_into().ok()
}

pub fn draw_qk_score_dot_challenges<T>(transcript: &mut T) -> Option<()>
where
    T: Transcript,
{
    transcript.append_message(b"q3/qk_score/dot/v1");
    Some(())
}

pub fn draw_qk_score_dot_bit_booleanity_challenges<T>(
    transcript: &mut T,
) -> Option<([Fr; 2], [Fr; FRAC_BITS])>
where
    T: Transcript,
{
    transcript.append_message(b"q3/qk_score/dot-bits/v1");
    let rounding_challenges = transcript.challenge_vector::<Fr>(2).try_into().ok()?;
    let booleanity_challenges = transcript
        .challenge_vector::<Fr>(FRAC_BITS)
        .try_into()
        .ok()?;
    Some((rounding_challenges, booleanity_challenges))
}

fn log2(value: usize) -> usize {
    value.ilog2() as usize
}

pub fn qk_score_inv_sqrt_q8(head_dim: usize) -> i32 {
    ((1.0 / (head_dim as f64).sqrt()) * 256.0).round() as i32
}
