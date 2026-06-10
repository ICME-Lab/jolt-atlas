use ark_bn254::Fr;
use joltworks::transcripts::Transcript;

use crate::{BitOpeningClaims, EvalClaim, SumCheckRounds};

use crate::FRAC_BITS;
const QWEN3_GQA_GROUP_SIZE: usize = 2;

pub struct PvMatmulOutput {
    pub rounding_bits: PvMatmulRoundingBits,
    pub k_reduction: PvMatmulReduction,
    pub rem: Fr,
    pub msb: Fr,
    pub p: Fr,
    pub v: Fr,
}

pub struct PvMatmulReduction {
    pub rounds: SumCheckRounds<3>,
}

pub struct PvMatmulRoundingBits {
    pub rounds: SumCheckRounds<4>,
    pub context_remainder_bits: [Fr; FRAC_BITS],
}

pub struct PvMatmulVerifierOutput {
    pub p: EvalClaim,
    pub v: EvalClaim,
    pub context_remainder_bits: BitOpeningClaims,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PvMatmulParams {
    pub seq: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl PvMatmulParams {
    pub fn new(seq: usize, q_heads: usize, kv_heads: usize, head_dim: usize) -> Option<Self> {
        let params = Self {
            seq,
            q_heads,
            kv_heads,
            head_dim,
        };
        params.validate().then_some(params)
    }

    pub fn context_vars(&self) -> usize {
        log2(self.seq) + log2(self.q_heads) + log2(self.head_dim)
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

pub fn draw_pv_matmul_challenges<T>(transcript: &mut T) -> Option<()>
where
    T: Transcript,
{
    transcript.append_message(b"q3/pv_matmul/v1");
    Some(())
}

pub fn draw_pv_matmul_bit_booleanity_challenges<T>(
    transcript: &mut T,
) -> Option<([Fr; 2], [Fr; FRAC_BITS])>
where
    T: Transcript,
{
    transcript.append_message(b"q3/pv_matmul/bits/v1");
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
