use ark_bn254::Fr;
use joltworks::transcripts::Transcript;

use crate::{BitOpeningClaims, EvalClaim, SumCheckRounds};

use crate::FRAC_BITS;

pub struct RopeOutput {
    pub rounds: SumCheckRounds<4>,
    pub input_first_half: Fr,
    pub input_second_half: Fr,
    pub output_first_half: Fr,
    pub output_second_half: Fr,
    pub first_half_bits: [Fr; FRAC_BITS],
    pub second_half_bits: [Fr; FRAC_BITS],
}

pub struct RopeVerifierOutput {
    pub input_first_half: EvalClaim,
    pub input_second_half: EvalClaim,
    pub first_half_bits: BitOpeningClaims,
    pub second_half_bits: BitOpeningClaims,
    pub input_first_half_claim: Vec<EvalClaim>,
    pub input_second_half_claim: Vec<EvalClaim>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RopeParams {
    pub seq: usize,
    pub heads: usize,
    pub head_dim: usize,
}

impl RopeParams {
    pub fn new(seq: usize, heads: usize, head_dim: usize) -> Option<Self> {
        let params = Self {
            seq,
            heads,
            head_dim,
        };
        params.validate().then_some(params)
    }

    pub fn tensor_vars(&self) -> usize {
        log2(self.seq) + log2(self.heads) + log2(self.head_dim)
    }

    pub fn relation_vars(&self) -> usize {
        log2(self.seq) + log2(self.heads) + log2(self.head_dim / 2)
    }

    pub fn validate(&self) -> bool {
        self.seq.is_power_of_two()
            && self.heads.is_power_of_two()
            && self.head_dim.is_power_of_two()
            && self.head_dim >= 2
    }
}

pub fn draw_rope_challenges<T>(transcript: &mut T) -> Option<(Fr, [Fr; FRAC_BITS], [Fr; FRAC_BITS])>
where
    T: Transcript,
{
    transcript.append_message(b"q3/rope/v1");
    let rotation_constraint_mix = transcript.challenge_scalar_optimized::<Fr>().into();
    let first_half_bit_booleanity_challenges = transcript
        .challenge_vector::<Fr>(FRAC_BITS)
        .try_into()
        .ok()?;
    let second_half_bit_booleanity_challenges = transcript
        .challenge_vector::<Fr>(FRAC_BITS)
        .try_into()
        .ok()?;
    Some((
        rotation_constraint_mix,
        first_half_bit_booleanity_challenges,
        second_half_bit_booleanity_challenges,
    ))
}

fn log2(value: usize) -> usize {
    value.ilog2() as usize
}
