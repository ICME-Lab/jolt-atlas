use ark_bn254::Fr;
use joltworks::transcripts::Transcript;

use crate::{BitOpeningClaims, EvalClaim, MatrixShape, SumCheckRounds};

use crate::FRAC_BITS;

pub struct RmsNormWeightedOutput {
    pub rounds: SumCheckRounds<4>,
    pub norm: Fr,
    pub output: Fr,
    pub bits: [Fr; FRAC_BITS],
}

pub struct RmsNormNormalizedInput {
    pub rounds: SumCheckRounds<4>,
    pub input: Fr,
    pub inv_rms: Fr,
    pub norm: Fr,
    pub bits: [Fr; FRAC_BITS],
}

pub struct RmsNormOutput {
    pub weighted_output: RmsNormWeightedOutput,
    pub normalized_input: RmsNormNormalizedInput,
}

pub struct RmsNormVerifierOutput {
    pub input: EvalClaim,
    pub norm_bits: BitOpeningClaims,
    pub output_bits: BitOpeningClaims,
}

pub struct RmsNormAdvice {
    pub sum_x2: Vec<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RmsNormParams {
    pub shape: MatrixShape,
}

impl RmsNormParams {
    pub fn new(rows: usize, cols: usize) -> Option<Self> {
        Some(Self {
            shape: MatrixShape::new(rows, cols)?,
        })
    }
}

pub fn draw_rms_norm_weighted_output_challenges<T>(transcript: &mut T) -> Option<(Fr, [Fr; 8])>
where
    T: Transcript,
{
    transcript.append_message(b"q3/rms_norm/weighted_output/v1");
    let round_mix = transcript.challenge_scalar_optimized::<Fr>().into();
    let bit_booleanity_challenges = transcript.challenge_vector::<Fr>(8).try_into().ok()?;
    Some((round_mix, bit_booleanity_challenges))
}

pub fn draw_rms_norm_normalized_input_challenges<T>(transcript: &mut T) -> Option<(Fr, Fr, [Fr; 8])>
where
    T: Transcript,
{
    transcript.append_message(b"q3/rms_norm/normalized_input/v1");
    let round_mix = transcript.challenge_scalar_optimized::<Fr>().into();
    let row_square_mix = transcript.challenge_scalar_optimized::<Fr>().into();
    let bit_booleanity_challenges = transcript.challenge_vector::<Fr>(8).try_into().ok()?;
    Some((round_mix, row_square_mix, bit_booleanity_challenges))
}
