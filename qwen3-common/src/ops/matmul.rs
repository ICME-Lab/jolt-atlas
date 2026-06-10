use ark_bn254::Fr;
use joltworks::transcripts::Transcript;

use crate::{BitOpeningClaims, EvalClaim, MatrixShape, SumCheckRounds};

use crate::FRAC_BITS;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatMulReduction {
    pub rounds: SumCheckRounds<3>,
}

pub struct MatMulRoundingBits {
    pub rounds: SumCheckRounds<4>,
    pub bits: [Fr; FRAC_BITS],
}

pub struct MatMulOutput {
    pub rounding_bits: MatMulRoundingBits,
    pub k_reduction: MatMulReduction,
    pub rem: Fr,
    pub msb: Fr,
    pub lhs: Fr,
    pub rhs: Fr,
}

pub struct MatMulVerifierOutput {
    pub lhs: EvalClaim,
    pub rhs: EvalClaim,
    pub rounding_bits: BitOpeningClaims,
}

pub struct MatMulVerifierInput {
    pub params: MatMulParams,
    pub weight: Vec<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatMulParams {
    pub output_shape: MatrixShape,
    pub inner: usize,
}

impl MatMulParams {
    pub fn new(rows: usize, cols: usize, inner: usize) -> Option<Self> {
        (inner.is_power_of_two()).then_some(Self {
            output_shape: MatrixShape::new(rows, cols)?,
            inner,
        })
    }

    pub fn lhs_shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.output_shape.rows,
            cols: self.inner,
        }
    }

    pub fn rhs_shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.inner,
            cols: self.output_shape.cols,
        }
    }
}

pub fn draw_matmul_rounding_bit_challenges<T>(
    transcript: &mut T,
) -> Option<([Fr; 2], [Fr; FRAC_BITS])>
where
    T: Transcript,
{
    transcript.append_message(b"q3/matmul/rounding-bits/v1");
    let rounding_challenges = transcript.challenge_vector::<Fr>(2).try_into().ok()?;
    let booleanity_challenges = transcript
        .challenge_vector::<Fr>(FRAC_BITS)
        .try_into()
        .ok()?;
    Some((rounding_challenges, booleanity_challenges))
}
