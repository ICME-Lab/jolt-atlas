use ark_bn254::Fr;
use joltworks::transcripts::Transcript;

use crate::{BitOpeningClaims, EvalClaim, MatrixShape, SumCheckRounds};

use crate::FRAC_BITS;

pub struct MulOutput {
    pub rounds: SumCheckRounds<4>,
    pub lhs: Fr,
    pub rhs: Fr,
    pub bits: [Fr; FRAC_BITS],
}

pub struct MulInputEvals {
    pub lhs: Fr,
    pub rhs: Fr,
    pub bits: [Fr; FRAC_BITS],
}

pub struct MulPublicEvals {
    pub eq: Fr,
    pub booleanity: [Fr; FRAC_BITS],
}

pub struct MulVerifierOutput {
    pub point: Vec<Fr>,
    pub input_evals: MulInputEvals,
    pub lhs: EvalClaim,
    pub rhs: EvalClaim,
    pub bits: BitOpeningClaims,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MulParams {
    pub len: usize,
    shape: Option<MatrixShape>,
}

impl MulParams {
    pub fn new(len: usize) -> Option<Self> {
        len.is_power_of_two().then_some(Self { len, shape: None })
    }

    pub fn matrix(rows: usize, cols: usize) -> Option<Self> {
        let shape = MatrixShape::new(rows, cols)?;
        Some(Self {
            len: shape.len(),
            shape: Some(shape),
        })
    }

    pub fn shape(&self) -> Option<MatrixShape> {
        self.shape
    }
}

pub fn draw_mul_booleanity_challenges<T>(transcript: &mut T) -> Option<[Fr; FRAC_BITS]>
where
    T: Transcript,
{
    transcript.append_message(b"q3/mul/booleanity/v1");
    transcript.challenge_vector::<Fr>(FRAC_BITS).try_into().ok()
}
