use ark_bn254::Fr;

use crate::{EvalClaim, MatrixShape, SumCheckRounds};

pub struct AddOutput {
    pub lhs: Fr,
    pub rhs: Fr,
    pub rounds: Option<SumCheckRounds<3>>,
}

pub struct AddVerifierOutput {
    pub lhs_claim: EvalClaim,
    pub rhs_claim: EvalClaim,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddParams {
    pub shape: MatrixShape,
}

impl AddParams {
    pub fn new(rows: usize, cols: usize) -> Option<Self> {
        Some(Self {
            shape: MatrixShape::new(rows, cols)?,
        })
    }
}
