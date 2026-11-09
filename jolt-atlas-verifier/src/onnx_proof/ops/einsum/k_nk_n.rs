//! `k,nk->n` layout (matrix-vector multiply) for the einsum [`dot`](super::dot) engine.

use crate::{
    onnx_proof::ops::einsum::dot::{EinsumGeometry, EinsumLayout, EqSchedule},
    utils::dims::EinsumDims,
};
use atlas_algebra::{
    field::JoltField,
    poly::opening_point::{OpeningPoint, BIG_ENDIAN},
    utils::{errors::ProofVerifyError, math::Math},
};

/// `k,nk->n`: contract the `k` axis of a vector with a matrix. The left operand
/// is the raw vector (no folding); the right operand is folded over the output `n`.
pub struct KNkNLayout {
    /// Output length.
    pub n: usize,
    /// Contracted axis.
    pub k: usize,
}

impl KNkNLayout {
    /// Build the layout from the canonical einsum dims. `Err` if they are not
    /// `left = [k]`, `output = [n]`.
    pub fn new(dims: &EinsumDims) -> Result<Self, ProofVerifyError> {
        let ([n], [k]) = (dims.output(), dims.left_operand()) else {
            return Err(ProofVerifyError::Malformed(format!(
                "model: einsum k,nk->n over dims {dims:?}"
            )));
        };
        Ok(Self { n: *n, k: *k })
    }
}

impl EinsumGeometry for KNkNLayout {
    fn num_rounds(&self) -> usize {
        self.k.log_2()
    }

    fn schedule(&self) -> EqSchedule {
        EqSchedule::None
    }
}

impl<F: JoltField> EinsumLayout<F> for KNkNLayout {
    fn operand_points(
        &self,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
        sumcheck_challenges: &[F],
    ) -> (Vec<F>, Vec<F>) {
        let left = sumcheck_challenges.to_vec();
        let right = [r_node_output.r.as_slice(), sumcheck_challenges].concat();
        (left, right)
    }

    fn output_eq(
        &self,
        _r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
        _sumcheck_challenges: &[F],
    ) -> F {
        F::one()
    }
}
