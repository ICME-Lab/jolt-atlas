//! `mk,kn->mn` layout (standard matrix multiply) for the einsum [`dot`](super::dot) engine.

use crate::{
    onnx_proof::ops::einsum::dot::{EinsumGeometry, EinsumLayout, EqSchedule},
    utils::dims::EinsumDims,
};
use atlas_algebra::{
    field::JoltField,
    poly::opening_point::{OpeningPoint, BIG_ENDIAN},
    utils::{errors::ProofVerifyError, math::Math},
};

/// `mk,kn->mn`: contract the shared `k` axis of two matrices.
pub struct MkKnMnLayout {
    /// Output rows.
    pub m: usize,
    /// Output columns.
    pub n: usize,
    /// Contracted axis.
    pub k: usize,
}

impl MkKnMnLayout {
    /// Build the layout from the canonical einsum dims. `Err` if they are not
    /// `right = [k, n]`, `output = [m, n]`.
    pub fn new(dims: &EinsumDims) -> Result<Self, ProofVerifyError> {
        let ([m, n], [k, _]) = (dims.output(), dims.right_operand()) else {
            return Err(ProofVerifyError::Malformed(format!(
                "model: einsum mk,kn->mn over dims {dims:?}"
            )));
        };
        Ok(Self {
            m: *m,
            n: *n,
            k: *k,
        })
    }
}

impl EinsumGeometry for MkKnMnLayout {
    fn num_rounds(&self) -> usize {
        self.k.log_2()
    }

    fn schedule(&self) -> EqSchedule {
        EqSchedule::None
    }
}

impl<F: JoltField> EinsumLayout<F> for MkKnMnLayout {
    fn operand_points(
        &self,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
        sumcheck_challenges: &[F],
    ) -> (Vec<F>, Vec<F>) {
        let (r_m, r_n) = r_node_output.split_at(self.m.log_2());
        let left = [r_m.r.as_slice(), sumcheck_challenges].concat();
        let right = [sumcheck_challenges, r_n.r.as_slice()].concat();
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
