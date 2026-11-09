//! `abmk,abnk->abmn` layout for the einsum [`dot`](super::dot) engine: a
//! batched matmul (RHS contraction-last) whose retained batch axes are packed
//! in front of both operands and the output. The rank-3 form `amk,ank->amn`
//! is the identical layout with a batch pack of `(1, a)` — the flattened
//! operand memory is the same, so both normalize here.
//!
//! The batch pack is bound first through a high eq polynomial
//! ([`EqSchedule::High`]); the contraction axis `k` follows.

use crate::{
    onnx_proof::ops::einsum::dot::{EinsumGeometry, EinsumLayout, EqSchedule},
    utils::dims::EinsumDims,
};
use atlas_algebra::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        opening_point::{OpeningPoint, BIG_ENDIAN},
    },
    utils::{errors::ProofVerifyError, math::Math},
};

/// Batched matmul layout: `out[h,i,l] = Σ_j left[h,i,j]·right[h,l,j]` with the
/// batch pack `h` flattened into the leading axis by the dims extractor.
pub struct AbmkAbnkAbmnLayout {
    /// log₂ of the batch pack.
    pub log_ab: usize,
    /// log₂ of the output rows.
    pub log_m: usize,
    /// log₂ of the output columns.
    pub log_n: usize,
    /// log₂ of the contracted axis.
    pub log_k: usize,
}

impl AbmkAbnkAbmnLayout {
    /// Build the layout from the canonical einsum dims. The extractors hand us
    /// `left = [ab, m, k]`, `right = [ab, n, k]`, `output = [ab, m, n]`; `Err` otherwise.
    pub fn new(dims: &EinsumDims) -> Result<Self, ProofVerifyError> {
        let unsupported = |what: &str| {
            ProofVerifyError::Malformed(format!(
                "model: einsum abmk,abnk->abmn over dims {dims:?}: {what}"
            ))
        };
        let ([left_ab, _, left_k], [right_ab, _, right_k], [ab, m, n]) =
            (dims.left_operand(), dims.right_operand(), dims.output())
        else {
            return Err(unsupported("expects rank-3 flattened dims"));
        };
        if left_ab != right_ab {
            return Err(unsupported(
                "requires both operands to share the batch pack",
            ));
        }
        if left_ab != ab {
            return Err(unsupported(
                "requires the operand batch pack to match the output",
            ));
        }
        if left_k != right_k {
            return Err(unsupported(
                "requires both operands to share the contraction axis",
            ));
        }
        Ok(Self {
            log_ab: ab.log_2(),
            log_m: m.log_2(),
            log_n: n.log_2(),
            log_k: left_k.log_2(),
        })
    }
}

impl EinsumGeometry for AbmkAbnkAbmnLayout {
    fn num_rounds(&self) -> usize {
        self.log_ab + self.log_k
    }

    fn schedule(&self) -> EqSchedule {
        EqSchedule::High {
            log_eq: self.log_ab,
            low_bits: self.log_k,
        }
    }
}

impl<F: JoltField> EinsumLayout<F> for AbmkAbnkAbmnLayout {
    fn operand_points(
        &self,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
        sumcheck_challenges: &[F],
    ) -> (Vec<F>, Vec<F>) {
        let (_, r_mn) = r_node_output.split_at(self.log_ab);
        let (r_m, r_n) = r_mn.split_at(self.log_m);
        let (r_ab, r_k) = sumcheck_challenges.split_at(self.log_ab);
        let left = [r_ab, r_m.r.as_slice(), r_k].concat();
        let right = [r_ab, r_n.r.as_slice(), r_k].concat();
        (left, right)
    }

    fn output_eq(
        &self,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
        sumcheck_challenges: &[F],
    ) -> F {
        let (r_ab_output, _) = r_node_output.split_at(self.log_ab);
        let (r_ab, _) = sumcheck_challenges.split_at(self.log_ab);
        EqPolynomial::mle(&r_ab_output.r, r_ab)
    }
}
