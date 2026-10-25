//! `k,nk->n` layout (matrix-vector multiply) for the einsum [`dot`](super::dot) engine.

use crate::{
    onnx_proof::ops::einsum::dot::{EinsumLayout, EqSchedule, Folded},
    utils::dims::EinsumDims,
};
use atlas_trace::Tensor;
use common::parallel::par_enabled;
use joltworks::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{OpeningPoint, BIG_ENDIAN},
    },
    utils::math::Math,
};
use rayon::prelude::*;

/// `k,nk->n`: contract the `k` axis of a vector with a matrix. The left operand
/// is the raw vector (no folding); the right operand is folded over the output `n`.
pub struct KNkNLayout {
    n: usize,
    k: usize,
}

impl KNkNLayout {
    /// Build the layout from the canonical einsum dims.
    pub fn new(dims: &EinsumDims) -> Self {
        Self {
            n: dims.output()[0],
            k: dims.left_operand()[0],
        }
    }
}

impl<F: JoltField> EinsumLayout<F> for KNkNLayout {
    fn num_rounds(&self) -> usize {
        self.k.log_2()
    }

    fn schedule(&self) -> EqSchedule {
        EqSchedule::None
    }

    fn fold(
        &self,
        left_operand: &Tensor<i32>,
        right_operand: &Tensor<i32>,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
    ) -> Folded<F> {
        let (n, k) = (self.n, self.k);
        let eq_r_node_output = EqPolynomial::evals(&r_node_output.r);
        let right: Vec<F> = (0..k)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|j| {
                (0..n)
                    .map(|h| eq_r_node_output[h].mul_i64(right_operand[h * k + j] as i64))
                    .sum()
            })
            .collect();
        Folded {
            left: MultilinearPolynomial::from(left_operand.clone()),
            right: MultilinearPolynomial::from(right),
            eq: None,
        }
    }

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
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        F::one()
    }
}
