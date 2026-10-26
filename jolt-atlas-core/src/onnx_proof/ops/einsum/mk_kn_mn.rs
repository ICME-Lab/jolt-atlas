//! `mk,kn->mn` layout (standard matrix multiply) for the einsum [`dot`](super::dot) engine.

use crate::{
    onnx_proof::ops::einsum::dot::{EinsumGeometry, EinsumLayout, EqSchedule, Folded},
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

/// `mk,kn->mn`: contract the shared `k` axis of two matrices.
pub struct MkKnMnLayout {
    m: usize,
    n: usize,
    k: usize,
}

impl MkKnMnLayout {
    /// Build the layout from the canonical einsum dims.
    pub fn new(dims: &EinsumDims) -> Self {
        Self {
            m: dims.output()[0],
            n: dims.output()[1],
            k: dims.right_operand()[0],
        }
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
    fn fold(
        &self,
        left_operand: &Tensor<i32>,
        right_operand: &Tensor<i32>,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
    ) -> Folded<F> {
        let (m, n, k) = (self.m, self.n, self.k);
        let (r_m, r_n) = r_node_output.split_at(m.log_2());
        let (eq_r_m, eq_r_n) = (EqPolynomial::evals(&r_m.r), EqPolynomial::evals(&r_n.r));
        let left: Vec<F> = (0..k)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|j| {
                (0..m)
                    .map(|i| eq_r_m[i].mul_i64(left_operand[i * k + j] as i64))
                    .sum()
            })
            .collect();
        let right: Vec<F> = (0..k)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|j| {
                (0..n)
                    .map(|h| eq_r_n[h].mul_i64(right_operand[j * n + h] as i64))
                    .sum()
            })
            .collect();
        Folded {
            left: MultilinearPolynomial::from(left),
            right: MultilinearPolynomial::from(right),
            eq: None,
        }
    }

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
