//! `mk,kn->mn` layout (standard matrix multiply) for the einsum [`dot`](super::dot) engine.

pub use jolt_atlas_verifier::onnx_proof::ops::einsum::mk_kn_mn::*;

use crate::onnx_proof::ops::einsum::dot::{EinsumFold, Folded};
use atlas_algebra::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::MultilinearPolynomial,
        opening_point::{OpeningPoint, BIG_ENDIAN},
    },
    utils::math::Math,
};
use atlas_trace::Tensor;
use common::parallel::par_enabled;
use rayon::prelude::*;

impl<F: JoltField> EinsumFold<F> for MkKnMnLayout {
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
}
