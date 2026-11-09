//! `k,nk->n` layout (matrix-vector multiply) for the einsum [`dot`](super::dot) engine.

pub use jolt_atlas_verifier::onnx_proof::ops::einsum::k_nk_n::*;

use crate::onnx_proof::ops::einsum::dot::{EinsumFold, Folded};
use atlas_algebra::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::MultilinearPolynomial,
        opening_point::{OpeningPoint, BIG_ENDIAN},
    },
};
use atlas_trace::Tensor;
use common::parallel::par_enabled;
use rayon::prelude::*;

impl<F: JoltField> EinsumFold<F> for KNkNLayout {
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
}
