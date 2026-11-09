//! `abmk,abnk->abmn` layout for the einsum [`dot`](super::dot) engine: a
//! batched matmul (RHS contraction-last) whose retained batch axes are packed
//! in front of both operands and the output. The rank-3 form `amk,ank->amn`
//! is the identical layout with a batch pack of `(1, a)` — the flattened
//! operand memory is the same, so both normalize here.
//!
//! The batch pack is bound first through a high eq polynomial
//! ([`EqSchedule::High`]); the contraction axis `k` follows.

pub use jolt_atlas_verifier::onnx_proof::ops::einsum::abmk_abnk_abmn::*;

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

impl<F: JoltField> EinsumFold<F> for AbmkAbnkAbmnLayout {
    fn fold(
        &self,
        left_operand: &Tensor<i32>,
        right_operand: &Tensor<i32>,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
    ) -> Folded<F> {
        let batch = 1usize << self.log_ab;
        let m = 1usize << self.log_m;
        let n = 1usize << self.log_n;
        let k = 1usize << self.log_k;

        let (r_ab, r_mn) = r_node_output.split_at(self.log_ab);
        let (r_m, r_n) = r_mn.split_at(self.log_m);
        let eq_r_ab = EqPolynomial::evals(&r_ab.r);
        let eq_r_m = EqPolynomial::evals(&r_m.r);
        let eq_r_n = EqPolynomial::evals(&r_n.r);

        let left: Vec<F> = (0..batch * k)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|hj| {
                let h = hj / k;
                let j = hj % k;
                let mut sum = F::zero();
                for i in 0..m {
                    let idx = (h * m + i) * k + j;
                    sum += eq_r_m[i].mul_i64(left_operand[idx] as i64);
                }
                sum
            })
            .collect();
        let right: Vec<F> = (0..batch * k)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|hj| {
                let h = hj / k;
                let j = hj % k;
                let mut sum = F::zero();
                for l in 0..n {
                    let idx = (h * n + l) * k + j;
                    sum += eq_r_n[l].mul_i64(right_operand[idx] as i64);
                }
                sum
            })
            .collect();
        Folded {
            left: MultilinearPolynomial::from(left),
            right: MultilinearPolynomial::from(right),
            eq: Some(MultilinearPolynomial::from(eq_r_ab)),
        }
    }
}
