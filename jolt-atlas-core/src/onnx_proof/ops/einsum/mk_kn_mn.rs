//! `mk,kn->mn` layout (standard matrix multiply) for the einsum [`dot`](super::dot) engine.

pub use jolt_atlas_verifier::onnx_proof::ops::einsum::mk_kn_mn::*;

use crate::onnx_proof::ops::einsum::dot::{EinsumFold, Folded};
use atlas_algebra::{
    field::JoltField,
    poly::{
        compact::{run_sum_unreduced, UNREDUCED_LOW_VARS},
        eq::EqPolynomial,
        multilinear::MultilinearPolynomial,
        point::{OpeningPoint, BIG_ENDIAN},
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
        let eq_r_m = EqPolynomial::evals(&r_m.r);
        let left: Vec<F> = (0..k)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|j| {
                (0..m)
                    .map(|i| eq_r_m[i].mul_i64(left_operand[i * k + j] as i64))
                    .sum()
            })
            .collect();
        // right[j] = Σ_h eq(r_n, h)·R[j, h], with eq split as eq_high ⊗ eq_low: a row is
        // read against a low table that stays in cache, each run of it summed unreduced, and
        // a run of zeros (the padding, mostly) costs nothing.
        let low_vars = r_n.r.len().min(UNREDUCED_LOW_VARS);
        let (r_high, r_low) = r_n.r.split_at(r_n.r.len() - low_vars);
        let (eq_high, eq_low) = (EqPolynomial::evals(r_high), EqPolynomial::evals(r_low));
        let right_data = right_operand.data();
        let right: Vec<F> = (0..k)
            .into_par_iter()
            .map(|j| {
                let row = &right_data[j * n..(j + 1) * n];
                row.chunks_exact(eq_low.len())
                    .zip(&eq_high)
                    .map(|(run, &high)| high * run_sum_unreduced(run, &eq_low))
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
