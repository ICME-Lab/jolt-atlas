//! `mk,kn->mn` layout (standard matrix multiply) for the einsum [`dot`](super::dot) engine.

use crate::{
    onnx_proof::ops::einsum::dot::{EinsumLayout, EqSchedule, Folded},
    utils::dims::EinsumDims,
};
use atlas_onnx_tracer::tensor::Tensor;
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

impl<F: JoltField> EinsumLayout<F> for MkKnMnLayout {
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
        let (m, n, k) = (self.m, self.n, self.k);
        let (r_m, r_n) = r_node_output.split_at(m.log_2());
        let (eq_r_m, eq_r_n) = (EqPolynomial::evals(&r_m.r), EqPolynomial::evals(&r_n.r));
        let left: Vec<F> = (0..k)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|j| {
                (0..m)
                    .map(|i| F::from_i32(left_operand[i * k + j]) * eq_r_m[i])
                    .sum()
            })
            .collect();
        let right: Vec<F> = (0..k)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|j| {
                (0..n)
                    .map(|h| F::from_i32(right_operand[j * n + h]) * eq_r_n[h])
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
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        F::one()
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{model::test::ModelBuilder, model::Model, tensor::Tensor};
    use rand::{rngs::StdRng, SeedableRng};

    fn matmul_model(rng: &mut StdRng, m: usize, k: usize, n: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![m, k]);
        let c = b.constant(Tensor::random_small(rng, &[k, n]));
        let res = b.einsum("mk,kn->mn", vec![i, c], vec![m, n]);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_mk_kn_mn() {
        let m = 1 << 6;
        let k = 1 << 7;
        let n = 1 << 8;
        let mut rng = StdRng::seed_from_u64(0x878);
        let input = Tensor::<i32>::random_small(&mut rng, &[m, k]);
        let model = matmul_model(&mut rng, m, k, n);
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "TODO: non-power-of-two einsum path not fully validated yet"]
    fn test_mk_kn_mn_non_power_of_two_input_len() {
        let m = 33;
        let k = 65;
        let n = 17;
        let mut rng = StdRng::seed_from_u64(0x879);
        let input = Tensor::<i32>::random_small(&mut rng, &[m, k]);
        let model = matmul_model(&mut rng, m, k, n);
        unit_test_op(model, &[input]);
    }

    /// Build a `mk,kn->mn` model with all-`input_value` input and all-`const_value`
    /// constant, so the rebased accumulation `(k · input_value · const_value) >> S`
    /// is predictable and large enough to saturate the fused i32 clamp .
    /// The `+1` in the callers makes the remainder `R = acc mod 2^S` non-zero.
    fn saturating_model(
        input_value: i32,
        const_value: i32,
        m: usize,
        k: usize,
        n: usize,
    ) -> (Model, Tensor<i32>) {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![m, k]);
        let c = b.constant(Tensor::new(Some(&vec![const_value; k * n]), &[k, n]).unwrap());
        let res = b.einsum("mk,kn->mn", vec![i, c], vec![m, n]);
        b.mark_output(res);
        let input = Tensor::new(Some(&vec![input_value; m * k]), &[m, k]).unwrap();
        (b.build(), input)
    }

    #[test]
    fn test_mk_kn_mn_saturating_clamp_positive() {
        // acc ≈ 16·(2^20)² = 2^44, rescaled = acc >> 8 ≈ 2^36 ≫ i32::MAX → clamps
        // to i32::MAX, with a non-zero remainder.
        let big = (1 << 20) + 1;
        let (model, input) = saturating_model(big, big, 1 << 1, 1 << 4, 1 << 1);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_mk_kn_mn_saturating_clamp_negative() {
        // Opposite signs → large negative accumulation: floor-rebase (acc < 0)
        // then clamp to i32::MIN, with a non-zero Euclidean remainder.
        let big = (1 << 20) + 1;
        let (model, input) = saturating_model(-big, big, 1 << 1, 1 << 4, 1 << 1);
        unit_test_op(model, &[input]);
    }
}
