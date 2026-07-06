//! `k,nk->n` layout (matrix-vector multiply) for the einsum [`dot`](super::dot) engine.

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
                    .map(|h| F::from_i32(right_operand[h * k + j]) * eq_r_node_output[h])
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

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{model::test::ModelBuilder, model::Model, tensor::Tensor};
    use rand::{rngs::StdRng, SeedableRng};

    fn matvec_model(rng: &mut StdRng, k: usize, n: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![k]);
        let c = b.constant(Tensor::random_small(rng, &[n, k]));
        let res = b.einsum("k,nk->n", vec![i, c], vec![n]);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_k_nk_n() {
        let k = 1 << 7;
        let n = 1 << 8;
        let mut rng = StdRng::seed_from_u64(0x878);
        let input = Tensor::<i32>::random_small(&mut rng, &[k]);
        let model = matvec_model(&mut rng, k, n);
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "TODO: non-power-of-two einsum path not fully validated yet"]
    fn test_k_nk_n_non_power_of_two_input_len() {
        let k = 65;
        let n = 33;
        let mut rng = StdRng::seed_from_u64(0x879);
        let input = Tensor::<i32>::random_small(&mut rng, &[k]);
        let model = matvec_model(&mut rng, k, n);
        unit_test_op(model, &[input]);
    }
}
