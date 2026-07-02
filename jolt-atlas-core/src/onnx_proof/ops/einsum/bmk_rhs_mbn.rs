//! `bmk,?->mbn` layouts for the einsum [`dot`](super::dot) engine.
//!
//! Both `bmk,bkn->mbn` and `bmk,kbn->mbn` contract `bmk` with a right operand that
//! differs only in the order of its batch (`b`) and contraction (`k`) axes; that
//! one difference is captured by [`BmkRhs`]. The batch eq-poly is bound *last*
//! ([`EqSchedule::Low`]). See issue #193.

use crate::{
    onnx_proof::ops::einsum::dot::{EinsumLayout, EqSchedule, Folded},
    utils::dims::{transpose_flat_matrix, EinsumDims},
};
use atlas_onnx_tracer::tensor::Tensor;
use joltworks::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{OpeningPoint, BIG_ENDIAN},
    },
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
};
use rayon::prelude::*;

/// Layout of the right operand's batch (`b`) and contraction (`k`) axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BmkRhs {
    /// Batch-major right operand: `bmk,bkn->mbn`.
    Bkn,
    /// Contraction-major right operand: `bmk,kbn->mbn`.
    Kbn,
}

/// `bmk,?->mbn`: batch matmul whose right operand is laid out per [`BmkRhs`].
pub struct BmkRhsMbnLayout {
    b: usize,
    m: usize,
    k: usize,
    n: usize,
    log_b: usize,
    log_k: usize,
    rhs: BmkRhs,
}

impl BmkRhsMbnLayout {
    /// Build the layout from the canonical einsum dims and right-operand layout.
    pub fn new(dims: &EinsumDims, rhs: BmkRhs) -> Self {
        let b = dims.left_operand()[0];
        let k = dims.left_operand()[2];
        Self {
            b,
            m: dims.left_operand()[1],
            k,
            n: dims.right_operand()[2],
            log_b: b.log_2(),
            log_k: k.log_2(),
            rhs,
        }
    }
}

impl<F: JoltField> EinsumLayout<F> for BmkRhsMbnLayout {
    fn num_rounds(&self) -> usize {
        self.log_b + self.log_k
    }

    fn schedule(&self) -> EqSchedule {
        EqSchedule::Low {
            log_k: self.log_k,
            log_b: self.log_b,
        }
    }

    fn fold(
        &self,
        left_operand: &Tensor<i32>,
        right_operand: &Tensor<i32>,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
    ) -> Folded<F> {
        let (b, m, k, n) = (self.b, self.m, self.k, self.n);
        let (r_m, r_bn) = r_node_output.split_at(m.log_2());
        let (r_b, r_n) = r_bn.split_at(b.log_2());
        let eq_r_m = EqPolynomial::evals(&r_m.r);
        let eq_r_n = EqPolynomial::evals(&r_n.r);
        let mut lo_r_m: Vec<F> = unsafe_allocate_zero_vec(k * b);
        let mut ro_r_n: Vec<F> = unsafe_allocate_zero_vec(k * b);
        lo_r_m.par_chunks_mut(k).enumerate().for_each(|(h, row)| {
            for j in 0..k {
                row[j] = (0..m)
                    .map(|i| F::from_i32(left_operand[h * (k * m) + i * (k) + j]) * eq_r_m[i])
                    .sum();
            }
        });
        ro_r_n.par_chunks_mut(b).enumerate().for_each(|(j, col)| {
            for h in 0..b {
                col[h] = (0..n)
                    .map(|l| F::from_i32(right_operand[j * (b * n) + h * (n) + l]) * eq_r_n[l])
                    .sum();
            }
        });
        let lo_r_m = transpose_flat_matrix(lo_r_m, b, k);
        // `bkn` additionally transposes the right operand's (b, k) block; `kbn`'s
        // natural row-major layout is already correct.
        let ro_r_n = match self.rhs {
            BmkRhs::Bkn => transpose_flat_matrix(ro_r_n, b, k),
            BmkRhs::Kbn => ro_r_n,
        };
        Folded {
            left: MultilinearPolynomial::from(lo_r_m),
            right: MultilinearPolynomial::from(ro_r_n),
            eq: Some(MultilinearPolynomial::from(EqPolynomial::evals(&r_b.r))),
        }
    }

    fn operand_points(
        &self,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
        sumcheck_challenges: &[F],
    ) -> (Vec<F>, Vec<F>) {
        let (r_m, r_bn) = r_node_output.split_at(self.m.log_2());
        let (_, r_n) = r_bn.split_at(self.b.log_2());
        let (r_j, r_h) = sumcheck_challenges.split_at(self.log_k);

        // Left operand is always `bmk`.
        let left = [r_h, r_m.r.as_slice(), r_j].concat();
        // Right operand axis order follows the layout: `bkn` vs `kbn`.
        let right = match self.rhs {
            BmkRhs::Bkn => [r_h, r_j, r_n.r.as_slice()].concat(),
            BmkRhs::Kbn => [r_j, r_h, r_n.r.as_slice()].concat(),
        };
        (left, right)
    }

    fn output_eq(
        &self,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let (_, r_bn) = r_node_output.split_at(self.m.log_2());
        let (r_b, _) = r_bn.split_at(self.b.log_2());
        let (_, r_h) = sumcheck_challenges.split_at(self.log_k);
        EqPolynomial::mle(&r_b.r, r_h)
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{model::test::ModelBuilder, model::Model, tensor::Tensor};
    use rand::{rngs::StdRng, SeedableRng};

    fn bmk_bkn_mbn_model(rng: &mut StdRng, b: usize, m: usize, k: usize, n: usize) -> Model {
        let mut builder = ModelBuilder::new();
        let i = builder.input(vec![b, m, k]);
        let c = builder.constant(Tensor::random_small(rng, &[b, k, n]));
        let res = builder.einsum("bmk,bkn->mbn", vec![i, c], vec![m, b, n]);
        builder.mark_output(res);
        builder.build()
    }

    fn bmk_kbn_mbn_model(rng: &mut StdRng, b: usize, m: usize, k: usize, n: usize) -> Model {
        let mut builder = ModelBuilder::new();
        let i = builder.input(vec![b, m, k]);
        let c = builder.constant(Tensor::random_small(rng, &[k, b, n]));
        let res = builder.einsum("bmk,kbn->mbn", vec![i, c], vec![m, b, n]);
        builder.mark_output(res);
        builder.build()
    }

    #[test]
    fn test_bmk_bkn_mbn() {
        let (b, m, k, n) = (1 << 3, 1 << 2, 1 << 4, 1 << 5);
        let mut rng = StdRng::seed_from_u64(0x878);
        let input = Tensor::<i32>::random_small(&mut rng, &[b, m, k]);
        let model = bmk_bkn_mbn_model(&mut rng, b, m, k, n);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_bmk_kbn_mbn() {
        let (b, m, k, n) = (1 << 3, 1 << 2, 1 << 4, 1 << 5);
        let mut rng = StdRng::seed_from_u64(0x878);
        let input = Tensor::<i32>::random_small(&mut rng, &[b, m, k]);
        let model = bmk_kbn_mbn_model(&mut rng, b, m, k, n);
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "TODO: non-power-of-two einsum path not fully validated yet"]
    fn test_bmk_bkn_mbn_non_power_of_two_input_len() {
        let (b, m, k, n) = (3, 5, 9, 7);
        let mut rng = StdRng::seed_from_u64(0x879);
        let input = Tensor::<i32>::random_small(&mut rng, &[b, m, k]);
        let model = bmk_bkn_mbn_model(&mut rng, b, m, k, n);
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "TODO: non-power-of-two einsum path not fully validated yet"]
    fn test_bmk_kbn_mbn_non_power_of_two_input_len() {
        let (b, m, k, n) = (3, 5, 9, 7);
        let mut rng = StdRng::seed_from_u64(0x879);
        let input = Tensor::<i32>::random_small(&mut rng, &[b, m, k]);
        let model = bmk_kbn_mbn_model(&mut rng, b, m, k, n);
        unit_test_op(model, &[input]);
    }
}
