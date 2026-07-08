//! `mbk,?->bmn` layouts for the einsum [`dot`](super::dot) engine.
//!
//! Both `mbk,bnk->bmn` and `mbk,nbk->bmn` contract `mbk` with a right operand that
//! differs only in the order of its batch (`b`) and free (`n`) axes; that one
//! difference is captured by [`MbkRhs`]. The batch eq-poly is bound *first*
//! ([`EqSchedule::High`]). See issue #193.

use crate::{
    onnx_proof::ops::einsum::dot::{EinsumLayout, EqSchedule, Folded},
    utils::dims::EinsumDims,
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

/// Layout of the right operand's batch (`b`) and free (`n`) axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MbkRhs {
    /// Batch-major right operand: `mbk,bnk->bmn`.
    Bnk,
    /// Free-axis-major right operand: `mbk,nbk->bmn`.
    Nbk,
}

/// `mbk,?->bmn`: batch matmul whose right operand is laid out per [`MbkRhs`].
pub struct MbkRhsBmnLayout {
    m: usize,
    b: usize,
    k: usize,
    n: usize,
    log_b: usize,
    log_k: usize,
    rhs: MbkRhs,
}

impl MbkRhsBmnLayout {
    /// Build the layout from the canonical einsum dims and right-operand layout.
    pub fn new(dims: &EinsumDims, rhs: MbkRhs) -> Self {
        let b = dims.left_operand()[1];
        let k = dims.left_operand()[2];
        // `n` sits at a different right-operand axis depending on layout.
        let n = match rhs {
            MbkRhs::Bnk => dims.right_operand()[1],
            MbkRhs::Nbk => dims.right_operand()[0],
        };
        Self {
            m: dims.left_operand()[0],
            b,
            k,
            n,
            log_b: b.log_2(),
            log_k: k.log_2(),
            rhs,
        }
    }
}

impl<F: JoltField> EinsumLayout<F> for MbkRhsBmnLayout {
    fn num_rounds(&self) -> usize {
        self.log_b + self.log_k
    }

    fn schedule(&self) -> EqSchedule {
        EqSchedule::High {
            log_eq: self.log_b,
            low_bits: self.log_k,
        }
    }

    fn fold(
        &self,
        left_operand: &Tensor<i32>,
        right_operand: &Tensor<i32>,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
    ) -> Folded<F> {
        let (m, b, k, n) = (self.m, self.b, self.k, self.n);
        let (r_b, r_mn) = r_node_output.split_at(b.log_2());
        let (r_m, r_n) = r_mn.split_at(m.log_2());
        let eq_r_m = EqPolynomial::evals(&r_m.r);
        let eq_r_n = EqPolynomial::evals(&r_n.r);
        let mut lo_r_m: Vec<F> = unsafe_allocate_zero_vec(k * b);
        let mut ro_r_n: Vec<F> = unsafe_allocate_zero_vec(k * b);
        lo_r_m.par_chunks_mut(k).enumerate().for_each(|(h, row)| {
            for j in 0..k {
                row[j] = (0..m)
                    .map(|i| F::from_i32(left_operand[i * (k * b) + h * (k) + j]) * eq_r_m[i])
                    .sum();
            }
        });
        // The flat index into the right operand differs only by axis order:
        // `bnk` -> [b, n, k], `nbk` -> [n, b, k]. Here h ranges over b, l over n, j over k.
        let rhs = self.rhs;
        ro_r_n.par_chunks_mut(k).enumerate().for_each(|(h, row)| {
            for j in 0..k {
                row[j] = (0..n)
                    .map(|l| {
                        let idx = match rhs {
                            MbkRhs::Bnk => h * (n * k) + l * (k) + j,
                            MbkRhs::Nbk => l * (k * b) + h * (k) + j,
                        };
                        F::from_i32(right_operand[idx]) * eq_r_n[l]
                    })
                    .sum();
            }
        });
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
        let (_, r_mn) = r_node_output.split_at(self.b.log_2());
        let (r_m, r_n) = r_mn.split_at(self.m.log_2());

        // Left operand is always `mbk`.
        let left = [r_m.r.as_slice(), sumcheck_challenges].concat();
        // Right operand axis order follows the layout: `bnk` vs `nbk`.
        let right = match self.rhs {
            MbkRhs::Bnk => {
                let (r_h, r_j) = sumcheck_challenges.split_at(self.log_b);
                [r_h, r_n.r.as_slice(), r_j].concat()
            }
            MbkRhs::Nbk => [r_n.r.as_slice(), sumcheck_challenges].concat(),
        };
        (left, right)
    }

    fn output_eq(
        &self,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let (r_b, _) = r_node_output.split_at(self.b.log_2());
        let (r_h, _r_j) = sumcheck_challenges.split_at(self.log_b);
        EqPolynomial::mle(&r_b.r, r_h)
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{model::test::ModelBuilder, model::Model, tensor::Tensor};
    use rand::{rngs::StdRng, SeedableRng};

    fn mbk_bnk_bmn_model(rng: &mut StdRng, m: usize, b: usize, k: usize, n: usize) -> Model {
        let mut builder = ModelBuilder::new();
        let i = builder.input(vec![m, b, k]);
        let c = builder.constant(Tensor::random_small(rng, &[b, n, k]));
        let res = builder.einsum("mbk,bnk->bmn", vec![i, c], vec![b, m, n]);
        builder.mark_output(res);
        builder.build()
    }

    fn mbk_nbk_bmn_model(rng: &mut StdRng, m: usize, b: usize, k: usize, n: usize) -> Model {
        let mut builder = ModelBuilder::new();
        let i = builder.input(vec![m, b, k]);
        let c = builder.constant(Tensor::random_small(rng, &[n, b, k]));
        let res = builder.einsum("mbk,nbk->bmn", vec![i, c], vec![b, m, n]);
        builder.mark_output(res);
        builder.build()
    }

    #[test]
    fn test_mbk_bnk_bmn() {
        let (m, b, k, n) = (1 << 2, 1 << 3, 1 << 4, 1 << 5);
        let mut rng = StdRng::seed_from_u64(0x86078);
        let input = Tensor::<i32>::random_small(&mut rng, &[m, b, k]);
        let model = mbk_bnk_bmn_model(&mut rng, m, b, k, n);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_mbk_nbk_bmn() {
        let (m, b, k, n) = (1 << 2, 1 << 3, 1 << 4, 1 << 5);
        let mut rng = StdRng::seed_from_u64(0x878);
        let input = Tensor::<i32>::random_small(&mut rng, &[m, b, k]);
        let model = mbk_nbk_bmn_model(&mut rng, m, b, k, n);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_mbk_bnk_bmn_batch_size_one() {
        let (m, b, k, n) = (1 << 2, 1, 1 << 4, 1 << 5);
        let mut rng = StdRng::seed_from_u64(0x8607a);
        let input = Tensor::<i32>::random_small(&mut rng, &[m, b, k]);
        let model = mbk_bnk_bmn_model(&mut rng, m, b, k, n);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_mbk_nbk_bmn_batch_size_one() {
        let (m, b, k, n) = (1 << 2, 1, 1 << 4, 1 << 5);
        let mut rng = StdRng::seed_from_u64(0x87a);
        let input = Tensor::<i32>::random_small(&mut rng, &[m, b, k]);
        let model = mbk_nbk_bmn_model(&mut rng, m, b, k, n);
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "TODO: non-power-of-two einsum path not fully validated yet"]
    fn test_mbk_bnk_bmn_non_power_of_two_input_len() {
        let (m, b, k, n) = (5, 3, 9, 7);
        let mut rng = StdRng::seed_from_u64(0x86079);
        let input = Tensor::<i32>::random_small(&mut rng, &[m, b, k]);
        let model = mbk_bnk_bmn_model(&mut rng, m, b, k, n);
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "TODO: non-power-of-two einsum path not fully validated yet"]
    fn test_mbk_nbk_bmn_non_power_of_two_input_len() {
        let (m, b, k, n) = (5, 3, 9, 7);
        let mut rng = StdRng::seed_from_u64(0x879);
        let input = Tensor::<i32>::random_small(&mut rng, &[m, b, k]);
        let model = mbk_nbk_bmn_model(&mut rng, m, b, k, n);
        unit_test_op(model, &[input]);
    }
}
