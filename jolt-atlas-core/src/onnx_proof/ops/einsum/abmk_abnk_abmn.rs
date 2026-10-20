//! `abmk,abnk->abmn` layout for the einsum [`dot`](super::dot) engine: a
//! batched matmul (RHS contraction-last) whose retained batch axes are packed
//! in front of both operands and the output. The rank-3 form `amk,ank->amn`
//! is the identical layout with a batch pack of `(1, a)` — the flattened
//! operand memory is the same, so both normalize here.
//!
//! The batch pack is bound first through a high eq polynomial
//! ([`EqSchedule::High`]); the contraction axis `k` follows.

use crate::compat::tensor::Tensor;
use crate::{
    onnx_proof::ops::einsum::dot::{EinsumLayout, EqSchedule, Folded},
    utils::dims::EinsumDims,
};
use common::parallel::par_enabled;
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{OpeningPoint, BIG_ENDIAN},
    },
    utils::math::Math,
};
use rayon::prelude::*;

/// Batched matmul layout: `out[h,i,l] = Σ_j left[h,i,j]·right[h,l,j]` with the
/// batch pack `h` flattened into the leading axis by the dims extractor.
pub struct AbmkAbnkAbmnLayout {
    log_ab: usize,
    log_m: usize,
    log_n: usize,
    log_k: usize,
}

impl AbmkAbnkAbmnLayout {
    /// Build the layout from the canonical einsum dims. The extractors hand us
    /// `left = [ab, m, k]`, `right = [ab, n, k]`, `output = [ab, m, n]`.
    pub fn new(dims: &EinsumDims) -> Self {
        assert_eq!(
            dims.left_operand()[0],
            dims.right_operand()[0],
            "abmk,abnk->abmn requires both operands to share the batch pack"
        );
        assert_eq!(
            dims.left_operand()[0],
            dims.output()[0],
            "abmk,abnk->abmn requires the operand batch pack to match the output"
        );
        assert_eq!(
            dims.left_operand()[2],
            dims.right_operand()[2],
            "abmk,abnk->abmn requires both operands to share the contraction axis"
        );
        Self {
            log_ab: dims.output()[0].log_2(),
            log_m: dims.output()[1].log_2(),
            log_n: dims.output()[2].log_2(),
            log_k: dims.left_operand()[2].log_2(),
        }
    }
}

impl<F: JoltField> EinsumLayout<F> for AbmkAbnkAbmnLayout {
    fn num_rounds(&self) -> usize {
        self.log_ab + self.log_k
    }

    fn schedule(&self) -> EqSchedule {
        EqSchedule::High {
            log_eq: self.log_ab,
            low_bits: self.log_k,
        }
    }

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

    fn operand_points(
        &self,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
        sumcheck_challenges: &[F],
    ) -> (Vec<F>, Vec<F>) {
        let (_, r_mn) = r_node_output.split_at(self.log_ab);
        let (r_m, r_n) = r_mn.split_at(self.log_m);
        let (r_ab, r_k) = sumcheck_challenges.split_at(self.log_ab);
        let left = [r_ab, r_m.r.as_slice(), r_k].concat();
        let right = [r_ab, r_n.r.as_slice(), r_k].concat();
        (left, right)
    }

    fn output_eq(
        &self,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let (r_ab_output, _) = r_node_output.split_at(self.log_ab);
        let sumcheck_opening = sumcheck_challenges.into_opening();
        let (r_ab, _) = sumcheck_opening.split_at(self.log_ab);
        EqPolynomial::mle(&r_ab_output.r, r_ab)
    }
}

#[cfg(test)]
mod tests {
    use crate::compat::{model::test::ModelBuilder, model::Model, tensor::Tensor};
    use crate::onnx_proof::ops::test::unit_test_op;
    use rand::{rngs::StdRng, SeedableRng};

    fn abmk_abnk_abmn_model(
        rng: &mut StdRng,
        a: usize,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Model {
        let mut builder = ModelBuilder::new();
        let left = builder.input(vec![a, b, m, k]);
        let right = builder.constant(Tensor::random_small(rng, &[a, b, n, k]));
        let out = builder.einsum("abmk,abnk->abmn", vec![left, right], vec![a, b, m, n]);
        builder.mark_output(out);
        builder.build()
    }

    fn amk_ank_amn_model(rng: &mut StdRng, a: usize, m: usize, k: usize, n: usize) -> Model {
        let mut builder = ModelBuilder::new();
        let left = builder.input(vec![a, m, k]);
        let right = builder.constant(Tensor::random_small(rng, &[a, n, k]));
        let out = builder.einsum("amk,ank->amn", vec![left, right], vec![a, m, n]);
        builder.mark_output(out);
        builder.build()
    }

    #[test]
    fn test_abmk_abnk_abmn() {
        let mut rng = StdRng::seed_from_u64(0);
        let model = abmk_abnk_abmn_model(&mut rng, 2, 2, 4, 4, 8);
        let input = Tensor::<i32>::random_small(&mut rng, &[2, 2, 4, 4]);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_amk_ank_amn() {
        let mut rng = StdRng::seed_from_u64(2);
        let model = amk_ank_amn_model(&mut rng, 4, 4, 8, 8);
        let input = Tensor::<i32>::random_small(&mut rng, &[4, 4, 8]);
        unit_test_op(model, &[input]);
    }

    /// The shape Qwen's repeat_kv lowering emits: batch 1 and a contraction
    /// axis of length 1 (an outer product), i.e. zero eq-pack and zero k
    /// sumcheck variables.
    #[test]
    fn test_amk_ank_amn_qwen_shape() {
        let mut rng = StdRng::seed_from_u64(3);
        let model = amk_ank_amn_model(&mut rng, 1, 32, 1, 16);
        let input = Tensor::<i32>::random_small(&mut rng, &[1, 32, 1]);
        unit_test_op(model, &[input]);
    }
}
