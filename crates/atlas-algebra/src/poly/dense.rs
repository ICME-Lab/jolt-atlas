#![allow(clippy::too_many_arguments)]
#![allow(clippy::uninlined_format_args)]
use crate::{
    field::ChallengeFieldOps,
    poly::{eq::EqPolynomial, multilinear::PolynomialEvaluation},
    utils::compute_dotproduct,
};
use common::parallel::{par_enabled, par_enabled_with};

use crate::{
    field::{FieldChallengeOps, JoltField, OptimizedMul},
    utils::math::Math,
};
use allocative::Allocative;
use core::ops::Index;
use rand_core::{CryptoRng, RngCore};
use rayon::prelude::*;
use std::ops::IndexMut;

use super::multilinear::BindingOrder;

#[derive(Default, Debug, PartialEq, Allocative)]
pub struct DensePolynomial<F: JoltField> {
    pub num_vars: usize, // the number of variables in the multilinear polynomial
    pub len: usize,
    pub Z: Vec<F>, // evaluations of the polynomial in all the 2^num_vars Boolean inputs
}

impl<F: JoltField> DensePolynomial<F> {
    pub fn new(Z: Vec<F>) -> Self {
        assert!(
            Z.len().is_power_of_two(),
            "Dense multi-linear polynomials must be made from a power of 2 (not {})",
            Z.len()
        );

        DensePolynomial {
            num_vars: Z.len().log_2(),
            len: Z.len(),
            Z,
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_bound(&self) -> bool {
        self.len != self.Z.len()
    }

    pub fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        match order {
            BindingOrder::LowToHigh => self.bound_poly_var_bot(&r),
            BindingOrder::HighToLow => self.bound_poly_var_top(&r),
        }
    }

    pub fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        match order {
            BindingOrder::LowToHigh => self.bound_poly_var_bot_01_optimized(&r),
            BindingOrder::HighToLow => self.bound_poly_var_top_zero_optimized(&r),
        }
    }

    pub fn bound_poly_var_top(&mut self, r: &F::Challenge) {
        let n = self.len() / 2;
        let (left, right) = self.Z.split_at_mut(n);

        left.iter_mut().zip(right.iter()).for_each(|(a, b)| {
            *a += *r * (*b - *a);
        });

        self.num_vars -= 1;
        self.len = n;
    }

    /// Bounds the polynomial's most significant index bit to 'r' optimized for a
    /// high P(eval = 0).
    pub fn bound_poly_var_top_zero_optimized(&mut self, r: &F::Challenge) {
        let n = self.len() / 2;

        let (left, right) = self.Z.split_at_mut(n);

        left.par_iter_mut()
            .zip(right.par_iter())
            .with_min_len(par_enabled_with(4096))
            .filter(|(&mut a, &b)| a != b)
            .for_each(|(a, b)| {
                *a += *r * (*b - *a);
            });

        self.num_vars -= 1;
        self.len = n;
    }

    /// Note: does not truncate
    pub fn bound_poly_var_bot(&mut self, r: &F::Challenge) {
        let n = self.len() / 2;
        for i in 0..n {
            self.Z[i] = self.Z[2 * i] + *r * (self.Z[2 * i + 1] - self.Z[2 * i]);
        }

        self.num_vars -= 1;
        self.len = n;
    }

    pub fn bound_poly_var_bot_01_optimized(&mut self, r: &F::Challenge) {
        let n = self.len() / 2;
        super::fold::fold_pairs_in_place(&mut self.Z[..2 * n], |low, high| {
            let m = high - low;
            if m.is_zero() {
                low
            } else if m.is_one() {
                low + *r
            } else {
                low + *r * m
            }
        });
        self.Z.truncate(n);
        self.num_vars -= 1;
        self.len = n;
    }

    pub fn evaluate_dot_product<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        // r must have a value for each variable
        assert_eq!(r.len(), self.get_num_vars());
        let chis = EqPolynomial::evals(r);
        assert_eq!(chis.len(), self.Z.len());
        self.evaluate_at_chi(&chis)
    }

    // returns Z(r) in O(n) time
    pub fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let m = r.len() / 2;
        let (r2, r1) = r.split_at(m);
        let (eq_one, eq_two) = rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));
        self.split_eq_evaluate(r.len(), &eq_one, &eq_two)
    }

    pub fn split_eq_evaluate(&self, r_len: usize, eq_one: &[F], eq_two: &[F]) -> F {
        const PARALLEL_THRESHOLD: usize = 16;
        if r_len < PARALLEL_THRESHOLD {
            self.evaluate_split_eq_serial(eq_one, eq_two)
        } else {
            self.evaluate_split_eq_parallel(eq_one, eq_two)
        }
    }
    fn evaluate_split_eq_parallel(&self, eq_one: &[F], eq_two: &[F]) -> F {
        let eval: F = (0..eq_one.len())
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|x1| {
                let partial_sum = (0..eq_two.len())
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|x2| {
                        let idx = x1 * eq_two.len() + x2;
                        OptimizedMul::mul_01_optimized(eq_two[x2], self.Z[idx])
                    })
                    .reduce(|| F::zero(), |acc, val| acc + val);
                OptimizedMul::mul_01_optimized(eq_one[x1], partial_sum)
            })
            .reduce(|| F::zero(), |acc, val| acc + val);
        eval
    }

    fn evaluate_split_eq_serial(&self, eq_one: &[F], eq_two: &[F]) -> F {
        let eval: F = (0..eq_one.len())
            .map(|x1| {
                let partial_sum = (0..eq_two.len())
                    .map(|x2| {
                        let idx = x1 * eq_two.len() + x2;
                        OptimizedMul::mul_01_optimized(eq_two[x2], self.Z[idx])
                    })
                    .fold(F::zero(), |acc, val| acc + val);
                OptimizedMul::mul_01_optimized(eq_one[x1], partial_sum)
            })
            .fold(F::zero(), |acc, val| acc + val);
        eval
    }

    pub fn evaluate_at_chi(&self, chis: &[F]) -> F {
        compute_dotproduct(&self.Z, chis)
    }

    pub fn evals(&self) -> Vec<F> {
        self.Z.clone()
    }

    #[tracing::instrument(skip_all, name = "DensePolynomial::from")]
    pub fn from_u64(Z: &[u64]) -> Self {
        DensePolynomial::new((0..Z.len()).map(|i| F::from_u64(Z[i])).collect::<Vec<F>>())
    }

    pub fn random<R: RngCore + CryptoRng>(num_vars: usize, mut rng: &mut R) -> Self {
        Self::new(
            std::iter::from_fn(|| Some(F::random(&mut rng)))
                .take(1 << num_vars)
                .collect(),
        )
    }
}

impl<F: JoltField> Clone for DensePolynomial<F> {
    fn clone(&self) -> Self {
        Self::new(self.Z[0..self.len].to_vec())
    }
}

impl<F: JoltField> Index<usize> for DensePolynomial<F> {
    type Output = F;

    #[inline(always)]
    fn index(&self, _index: usize) -> &F {
        &(self.Z[_index])
    }
}

impl<F: JoltField> IndexMut<usize> for DensePolynomial<F> {
    #[inline(always)]
    fn index_mut(&mut self, _index: usize) -> &mut F {
        &mut (self.Z[_index])
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for DensePolynomial<F> {
    fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        self.evaluate(r)
    }

    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        debug_assert!(degree > 0);
        debug_assert!(index < self.len() / 2);

        let mut evals = vec![F::zero(); degree];
        match order {
            BindingOrder::HighToLow => {
                evals[0] = self[index];
                if degree == 1 {
                    return evals;
                }
                let mut eval = self[index + self.len() / 2];
                let m = eval - evals[0];
                for i in 1..degree {
                    eval += m;
                    evals[i] = eval;
                }
            }
            BindingOrder::LowToHigh => {
                evals[0] = self[2 * index];
                if degree == 1 {
                    return evals;
                }
                let mut eval = self[2 * index + 1];
                let m = eval - evals[0];
                for i in 1..degree {
                    eval += m;
                    evals[i] = eval;
                }
            }
        };
        evals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::fp128::Fp128 as Fr;
    use crate::utils::test_rng;

    use rand::Rng;

    pub fn compute_chis_at_r<F: JoltField>(r: &[F::Challenge]) -> Vec<F> {
        let ell = r.len();
        let n = ell.pow2();
        let mut chis: Vec<F> = Vec::new();
        for i in 0..n {
            let mut chi_i = F::one();
            for j in 0..r.len() {
                let bit_j = (i & (1 << (r.len() - j - 1))) > 0;
                if bit_j {
                    chi_i *= r[j].into();
                } else {
                    chi_i *= F::one() - r[j];
                }
            }
            chis.push(chi_i);
        }
        chis
    }

    #[test]
    fn check_memoized_chis() {
        check_memoized_chis_helper::<Fr>()
    }

    fn check_memoized_chis_helper<F: JoltField>() {
        let mut prng = test_rng();

        let s = 10;
        let mut r: Vec<F::Challenge> = Vec::new();
        for _i in 0..s {
            r.push(F::Challenge::from(prng.gen::<u128>()));
        }
        let chis = compute_chis_at_r::<F>(&r);
        let chis_m = EqPolynomial::<F>::evals(&r);
        assert_eq!(chis, chis_m);
    }

    #[test]
    fn evaluation() {
        let num_evals = 4;
        let mut evals: Vec<Fr> = Vec::with_capacity(num_evals);
        for _ in 0..num_evals {
            evals.push(Fr::from(8));
        }
        let dense_poly: DensePolynomial<Fr> = DensePolynomial::new(evals.clone());

        // Evaluate at 3:
        // (0, 0) = 1
        // (0, 1) = 1
        // (1, 0) = 1
        // (1, 1) = 1
        // g(x_0,x_1) => c_0*(1 - x_0)(1 - x_1) + c_1*(1-x_0)(x_1) + c_2*(x_0)(1-x_1) + c_3*(x_0)(x_1)
        // g(3, 4) = 8*(1 - 3)(1 - 4) + 8*(1-3)(4) + 8*(3)(1-4) + 8*(3)(4) = 48 + -64 + -72 + 96  = 8
        // g(5, 10) = 8*(1 - 5)(1 - 10) + 8*(1 - 5)(10) + 8*(5)(1-10) + 8*(5)(10) = 96 + -16 + -72 + 96  = 8
        assert_eq!(
            dense_poly.evaluate(vec![Fr::from(3), Fr::from(4)].as_slice()),
            Fr::from(8)
        );
    }
}
