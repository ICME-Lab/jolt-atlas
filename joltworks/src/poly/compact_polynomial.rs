use super::multilinear_polynomial::{BindingOrder, PolynomialBinding};
use crate::{
    field::{JoltField, OptimizedMul},
    utils::{math::Math, small_scalar::SmallScalar},
};
use allocative::Allocative;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::parallel::{par_enabled, par_enabled_with};
use rayon::prelude::*;
use std::{cmp::Ordering, ops::Index};

/// Compact polynomials are used to store coefficients of small scalars.
/// They have two representations:
/// 1. `coeffs` is a vector of small scalars
/// 2. `bound_coeffs` is a vector of field elements (e.g. big scalars)
///
/// They are often initialized with `coeffs` and then converted to `bound_coeffs`
/// when binding the polynomial.
#[derive(Default, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize, Allocative)]
pub struct CompactPolynomial<T: SmallScalar, F: JoltField> {
    num_vars: usize,
    len: usize,
    pub coeffs: Vec<T>,
    pub bound_coeffs: Vec<F>,
}

impl<T: SmallScalar, F: JoltField> CompactPolynomial<T, F> {
    pub fn from_coeffs(coeffs: Vec<T>) -> Self {
        assert!(
            coeffs.len().is_power_of_two(),
            "Multilinear polynomials must be made from a power of 2 (not {})",
            coeffs.len()
        );

        CompactPolynomial {
            num_vars: coeffs.len().log_2(),
            len: coeffs.len(),
            coeffs,
            bound_coeffs: vec![],
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

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.coeffs.iter()
    }

    pub fn coeffs_as_field_elements(&self) -> Vec<F> {
        self.coeffs
            .par_iter()
            .with_min_len(par_enabled())
            .map(|x| x.to_field())
            .collect()
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
                // Rows already provide enough independent work. Keep each
                // dot product local instead of scheduling nested parallel jobs.
                let start = x1 * eq_two.len();
                let partial_sum = self.coeffs[start..start + eq_two.len()]
                    .iter()
                    .zip(eq_two)
                    .map(|(coefficient, weight)| coefficient.field_mul(*weight))
                    .fold(F::zero(), |acc, value| acc + value);
                OptimizedMul::mul_01_optimized(partial_sum, eq_one[x1])
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
                        self.coeffs[idx].field_mul(eq_two[x2])
                    })
                    .fold(F::zero(), |acc, val| acc + val);
                OptimizedMul::mul_01_optimized(partial_sum, eq_one[x1])
            })
            .fold(F::zero(), |acc, val| acc + val);
        eval
    }

    // Faster evaluation based on
    // https://randomwalks.xyz/publish/fast_polynomial_evaluation.html
    // Shaves a factor of 2 from run time.
    pub fn inside_out_evaluate(&self, r: &[F]) -> F {
        // Copied over from eq_poly
        // If the number of variables are greater
        // than 2^16 -- use parallel evaluate
        // Below that it's better to just do things linearly.
        const PARALLEL_THRESHOLD: usize = 16;
        // r must have a value for each variable
        assert_eq!(r.len(), self.get_num_vars());
        let m = r.len();
        if m < PARALLEL_THRESHOLD {
            self.inside_out_serial(r)
        } else {
            self.inside_out_parallel(r)
        }
    }

    fn inside_out_serial(&self, r: &[F]) -> F {
        // coeffs is a vector small scalars
        let mut current: Vec<F> = self.coeffs.iter().map(|&c| c.to_field()).collect();
        let m = r.len();
        for i in (0..m).rev() {
            let stride = 1 << i;
            let r_val = r[m - 1 - i];
            for j in 0..stride {
                let f0 = current[j];
                let f1 = current[j + stride];
                let slope = f1 - f0;
                if slope.is_zero() {
                    current[j] = f0;
                }
                if slope.is_one() {
                    current[j] = f0 + r_val;
                } else {
                    current[j] = f0 + slope * (r_val);
                }
            }
        }
        current[0]
    }

    fn inside_out_parallel(&self, r: &[F]) -> F {
        let mut current: Vec<F> = self
            .coeffs
            .par_iter()
            .with_min_len(par_enabled())
            .map(|&c| c.to_field())
            .collect();
        let m = r.len();
        for i in (0..m).rev() {
            let stride = 1 << i;
            let r_val = r[m - 1 - i];
            let (evals_left, evals_right) = current.split_at_mut(stride);
            let (evals_right, _) = evals_right.split_at_mut(stride);

            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter())
                .with_min_len(par_enabled())
                .for_each(|(x, y)| {
                    //*x = *x + r_val * (*y - *x);
                    let slope = *y - *x;
                    if slope.is_zero() {
                        return;
                    }
                    if slope.is_one() {
                        *x += r_val;
                    } else {
                        *x += r_val * slope;
                    }
                });
        }
        current[0]
    }
}

impl<T: SmallScalar, F: JoltField> PolynomialBinding<F> for CompactPolynomial<T, F> {
    fn is_bound(&self) -> bool {
        !self.bound_coeffs.is_empty()
    }

    #[tracing::instrument(skip_all, name = "CompactPolynomial::bind")]
    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        let n = self.len() / 2;
        if self.is_bound() {
            match order {
                BindingOrder::LowToHigh => {
                    for i in 0..n {
                        if self.bound_coeffs[2 * i + 1] == self.bound_coeffs[2 * i] {
                            self.bound_coeffs[i] = self.bound_coeffs[2 * i];
                        } else {
                            self.bound_coeffs[i] = self.bound_coeffs[2 * i]
                                + r * (self.bound_coeffs[2 * i + 1] - self.bound_coeffs[2 * i]);
                        }
                    }
                }
                BindingOrder::HighToLow => {
                    let (left, right) = self.bound_coeffs.split_at_mut(n);
                    left.iter_mut()
                        .zip(right.iter())
                        .filter(|(a, b)| a != b)
                        .for_each(|(a, b)| {
                            *a += r * (*b - *a);
                        });
                }
            }
        } else {
            // We want to compute `a * (1 - r) + b * r` where `a` and `b` are small scalars
            // If `a == b`, we can just return `a`
            // If `a < b`, we can compute `a + r * (b - a)`
            // If `a > b`, we can compute `a - r * (a - b)`
            match order {
                BindingOrder::LowToHigh => {
                    self.bound_coeffs = (0..n)
                        .map(|i| {
                            let a = self.coeffs[2 * i];
                            let b = self.coeffs[2 * i + 1];
                            match a.cmp(&b) {
                                Ordering::Equal => a.to_field(),
                                // a < b: Compute a + r * (b - a)
                                Ordering::Less => {
                                    a.to_field::<F>() + b.diff_mul_field::<F>(a, r.into())
                                }
                                // a > b: Compute a - r * (a - b)
                                Ordering::Greater => {
                                    a.to_field::<F>() - a.diff_mul_field::<F>(b, r.into())
                                }
                            }
                        })
                        .collect();
                }
                BindingOrder::HighToLow => {
                    let (left, right) = self.coeffs.split_at(n);
                    self.bound_coeffs = left
                        .iter()
                        .zip(right.iter())
                        .map(|(&a, &b)| {
                            match a.cmp(&b) {
                                Ordering::Equal => a.to_field(),
                                // a < b: Compute a + r * (b - a)
                                Ordering::Less => {
                                    a.to_field::<F>() + b.diff_mul_field::<F>(a, r.into())
                                }
                                // a > b: Compute a - r * (a - b)
                                Ordering::Greater => {
                                    a.to_field::<F>() - a.diff_mul_field::<F>(b, r.into())
                                }
                            }
                        })
                        .collect();
                }
            }
        }

        self.num_vars -= 1;
        self.len = n;
    }

    #[tracing::instrument(skip_all, name = "CompactPolynomial::bind")]
    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        let n = self.len() / 2;
        if self.is_bound() {
            match order {
                BindingOrder::LowToHigh => {
                    let mut bound_coeffs = Vec::with_capacity(n);
                    (
                        bound_coeffs.spare_capacity_mut(),
                        self.bound_coeffs.par_chunks_exact(2),
                    )
                        .into_par_iter()
                        .with_min_len(par_enabled_with(512))
                        .for_each(|(bound_coeff, coeffs)| {
                            bound_coeff.write(if coeffs[1] == coeffs[0] {
                                coeffs[0]
                            } else {
                                (coeffs[1] - coeffs[0]) * r + coeffs[0]
                            });
                        });
                    unsafe { bound_coeffs.set_len(n) };
                    self.bound_coeffs = bound_coeffs;
                }
                BindingOrder::HighToLow => {
                    let (left, right) = self.bound_coeffs.split_at_mut(n);
                    left.par_iter_mut()
                        .zip(right.par_iter())
                        .with_min_len(par_enabled_with(4096))
                        .filter(|(a, b)| a != b)
                        .for_each(|(a, b)| {
                            *a += r * (*b - *a);
                        });
                }
            }
        } else {
            match order {
                BindingOrder::LowToHigh => {
                    self.bound_coeffs = (0..n)
                        .into_par_iter()
                        .with_min_len(par_enabled())
                        .map(|i| {
                            let a = self.coeffs[2 * i];
                            let b = self.coeffs[2 * i + 1];
                            match a.cmp(&b) {
                                Ordering::Equal => a.to_field(),
                                // a < b: Compute a + r * (b - a)
                                Ordering::Less => {
                                    a.to_field::<F>() + b.diff_mul_field::<F>(a, r.into())
                                }
                                // a > b: Compute a - r * (a - b)
                                Ordering::Greater => {
                                    a.to_field::<F>() - a.diff_mul_field::<F>(b, r.into())
                                }
                            }
                        })
                        .collect();
                }
                BindingOrder::HighToLow => {
                    let (left, right) = self.coeffs.split_at(n);
                    self.bound_coeffs = left
                        .par_iter()
                        .zip(right.par_iter())
                        .with_min_len(par_enabled())
                        .map(|(&a, &b)| {
                            match a.cmp(&b) {
                                Ordering::Equal => a.to_field(),
                                // a < b: Compute a + r * (b - a)
                                Ordering::Less => {
                                    a.to_field::<F>() + b.diff_mul_field::<F>(a, r.into())
                                }
                                // a > b: Compute a - r * (a - b)
                                Ordering::Greater => {
                                    a.to_field::<F>() - a.diff_mul_field::<F>(b, r.into())
                                }
                            }
                        })
                        .collect();
                }
            }
        }
        self.num_vars -= 1;
        self.len = n;
    }

    fn final_claim(&self) -> F {
        assert_eq!(self.len, 1);
        if self.is_bound() {
            self.bound_coeffs[0]
        } else {
            self.coeffs[0].to_field()
        }
    }
}

impl<T: SmallScalar, F: JoltField> Clone for CompactPolynomial<T, F> {
    fn clone(&self) -> Self {
        Self::from_coeffs(self.coeffs.to_vec())
    }
}

impl<T: SmallScalar, F: JoltField> Index<usize> for CompactPolynomial<T, F> {
    type Output = T;

    #[inline(always)]
    fn index(&self, _index: usize) -> &T {
        &(self.coeffs[_index])
    }
}

#[cfg(test)]
mod row_evaluation_tests {
    use super::*;
    use crate::poly::eq_poly::EqPolynomial;
    use ark_bn254::Fr;
    use ark_ff::{One, Zero};

    fn check<T: SmallScalar>(pattern: &[T]) {
        for log_rows in [0usize, 7, 15, 16, 17] {
            let polynomial = CompactPolynomial::<T, Fr>::from_coeffs(
                (0..1 << log_rows)
                    .map(|i| pattern[i % pattern.len()])
                    .collect(),
            );
            for boolean in [false, true] {
                let point = (0..log_rows)
                    .map(|i| {
                        if boolean {
                            Fr::from((i % 2) as u64)
                        } else {
                            Fr::from((i * 131 + 7) as u64)
                        }
                    })
                    .collect::<Vec<_>>();
                let eq: Vec<Fr> = EqPolynomial::evals(&point);
                let expected: Fr = polynomial
                    .coeffs
                    .iter()
                    .zip(&eq)
                    .map(|(coefficient, weight)| {
                        let value: Fr = coefficient.to_field();
                        value * weight
                    })
                    .sum();
                let (high, low) = point.split_at(log_rows / 2);
                let high = EqPolynomial::evals(high);
                let low = EqPolynomial::evals(low);
                assert_eq!(
                    polynomial.split_eq_evaluate(log_rows, &high, &low),
                    expected
                );
                assert_eq!(polynomial.evaluate_split_eq_parallel(&high, &low), expected);
            }
        }
    }

    #[test]
    fn native_row_evaluation_matches_flat_field_sum_and_worker_controls() {
        for workers in [1, 2, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    check(&[false, true]);
                    check(&[0u8, 1, u8::MAX]);
                    check(&[0u16, 1, u16::MAX]);
                    check(&[0u32, 1, u32::MAX]);
                    check(&[0u64, 1, u64::MAX]);
                    check(&[i32::MIN, -1, 0, 1, i32::MAX]);
                    check(&[i64::MIN, -1, 0, 1, i64::MAX]);
                    check(&[i128::MIN, -1, 0, 1, i128::MAX]);
                    check(&[0u128, 1, u128::MAX]);
                });
        }
        let _guard = common::parallel::ParallelFlagGuard::disabled();
        check(&[i32::MIN, -1, 0, 1, i32::MAX]);
        let empty = CompactPolynomial::<i32, Fr>::from_coeffs(vec![7]);
        assert_eq!(
            empty.evaluate_split_eq_parallel(&[], &[Fr::one()]),
            Fr::zero()
        );
        assert_eq!(
            empty.evaluate_split_eq_parallel(&[Fr::one()], &[]),
            Fr::zero()
        );
    }

    fn nested(polynomial: &CompactPolynomial<i32, Fr>, high: &[Fr], low: &[Fr]) -> Fr {
        (0..high.len())
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|i| {
                let partial: Fr = (0..low.len())
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|j| polynomial.coeffs[i * low.len() + j].field_mul(low[j]))
                    .reduce(Fr::zero, |a, b| a + b);
                OptimizedMul::mul_01_optimized(partial, high[i])
            })
            .reduce(Fr::zero, |a, b| a + b)
    }

    #[test]
    #[ignore = "Isolated compact polynomial evaluation, not complete proving"]
    fn native_row_evaluation_benchmark() {
        let mode = std::env::var("NATIVE_ROW_EVALUATION_MODE").unwrap();
        assert!(mode == "nested" || mode == "rows");
        let polynomial = CompactPolynomial::<i32, Fr>::from_coeffs(
            (0..1 << 22)
                .map(|i| match i % 7 {
                    0 => 0,
                    1 => 1,
                    2 => -1,
                    _ => (i % 65536) - 32768,
                })
                .collect(),
        );
        let point = (0..22)
            .map(|i| Fr::from((i * 131 + 7) as u64))
            .collect::<Vec<_>>();
        let high = EqPolynomial::evals(&point[..11]);
        let low = EqPolynomial::evals(&point[11..]);
        let expected = nested(&polynomial, &high, &low);
        assert_eq!(polynomial.evaluate_split_eq_parallel(&high, &low), expected);
        let start = std::time::Instant::now();
        for _ in 0..8 {
            let actual = if mode == "nested" {
                nested(&polynomial, &high, &low)
            } else {
                polynomial.evaluate_split_eq_parallel(&high, &low)
            };
            assert_eq!(actual, expected);
            std::hint::black_box(actual);
        }
        println!("ROW_EVALUATION_BENCH {{\"mode\":\"{}\",\"seconds\":{},\"evaluations\":8,\"log_coefficients\":22,\"workers\":{},\"complete_proof\":false}}",mode,start.elapsed().as_secs_f64(),rayon::current_num_threads());
    }
}
