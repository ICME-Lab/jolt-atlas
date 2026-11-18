use super::eq::EqPolynomial;
use super::multilinear::{BindingOrder, PolynomialBinding};
use crate::field::{BarrettReduce, FMAdd};
use crate::utils::accumulation::SmallAccumS;
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField, OptimizedMul},
    utils::{math::Math, small_scalar::SmallScalar},
};
use allocative::Allocative;
use common::parallel::{par_enabled, par_enabled_with};
use num_traits::Zero;
use rayon::prelude::*;
use std::{cmp::Ordering, ops::Index};

/// Compact polynomials are used to store coefficients of small scalars.
/// They have two representations:
/// 1. `coeffs` is a vector of small scalars
/// 2. `bound_coeffs` is a vector of field elements (e.g. big scalars)
///
/// They are often initialized with `coeffs` and then converted to `bound_coeffs`
/// when binding the polynomial.
#[derive(Default, Debug, PartialEq, Allocative)]
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
        split_eq_evaluate_coeffs(&self.coeffs, r_len, eq_one, eq_two)
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

/// Evaluates the multilinear extension of `coeffs` at `r` by the split-eq
/// method, without taking ownership of the coefficients.
///
/// `eq_one` indexes the high-order (first) variables of `r` and `eq_two` the
/// low-order remainder, so `coeffs.len()` must equal `eq_one.len() * eq_two.len()`.
pub fn split_eq_evaluate_coeffs<T: SmallScalar, F: JoltField>(
    coeffs: &[T],
    r_len: usize,
    eq_one: &[F],
    eq_two: &[F],
) -> F {
    const PARALLEL_THRESHOLD: usize = 16;
    if r_len < PARALLEL_THRESHOLD {
        split_eq_evaluate_serial(coeffs, eq_one, eq_two)
    } else {
        split_eq_evaluate_parallel(coeffs, eq_one, eq_two)
    }
}

fn split_eq_evaluate_parallel<T: SmallScalar, F: JoltField>(
    coeffs: &[T],
    eq_one: &[F],
    eq_two: &[F],
) -> F {
    let eval: F = (0..eq_one.len())
        .into_par_iter()
        .with_min_len(par_enabled())
        .map(|x1| {
            let partial_sum = (0..eq_two.len())
                .into_par_iter()
                .with_min_len(par_enabled())
                .map(|x2| {
                    let idx = x1 * eq_two.len() + x2;
                    // field_mul now already checks for 0 and 1 optimisation
                    // via Jolfield mul_64 method
                    coeffs[idx].field_mul(eq_two[x2])
                })
                .reduce(|| F::zero(), |acc, val| acc + val);
            OptimizedMul::mul_01_optimized(partial_sum, eq_one[x1])
        })
        .reduce(|| F::zero(), |acc, val| acc + val);
    eval
}

fn split_eq_evaluate_serial<T: SmallScalar, F: JoltField>(
    coeffs: &[T],
    eq_one: &[F],
    eq_two: &[F],
) -> F {
    let eval: F = (0..eq_one.len())
        .map(|x1| {
            let partial_sum = (0..eq_two.len())
                .map(|x2| {
                    let idx = x1 * eq_two.len() + x2;
                    coeffs[idx].field_mul(eq_two[x2])
                })
                .fold(F::zero(), |acc, val| acc + val);
            OptimizedMul::mul_01_optimized(partial_sum, eq_one[x1])
        })
        .fold(F::zero(), |acc, val| acc + val);
    eval
}

/// Evaluates the multilinear extension of `coeffs` at `r` without cloning the
/// coefficients into a [`MultilinearPolynomial`](super::multilinear::MultilinearPolynomial).
///
/// Equivalent to `MultilinearPolynomial::from(coeffs.to_vec()).evaluate(r)`:
/// evaluation is read-only, so borrowed coefficients suffice. `coeffs.len()`
/// must equal `1 << r.len()` (i.e. the data is already power-of-two padded).
pub fn evaluate_mle<T, F, C>(coeffs: &[T], r: &[C]) -> F
where
    T: SmallScalar,
    F: JoltField + FieldChallengeOps<C>,
    C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
{
    assert_eq!(
        coeffs.len(),
        1 << r.len(),
        "evaluate_mle: {} coefficients but {} variables",
        coeffs.len(),
        r.len()
    );
    if r.len() >= UNREDUCED_MIN_VARS && coeffs.first().is_some_and(|c| c.narrow().is_some()) {
        let (r_high, r_low) = r.split_at(r.len() - UNREDUCED_LOW_VARS);
        let (eq_high, eq_low) = rayon::join(
            || EqPolynomial::evals(r_high),
            || EqPolynomial::evals(r_low),
        );
        return split_eq_sum_unreduced(coeffs, &eq_high, &eq_low);
    }
    let m = r.len() / 2;
    let (r2, r1) = r.split_at(m);
    let (eq_one, eq_two) = rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));
    split_eq_evaluate_coeffs(coeffs, r.len(), &eq_one, &eq_two)
}

/// The variables of the low `eq` table of [`split_eq_sum_unreduced`]: 512 entries, which
/// stay in the first-level cache and whose products with 32-bit values sum unreduced.
pub const UNREDUCED_LOW_VARS: usize = 9;
/// The fewest variables [`evaluate_mle`] takes the unreduced path for.
const UNREDUCED_MIN_VARS: usize = 12;

/// `Σ_i eq_high[i / L]·eq_low[i mod L]·coeffs[i]` for scalars of at most 32 bits
/// ([`SmallScalar::narrow`]), with `L = eq_low.len()` at most `2^UNREDUCED_LOW_VARS`: each
/// run of `L` coefficients is summed against `eq_low` without reducing and multiplied by its
/// `eq_high` entry once, and a run of zeros costs its comparisons. `coeffs.len()` must be
/// `eq_high.len() * eq_low.len()`.
pub fn split_eq_sum_unreduced<T: SmallScalar, F: JoltField>(
    coeffs: &[T],
    eq_high: &[F],
    eq_low: &[F],
) -> F {
    assert_eq!(coeffs.len(), eq_high.len() * eq_low.len());
    assert!(eq_low.len() <= 1 << UNREDUCED_LOW_VARS);
    coeffs
        .par_chunks(eq_low.len())
        .zip(eq_high.par_iter())
        .with_min_len(RUNS_PER_TASK)
        .map(|(run, &high)| high * run_sum_unreduced(run, eq_low))
        .reduce(|| F::zero(), |acc, value| acc + value)
}

/// The fewest runs a task of [`split_eq_sum_unreduced`] takes.
const RUNS_PER_TASK: usize = 16;

/// `Σ_i weights[i]·run[i]` for scalars of at most 32 bits, reduced once.
#[inline]
pub fn run_sum_unreduced<T: SmallScalar, F: JoltField>(run: &[T], weights: &[F]) -> F {
    let mut sum = SmallAccumS::<F>::zero();
    for (&value, weight) in run.iter().zip(weights) {
        let value = value.narrow().unwrap_or_default();
        sum.fmadd(weight, &value);
    }
    if sum.is_zero() {
        F::zero()
    } else {
        sum.barrett_reduce()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::fp128::Fp128;

    /// The borrowed evaluation is the sum over the cube written out, on both of its paths: below and above
    /// the size the unreduced sum starts at, for signed and unsigned scalars, with runs of
    /// zeros and with the extreme values.
    #[test]
    fn evaluate_mle_is_the_polynomial_evaluation() {
        let point = |vars: usize| -> Vec<Fp128> {
            (0..vars)
                .map(|i| Fp128::from(0x9e37_79b9_7f4a_7c15u128 * (i as u128 + 3) + 1))
                .collect()
        };
        for vars in [
            5,
            UNREDUCED_MIN_VARS - 1,
            UNREDUCED_MIN_VARS,
            UNREDUCED_MIN_VARS + 3,
        ] {
            let len = 1usize << vars;
            let r = point(vars);
            let mut signed: Vec<i32> = (0..len)
                .map(|i| ((i as i64 * 2_654_435_761) % 4_000_003 - 2_000_001) as i32)
                .collect();
            signed[..len / 4].fill(0);
            signed[len / 2] = i32::MIN;
            signed[len / 2 + 1] = i32::MAX;
            let eq = EqPolynomial::<Fp128>::evals(&r);
            let want = signed
                .iter()
                .zip(&eq)
                .fold(Fp128::from(0u128), |acc, (&v, &e)| {
                    acc + e * v.to_field::<Fp128>()
                });
            assert_eq!(
                evaluate_mle::<_, Fp128, _>(&signed, &r),
                want,
                "i32, {vars}"
            );

            let bytes: Vec<u8> = (0..len).map(|i| (i * 31 % 251) as u8).collect();
            let want = bytes
                .iter()
                .zip(&eq)
                .fold(Fp128::from(0u128), |acc, (&v, &e)| {
                    acc + e * v.to_field::<Fp128>()
                });
            assert_eq!(evaluate_mle::<_, Fp128, _>(&bytes, &r), want, "u8, {vars}");

            let zeros = vec![0i32; len];
            assert_eq!(evaluate_mle::<_, Fp128, _>(&zeros, &r), Fp128::from(0u128));
        }
    }
}
