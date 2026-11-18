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
                    super::fold::fold_pairs_in_place(
                        &mut self.bound_coeffs[..2 * n],
                        |low, high| {
                            if high == low {
                                low
                            } else {
                                (high - low) * r + low
                            }
                        },
                    );
                    self.bound_coeffs.truncate(n);
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
/// run of `L` coefficients is summed against `eq_low` without reducing
/// ([`LimbTable::sum`]) and multiplied by its `eq_high` entry once. `coeffs.len()` must be
/// `eq_high.len() * eq_low.len()`.
pub fn split_eq_sum_unreduced<T: SmallScalar, F: JoltField>(
    coeffs: &[T],
    eq_high: &[F],
    eq_low: &[F],
) -> F {
    assert_eq!(coeffs.len(), eq_high.len() * eq_low.len());
    let table = LimbTable::new(eq_low);
    coeffs
        .par_chunks(eq_low.len())
        .zip(eq_high.par_iter())
        .with_min_len(RUNS_PER_TASK)
        .map(|(run, &high)| high * table.sum(run))
        .reduce(|| F::zero(), |acc, value| acc + value)
}

/// The fewest runs a task of [`split_eq_sum_unreduced`] takes.
const RUNS_PER_TASK: usize = 16;

/// Whether every value is zero: one branch-free pass the compiler vectorizes, in blocks that
/// stop at the first block holding a value.
#[inline]
fn all_zero(values: &[i32]) -> bool {
    values
        .chunks(64)
        .all(|block| block.iter().fold(0, |acc, &v| acc | v) == 0)
}

/// `Σ_i values[i]·limbs[k][i]` in `i64` for each limb `k`: one pass over the values with
/// widening multiply-adds of `i32` lanes. Every `limbs[k]` is at least as long as `values`.
#[inline]
fn limb_dots(values: &[i32], limbs: &[Vec<i32>]) -> [i64; LIMBS] {
    debug_assert!(limbs.iter().all(|limb| limb.len() >= values.len()));
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { neon::limb_dots(values, limbs) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        std::array::from_fn(|k| {
            values
                .iter()
                .zip(&limbs[k])
                .map(|(&x, &y)| i64::from(x) * i64::from(y))
                .sum()
        })
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::LIMBS;
    use core::arch::aarch64::{
        int64x2_t, vaddvq_s64, vdupq_n_s64, vget_low_s32, vld1q_s32, vmlal_high_s32, vmlal_s32,
    };

    /// [`super::limb_dots`]: four values at a time, each limb's products accumulated in two
    /// lanes of `i64` per half of the four.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn limb_dots(values: &[i32], limbs: &[Vec<i32>]) -> [i64; LIMBS] {
        let quads = values.len() / 4;
        let limb_ptrs: [*const i32; LIMBS] = std::array::from_fn(|k| limbs[k].as_ptr());
        let mut low: [int64x2_t; LIMBS] = [vdupq_n_s64(0); LIMBS];
        let mut high: [int64x2_t; LIMBS] = [vdupq_n_s64(0); LIMBS];
        for quad in 0..quads {
            // SAFETY: `4·quad + 3 < values.len()`, and every limb is at least as long.
            unsafe {
                let v = vld1q_s32(values.as_ptr().add(4 * quad));
                let v_low = vget_low_s32(v);
                for k in 0..LIMBS {
                    let w = vld1q_s32(limb_ptrs[k].add(4 * quad));
                    low[k] = vmlal_s32(low[k], v_low, vget_low_s32(w));
                    high[k] = vmlal_high_s32(high[k], v, w);
                }
            }
        }
        std::array::from_fn(|k| {
            let tail: i64 = values[4 * quads..]
                .iter()
                .zip(&limbs[k][4 * quads..])
                .map(|(&x, &y)| i64::from(x) * i64::from(y))
                .sum();
            vaddvq_s64(low[k]) + vaddvq_s64(high[k]) + tail
        })
    }
}

/// The bits of one limb of a weight in a [`LimbTable`].
const LIMB_BITS: usize = 22;
/// The limbs of a weight: `6·22 = 132` bits hold a canonical value below `2^128`.
const LIMBS: usize = 6;

/// Field weights, each also split into [`LIMBS`] limbs of [`LIMB_BITS`] bits of its canonical
/// integer, so that a run of 32-bit values is summed against them with integer multiply-adds
/// alone ([`Self::sum`]). A table holds at most `2^UNREDUCED_LOW_VARS` weights.
pub struct LimbTable<F: JoltField> {
    weights: Vec<F>,
    /// Limb `k` of every weight, for each `k` from the least significant; empty for a field
    /// wider than 128 bits.
    limbs: Vec<Vec<i32>>,
    /// `2^(k·LIMB_BITS)` for each limb `k`.
    powers: [F; LIMBS],
}

impl<F: JoltField> LimbTable<F> {
    /// The table of `weights`. Panics if there are more than `2^UNREDUCED_LOW_VARS`.
    pub fn new(weights: &[F]) -> Self {
        assert!(weights.len() <= 1 << UNREDUCED_LOW_VARS);
        let bytes = <F as JoltField>::NUM_BYTES;
        let limbs = if bytes <= 16 {
            let values: Vec<u128> = weights
                .iter()
                .map(|weight| {
                    let mut buffer = [0u8; 16];
                    weight.to_bytes_le(&mut buffer[..bytes]);
                    u128::from_le_bytes(buffer)
                })
                .collect();
            (0..LIMBS)
                .map(|k| {
                    values
                        .iter()
                        .map(|value| {
                            ((value >> (k * LIMB_BITS)) & ((1u128 << LIMB_BITS) - 1)) as i32
                        })
                        .collect()
                })
                .collect()
        } else {
            Vec::new()
        };
        Self {
            weights: weights.to_vec(),
            limbs,
            powers: std::array::from_fn(|k| F::pow2(k * LIMB_BITS)),
        }
    }

    /// `Σ_j eq_high[j]·self.sum(row[j·L..(j+1)·L])` with `L` the table's length: a row
    /// weighted by `eq_high ⊗ weights`. `row.len()` must be `eq_high.len()·L`.
    #[inline]
    pub fn split_sum<T: SmallScalar>(&self, row: &[T], eq_high: &[F]) -> F {
        debug_assert_eq!(row.len(), eq_high.len() * self.weights.len());
        row.chunks_exact(self.weights.len())
            .zip(eq_high)
            .fold(F::zero(), |acc, (run, &high)| acc + high * self.sum(run))
    }

    /// `Σ_i weights[i]·run[i]` for scalars of at most 32 bits, reduced once. A run of values
    /// that are all `i32` ([`SmallScalar::narrow_i32`]) is summed limb by limb: each product
    /// of a value and a limb is below `2^53` in magnitude, so a run of at most
    /// `2^UNREDUCED_LOW_VARS` sums into an `i64` per limb exactly. A run of `i32` zeros (the
    /// padding of a weight matrix, mostly) costs the one pass that finds it zero. Other runs
    /// are summed into an unreduced field accumulator.
    #[inline]
    pub fn sum<T: SmallScalar>(&self, run: &[T]) -> F {
        debug_assert!(run.len() <= self.weights.len());
        if !self.limbs.is_empty() && run.first().is_some_and(|v| v.narrow_i32().is_some()) {
            let acc = match T::as_i32_slice(run) {
                Some(values) if all_zero(values) => return F::zero(),
                Some(values) => limb_dots(values, &self.limbs),
                None => {
                    let mut buffer = [0i32; 1 << UNREDUCED_LOW_VARS];
                    let values = &mut buffer[..run.len()];
                    for (slot, &value) in values.iter_mut().zip(run) {
                        *slot = value.narrow_i32().unwrap_or(0);
                    }
                    limb_dots(values, &self.limbs)
                }
            };
            if acc == [0; LIMBS] {
                return F::zero();
            }
            return acc
                .iter()
                .zip(&self.powers)
                .fold(F::zero(), |sum, (&a, &power)| sum + F::from_i64(a) * power);
        }
        let mut sum = SmallAccumS::<F>::zero();
        for (&value, weight) in run.iter().zip(&self.weights) {
            sum.fmadd(weight, &value.narrow().unwrap_or_default());
        }
        if sum.is_zero() {
            F::zero()
        } else {
            sum.barrett_reduce()
        }
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

    /// A limb table's sums are the field's, for `i32` runs (summed limb by limb, the extreme
    /// values and every limb's top bit included), for runs of zeros, for a type copied into
    /// `i32`s, for a type that is not, for a run shorter than the table, and over a row.
    #[test]
    fn limb_sums_are_the_field_sums() {
        let weights: Vec<Fp128> = (0..512u128)
            .map(|i| -Fp128::from(0x9e37_79b9_7f4a_7c15u128 * (i + 1)))
            .collect();
        let table = LimbTable::new(&weights);
        let field_sum = |values: &[i64]| -> Fp128 {
            values
                .iter()
                .zip(&weights)
                .fold(Fp128::from(0u128), |acc, (&v, &w)| {
                    acc + w * <Fp128 as crate::field::Ring>::from_i64(v)
                })
        };
        let mut signed: Vec<i32> = (0..512)
            .map(|i| ((i as i64 * 2_654_435_761) % 4_000_001 - 2_000_000) as i32)
            .collect();
        signed[..256].fill(i32::MIN);
        signed[256..300].fill(i32::MAX);
        let as_i64 = |v: &[i32]| v.iter().map(|&x| x as i64).collect::<Vec<_>>();
        assert_eq!(table.sum(&signed), field_sum(&as_i64(&signed)));
        assert_eq!(
            table.sum(&signed[..101]),
            field_sum(&as_i64(&signed[..101]))
        );
        // A run of zeros, and one whose only value is past the first zero-checked block.
        let mut sparse = vec![0i32; 512];
        assert_eq!(table.sum(&sparse), Fp128::from(0u128));
        sparse[511] = -7;
        assert_eq!(table.sum(&sparse), field_sum(&as_i64(&sparse)));
        // A row weighted by eq_high ⊗ weights.
        let eq_high = [Fp128::from(3u128), -Fp128::from(5u128)];
        let row: Vec<i32> = signed.iter().chain(&sparse).copied().collect();
        assert_eq!(
            table.split_sum(&row, &eq_high),
            eq_high[0] * field_sum(&as_i64(&signed)) + eq_high[1] * field_sum(&as_i64(&sparse))
        );
        let bytes: Vec<u8> = (0..512).map(|i| (i * 31 % 251) as u8).collect();
        let wide: Vec<i64> = bytes.iter().map(|&b| b as i64).collect();
        assert_eq!(table.sum(&bytes), field_sum(&wide));
        let words: Vec<u32> = (0..512).map(|i| u32::MAX - i).collect();
        let wide: Vec<i64> = words.iter().map(|&w| w as i64).collect();
        assert_eq!(table.sum(&words), field_sum(&wide));
    }
}
