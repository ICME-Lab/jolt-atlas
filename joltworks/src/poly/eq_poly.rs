use crate::field::JoltField;
use crate::poly::opening_proof::{Endianness, OpeningPoint};
use crate::utils::{math::Math, thread::unsafe_allocate_zero_vec};
use common::parallel::par_enabled;
use rayon::prelude::*;
use std::{
    marker::PhantomData,
    ops::{Mul, Sub},
};

const PARALLEL_THRESHOLD: usize = 16;

/// Utilities for the equality polynomial `eq(x, y) = ∏ᵢ (xᵢ yᵢ + (1 - xᵢ)(1 - yᵢ))`.
///
/// The equality polynomial evaluates to 1 when `x = y` (over the boolean hypercube) and 0
/// otherwise. Its multilinear extension (MLE) is used throughout sumcheck protocols.
pub struct EqPolynomial<F: JoltField>(PhantomData<F>);

impl<F: JoltField> EqPolynomial<F> {
    /// Computes the MLE of the equality polynomial: `eq(x, y) = ∏ᵢ (xᵢ yᵢ + (1 - xᵢ)(1 - yᵢ))`.
    ///
    /// Pairs elements positionally: `x[i]` is matched with `y[i]`.
    pub fn mle<X, Y>(x: &[X], y: &[Y]) -> F
    where
        X: Copy + Send + Sync,
        Y: Copy + Send + Sync,
        F: JoltField + Sub<X, Output = F> + Sub<Y, Output = F>,
        X: Mul<Y, Output = F>,
    {
        assert_eq!(x.len(), y.len());
        x.par_iter()
            .zip(y.par_iter())
            .with_min_len(par_enabled())
            .map(|(x_i, y_i)| {
                let xy = *x_i * *y_i;
                // x*y + (1-x)*(1-y) = 1-x-y+2*x*y.
                (F::one() - *x_i - *y_i) + xy + xy
            })
            .product()
    }

    /// Computes `eq(x, y)` for [`OpeningPoint`]s, handling endianness automatically.
    ///
    /// If `x` and `y` have the **same** endianness, pairs elements positionally.
    /// If they differ, one is reversed so that MSB aligns with MSB.
    pub fn mle_endian<const E1: Endianness, const E2: Endianness>(
        x: &OpeningPoint<E1, F>,
        y: &OpeningPoint<E2, F>,
    ) -> F {
        assert_eq!(x.len(), y.len());
        if E1 == E2 {
            x.r.par_iter()
                .zip(y.r.par_iter())
                .with_min_len(par_enabled())
                .map(|(x_i, y_i)| {
                    let xy = *x_i * y_i;
                    (F::one() - x_i - y_i) + xy + xy
                })
                .product()
        } else {
            x.r.par_iter()
                .zip(y.r.par_iter().rev())
                .with_min_len(par_enabled())
                .map(|(x_i, y_i)| {
                    let xy = *x_i * y_i;
                    (F::one() - x_i - y_i) + xy + xy
                })
                .product()
        }
    }

    #[tracing::instrument(skip_all, name = "EqPolynomial::evals")]
    /// Computes the table of evaluations: `{ eq(r, x) : x ∈ {0, 1}^n }`.
    ///
    /// ### Index / bit order: Big-endian
    ///
    /// The returned vector is ordered by interpreting `x` as an `n`-bit binary number.
    /// `r[0]` corresponds to the **most-significant bit** and `r[n - 1]` to the
    /// **least-significant bit**.
    ///
    /// Concretely, if `i ∈ [0, 2^n)` has bit-decomposition `i = Σ_{j=0}^{n-1} b_j · 2^{n-1-j}`
    /// (so `b_0` is the MSB), then:
    ///
    /// `evals(r)[i] = Π_{j=0}^{n-1} ( b_j ? r[j] : (1 - r[j]) ) = eq(r, b_0…b_{n-1})`.
    ///
    /// For a scaled table, use [`EqPolynomial::evals_with_scaling`].
    pub fn evals<C>(r: &[C]) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        Self::evals_with_scaling(r, None)
    }

    /// Computes the table of evaluations: `scaling_factor · eq(r, x)` for all `x ∈ {0,1}^n`.
    ///
    /// Uses the same **big-endian** index order as [`EqPolynomial::evals`]. (See `evals` for the
    /// precise definition and bit/index mapping.)
    ///
    /// If `scaling_factor` is `None`, defaults to 1 (no scaling).
    #[inline]
    pub fn evals_with_scaling<C>(r: &[C], scaling_factor: Option<F>) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        match r.len() {
            0..=PARALLEL_THRESHOLD => Self::evals_serial(r, scaling_factor),
            _ => Self::evals_parallel(r, scaling_factor),
        }
    }

    #[tracing::instrument(skip_all, name = "EqPolynomial::evals_cached")]
    /// Computes eq evaluations like [`Self::evals`], but also caches intermediate tables.
    ///
    /// Returns `result` where `result[j]` contains evaluations for the **prefix** `r[..j]`:
    ///
    /// ```text
    /// result[j][x] = eq(r[..j], x)   for x ∈ {0,1}^j
    /// ```
    ///
    /// So `result[0] = [1]`, `result[1]` has 2 entries, …, and `result[n]` equals [`Self::evals(r)`].
    ///
    /// ### Index order
    /// Same **big-endian** convention as [`Self::evals`]: within each `result[j]`, index bit 0
    /// corresponds to `r[0]}` (MSB) and bit `j-1` to `r[j-1]` (LSB).
    pub fn evals_cached<C>(r: &[C]) -> Vec<Vec<F>>
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        // TODO: implement parallel version & determine switchover point
        Self::evals_serial_cached(r, None)
    }

    /// Like [`Self::evals_cached`], but for **high-to-low (little-endian)** binding order.
    ///
    /// Returns `result` where `result[j]` contains evaluations for the **suffix** `r[(n-j)..]`:
    ///
    /// ```text
    /// result[j][x] = eq(r[(n-j)..], x)   for x ∈ {0,1}^j
    /// ```
    ///
    /// Here, index bit 0 of `x` corresponds to `r[n-1]` (the last challenge, i.e. MSB of the
    /// suffix), and bit `j-1` to `r[n-j]` (LSB of the suffix).
    pub fn evals_cached_rev<U>(r: &[U]) -> Vec<Vec<F>>
    where
        U: Copy + Send + Sync + Into<F>,
        F: Mul<U, Output = F>,
    {
        Self::evals_serial_cached_rev(r, None)
    }

    /// Serial (single-threaded) version of [`Self::evals_with_scaling`].
    ///
    /// More efficient than the parallel version for short `r` (≤16 elements).
    /// Uses the same **big-endian** index order as [`Self::evals`].
    #[inline]
    pub fn evals_serial<C>(r: &[C], scaling_factor: Option<F>) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        let mut evals: Vec<F> = vec![scaling_factor.unwrap_or(F::one()); r.len().pow2()];
        let mut size = 1;
        for j in 0..r.len() {
            // in each iteration, we double the size of chis
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                // copy each element from the prior iteration twice
                let scalar = evals[i / 2];
                evals[i] = scalar * r[j];
                evals[i - 1] = scalar - evals[i];
            }
        }
        evals
    }

    /// Serial version of [`Self::evals_cached`] with optional scaling.
    ///
    /// Returns `result` where `result[j][x] = scaling_factor * eq(r[..j], x)` for `x ∈ {0,1}^j`.
    /// Uses the same **big-endian** index order as [`Self::evals`].
    #[inline]
    pub fn evals_serial_cached<C>(r: &[C], scaling_factor: Option<F>) -> Vec<Vec<F>>
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        let mut evals: Vec<Vec<F>> = (0..r.len() + 1)
            .map(|i| vec![scaling_factor.unwrap_or(F::one()); 1 << i])
            .collect();
        let mut size = 1;
        for j in 0..r.len() {
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let scalar = evals[j][i / 2];
                evals[j + 1][i] = scalar * r[j];
                evals[j + 1][i - 1] = scalar - evals[j + 1][i];
            }
        }
        evals
    }
    /// Serial version of [`Self::evals_cached_rev`] with optional scaling.
    ///
    /// Returns `result` where `result[j][x] = scaling_factor * eq(r[(n-j)..], x)` for `x ∈ {0,1}^j`.
    /// Uses **little-endian** (high-to-low) index order; see [`Self::evals_cached_rev`] for details.
    pub fn evals_serial_cached_rev<U>(r: &[U], scaling_factor: Option<F>) -> Vec<Vec<F>>
    where
        U: Copy + Send + Sync + Into<F>,
        F: Mul<U, Output = F>,
    {
        let rev_r = r.iter().rev().collect::<Vec<_>>();
        let mut evals: Vec<Vec<F>> = (0..r.len() + 1)
            .map(|i| vec![scaling_factor.unwrap_or(F::one()); 1 << i])
            .collect();
        let mut size = 1;
        for j in 0..r.len() {
            for i in 0..size {
                let scalar = evals[j][i];
                let multiple = 1 << j;
                evals[j + 1][i + multiple] = scalar * *rev_r[j];
                evals[j + 1][i] = scalar - evals[j + 1][i + multiple];
            }
            size *= 2;
        }
        evals
    }

    /// Parallel version of [`Self::evals_with_scaling`].
    ///
    /// Uses rayon to compute the largest layers of the DP tree in parallel.
    /// Uses the same **big-endian** index order as [`Self::evals`].
    #[tracing::instrument(skip_all, "EqPolynomial::evals_parallel")]
    #[inline]
    pub fn evals_parallel<C>(r: &[C], scaling_factor: Option<F>) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        let final_size = r.len().pow2();
        let mut evals: Vec<F> = unsafe_allocate_zero_vec(final_size);
        let mut size = 1;
        evals[0] = scaling_factor.unwrap_or(F::one());

        for r in r.iter().rev() {
            let (evals_left, evals_right) = evals.split_at_mut(size);
            let (evals_right, _) = evals_right.split_at_mut(size);

            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter_mut())
                .with_min_len(par_enabled())
                .for_each(|(x, y)| {
                    *y = *x * *r;
                    *x -= *y;
                });

            size *= 2;
        }

        evals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::{test_rng, One, Zero};
    use std::time::Instant;

    #[test]
    fn test_mle_matches_two_product_formula() {
        use crate::poly::opening_proof::{BIG_ENDIAN, LITTLE_ENDIAN};
        let mut rng = test_rng();
        for len in [0, 1, 2, 7, 16, 23, 64] {
            for _ in 0..16 {
                let x: Vec<Fr> = (0..len).map(|_| Fr::random(&mut rng)).collect();
                let y: Vec<Fr> = (0..len).map(|_| Fr::random(&mut rng)).collect();
                let c: Vec<_> = (0..len)
                    .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
                    .collect();
                let cf: Vec<Fr> = c.iter().copied().map(Into::into).collect();
                let reference = |a: &[Fr], b: &[Fr]| -> Fr {
                    a.iter()
                        .zip(b)
                        .map(|(a, b)| *a * b + (Fr::one() - a) * (Fr::one() - b))
                        .product()
                };
                assert_eq!(EqPolynomial::<Fr>::mle(&x, &y), reference(&x, &y));
                assert_eq!(EqPolynomial::<Fr>::mle(&x, &c), reference(&x, &cf));
                assert_eq!(EqPolynomial::<Fr>::mle(&c, &y), reference(&cf, &y));
                assert_eq!(EqPolynomial::<Fr>::mle(&c, &c), reference(&cf, &cf));
                let xb = OpeningPoint::<BIG_ENDIAN, Fr>::new(x.clone());
                let yb = OpeningPoint::<BIG_ENDIAN, Fr>::new(y.clone());
                let yl = OpeningPoint::<LITTLE_ENDIAN, Fr>::new(y.iter().copied().rev().collect());
                assert_eq!(EqPolynomial::<Fr>::mle_endian(&xb, &yb), reference(&x, &y));
                assert_eq!(EqPolynomial::<Fr>::mle_endian(&xb, &yl), reference(&x, &y));
                assert_eq!(EqPolynomial::<Fr>::mle_endian(&yl, &xb), reference(&y, &x));
            }
        }
        for a in [Fr::zero(), Fr::one(), -Fr::one()] {
            for b in [Fr::zero(), Fr::one(), -Fr::one()] {
                assert_eq!(
                    EqPolynomial::<Fr>::mle(&[a], &[b]),
                    a * b + (Fr::one() - a) * (Fr::one() - b)
                );
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_mle_rejects_different_lengths() {
        EqPolynomial::<Fr>::mle(&[Fr::one()], &[] as &[Fr]);
    }

    #[test]
    /// Test that the results of running `evals_serial`, `evals_parallel`, and `evals_serial_cached`
    /// (taking the last vector) are the same (and also benchmark them)
    fn test_evals() {
        let mut rng = test_rng();
        for len in 5..22 {
            let r = (0..len)
                .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
                .collect::<Vec<_>>();
            let start = Instant::now();
            let evals_serial: Vec<Fr> = EqPolynomial::evals_serial(&r, None);
            let end_first = Instant::now();
            let evals_parallel = EqPolynomial::evals_parallel(&r, None);
            let end_second = Instant::now();
            let evals_serial_cached = EqPolynomial::evals_serial_cached(&r, None);
            let end_third = Instant::now();
            println!(
                "len: {}, Time taken to compute evals_serial: {:?}",
                len,
                end_first - start
            );
            println!(
                "len: {}, Time taken to compute evals_parallel: {:?}",
                len,
                end_second - end_first
            );
            println!(
                "len: {}, Time taken to compute evals_serial_cached: {:?}",
                len,
                end_third - end_second
            );
            assert_eq!(evals_serial, evals_parallel);
            assert_eq!(evals_serial, *evals_serial_cached.last().unwrap());
        }
    }

    #[test]
    /// Test that the `i`th vector of `evals_serial_cached` is equivalent to
    /// `evals(&r[..i])`, for all `i`.
    fn test_evals_cached() {
        let mut rng = test_rng();
        for len in 2..22 {
            let r = (0..len)
                .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
                .collect::<Vec<_>>();
            let evals_serial_cached = EqPolynomial::<Fr>::evals_serial_cached(&r, None);
            for i in 0..len {
                let evals = EqPolynomial::<Fr>::evals(&r[..i]);
                assert_eq!(evals_serial_cached[i], evals);
            }
        }
    }
}
