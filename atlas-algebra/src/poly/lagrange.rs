use crate::field::JoltField;
use std::marker::PhantomData;
use std::ops::{Mul, Sub};

/// Lagrange polynomials over zero-centered, symmetric, consecutive-integer domain, i.e.
/// grids like [-6, -5, ..., 6, 7].
/// This is the high-degree univariate analogue of EqPolynomial, which are for multilinear polynomials.
/// We use this in the univariate skip optimization in Spartan's outer sum-check.
pub struct LagrangePolynomial<F: JoltField>(PhantomData<F>);

impl<F: JoltField> LagrangePolynomial<F> {
    /// Univariate Lagrange kernel on the symmetric integer grid.
    /// Computes K(x, y) = Σ_i L_i(x) · L_i(y), where {L_i} are Lagrange basis
    /// polynomials for nodes start..start+N-1 with start = -floor((N-1)/2).
    /// Returns 1 if x and y coincide at the same node, else 0 at nodes; otherwise
    /// the barycentric kernel value. Constraint: N <= 20.
    pub fn lagrange_kernel<C, const N: usize>(x: &C, y: &C) -> F
    where
        C: Copy + Send + Sync + Sub<F, Output = F>,
        F: Mul<C, Output = F>,
    {
        debug_assert!(N > 0, "N must be positive");
        debug_assert!(N <= 20, "lagrange_kernel intended for small N (<= 20)");
        // Grid nodes are consecutive integers centered at 0:
        // x_i = start + i where start = -floor((N-1)/2)
        let d = N - 1;
        let start: i64 = -((d / 2) as i64);

        // Distances to nodes for x and y; detect on-node early exits
        let mut dists_x = [F::zero(); N];
        let mut dists_y = [F::zero(); N];
        let mut base_x = *x - F::from_i64(start);
        let mut base_y = *y - F::from_i64(start);
        let one = F::one();
        let mut ix: Option<usize> = None;
        let mut it: Option<usize> = None;
        let mut i: usize = 0;
        while i < N {
            let dx = base_x;
            let dy = base_y;
            if dx == F::zero() {
                ix = Some(i);
            }
            if dy == F::zero() {
                it = Some(i);
            }
            dists_x[i] = dx;
            dists_y[i] = dy;
            base_x -= one;
            base_y -= one;
            i += 1;
        }

        // If both points are at nodes, equality is Kronecker delta
        if let (Some(ix), Some(it)) = (ix, it) {
            return if ix == it { F::one() } else { F::zero() };
        }

        // Precompute inverse denominators (shared helper)
        let inv_denom = Self::inv_denom::<N>();

        // If exactly one is at a node, say x at node i, then result is L_i(y)
        if let Some(ix) = ix {
            // Compute L_i(y) via barycentric terms
            let (terms_y, sum_y) = Self::bary_terms_from_dists::<N>(&dists_y, &inv_denom);
            return terms_y[ix] * sum_y.inverse().unwrap();
        }

        if let Some(it) = it {
            // Symmetric: result is L_it(x)
            // Symmetric: compute L_it(x)
            let (terms_x, sum_x) = Self::bary_terms_from_dists::<N>(&dists_x, &inv_denom);
            return terms_x[it] * sum_x.inverse().unwrap();
        }

        // General case: fused kernel
        // Use shared barycentric terms for x and y, then combine
        let (terms_x, s_x) = Self::bary_terms_from_dists::<N>(&dists_x, &inv_denom);
        let (terms_y, s_y) = Self::bary_terms_from_dists::<N>(&dists_y, &inv_denom);
        let mut num = F::zero();
        let mut i = 0usize;
        while i < N {
            num += terms_x[i] * terms_y[i];
            i += 1;
        }

        // eq(x,y) = num / (Sx * Sy)
        let inv_den = (s_x * s_y).inverse().unwrap();
        num * inv_den
    }

    /// Start of the symmetric integer grid.
    /// Formula: start = −⌊(N−1)/2⌋.
    #[inline]
    fn start_i64<const N: usize>() -> i64 {
        let d = N - 1;
        -((d / 2) as i64)
    }

    /// Distances to grid nodes.
    /// Grid nodes: xᵢ = start + i. Returns ([dᵢ], hit) where dᵢ = r − xᵢ and
    /// hit = Some(i) if dᵢ = 0.
    #[inline]
    fn distances<C, const N: usize>(r: &C) -> ([F; N], Option<usize>)
    where
        C: Copy + Sub<F, Output = F>,
    {
        let start = Self::start_i64::<N>();
        let mut dists = [F::zero(); N];
        let mut base = *r - F::from_i64(start);
        let one = F::one();
        let mut hit: Option<usize> = None;
        let mut i: usize = 0;
        while i < N {
            let di = base;
            if di == F::zero() {
                hit = Some(i);
            }
            dists[i] = di;
            base -= one;
            i += 1;
        }
        (dists, hit)
    }

    /// Inverse denominators for barycentric weights.
    /// Weights: wᵢ = (−1)^(N−1−i) / (i! · (N−1−i)!). Returns [wᵢ].
    #[inline]
    fn inv_denom<const N: usize>() -> [F; N] {
        let den_i64 = LagrangeHelper::den_row_i64::<N>();
        let mut denom = [F::zero(); N];
        let mut i = 0usize;
        while i < N {
            denom[i] = F::from_i64(den_i64[i]);
            i += 1;
        }
        let mut left = [F::one(); N];
        i = 1;
        while i < N {
            left[i] = left[i - 1] * denom[i - 1];
            i += 1;
        }
        let inv_total = (left[N - 1] * denom[N - 1]).inverse().unwrap();
        let mut inv_denom = [F::zero(); N];
        let mut right = F::one();
        let mut t: isize = (N as isize) - 1;
        while t >= 0 {
            let u = t as usize;
            inv_denom[u] = left[u] * right * inv_total;
            right *= denom[u];
            t -= 1;
        }
        inv_denom
    }

    /// Unnormalized barycentric terms and their sum.
    /// Given [dᵢ] and [wᵢ]: termᵢ = wᵢ / dᵢ, S = Σᵢ termᵢ.
    /// Normalized basis: Lᵢ(r) = termᵢ / S.
    #[inline]
    fn bary_terms_from_dists<const N: usize>(dists: &[F; N], inv_denom: &[F; N]) -> ([F; N], F) {
        // prefix/suffix and total inverse product for 1/(r - x_i)
        let mut prefix = [F::one(); N];
        let mut i = 1usize;
        while i < N {
            prefix[i] = prefix[i - 1] * dists[i - 1];
            i += 1;
        }
        let inv_prod = (prefix[N - 1] * dists[N - 1]).inverse().unwrap();

        let mut suffix = [F::one(); N];
        let mut j: isize = (N as isize) - 2;
        while j >= 0 {
            let u = j as usize;
            suffix[u] = suffix[u + 1] * dists[u + 1];
            j -= 1;
        }

        let mut terms = [F::zero(); N];
        let mut sum = F::zero();
        i = 0;
        while i < N {
            let inv_di = prefix[i] * suffix[i] * inv_prod;
            let term = inv_denom[i] * inv_di;
            terms[i] = term;
            sum += term;
            i += 1;
        }
        (terms, sum)
    }

    /// Evaluate p(r) from values on the symmetric grid using barycentric Lagrange.
    /// Nodes: xᵢ = start + i, with start = −⌊(N−1)/2⌋. Weights: wᵢ = (−1)^(N−1−i)/(i!·(N−1−i)!).
    /// Basis: Lᵢ(r) = (wᵢ/(r−xᵢ)) / Σⱼ (wⱼ/(r−xⱼ)).
    /// Value: p(r) = Σᵢ Lᵢ(r)·values[i]. If r = x_k, p(r) = values[k].
    /// Uses prefix/suffix products and batch inversion (≈3 field inversions total).
    #[inline]
    pub fn evaluate<C, const N: usize>(values: &[F; N], r: &C) -> F
    where
        C: Copy + Send + Sync + Sub<F, Output = F>,
        F: Mul<C, Output = F>,
    {
        debug_assert!(N > 0, "N must be positive");
        debug_assert!(N <= 20, "evaluate intended for small N (<= 20)");
        let (dists, hit) = Self::distances::<C, N>(r);
        if let Some(i) = hit {
            return values[i];
        }
        let inv_denom = Self::inv_denom::<N>();
        let (terms, sum) = Self::bary_terms_from_dists::<N>(&dists, &inv_denom);
        let inv_sum = sum.inverse().unwrap();
        let mut num = F::zero();
        let mut i = 0usize;
        while i < N {
            num += values[i] * terms[i];
            i += 1;
        }
        num * inv_sum
    }

    /// Compute all Lagrange basis polynomial values `[L_i(r)]` at point `r` for symmetric grid of size `N`.
    /// Grid nodes are `{start, start+1, ..., start+N-1}` where `start = -floor((N-1)/2)`.
    /// Returns: `[L_0(r), L_1(r), ..., L_{N-1}(r)]` such that `p(r) = sum_i L_i(r) * p(x_i)`.
    ///
    /// **Constraint**: N <= 20 (all we need for now)
    pub fn evals<C, const N: usize>(r: &C) -> [F; N]
    where
        C: Copy + Send + Sync + Sub<F, Output = F>,
        F: Mul<C, Output = F>,
    {
        debug_assert!(
            N <= 20,
            "N cannot be greater than 20 for current implementation"
        );
        debug_assert!(N > 0, "N must be positive");
        let (dists, hit) = Self::distances::<C, N>(r);
        if let Some(i) = hit {
            let mut out = [F::zero(); N];
            out[i] = F::one();
            return out;
        }
        let inv_denom = Self::inv_denom::<N>();
        let (terms, sum) = Self::bary_terms_from_dists::<N>(&dists, &inv_denom);
        let inv_sum = sum.inverse().unwrap();
        let mut out = [F::zero(); N];
        let mut i = 0usize;
        while i < N {
            out[i] = terms[i] * inv_sum;
            i += 1;
        }
        out
    }
}

/// A collection of helper and utility functions for Lagrange interpolation,
/// binomial coefficients, and polynomial extrapolation over integer domains.
pub struct LagrangeHelper;

impl LagrangeHelper {
    /// Factorial in u64 for small n (valid up to n = 20).
    #[inline]
    pub const fn fact(n: usize) -> u64 {
        let mut acc: u64 = 1;
        let mut i: usize = 2;
        while i <= n {
            acc *= i as u64;
            i += 1;
        }
        acc
    }

    /// Precomputed `[0!, 1!, ..., 20!]` as u64. All entries fit in u64.
    pub const FACT_U64_0_TO_20: [u64; 21] = {
        let mut out = [0u64; 21];
        let mut i: usize = 0;
        while i <= 20 {
            out[i] = Self::fact(i);
            i += 1;
        }
        out
    };

    /// Returns `[den[0], ..., den[N-1]]` as i64 where den[i] = (-1)^{N-1-i} * i! * (N-1-i)!
    /// Constraint: N <= 20
    #[inline]
    pub const fn den_row_i64<const N: usize>() -> [i64; N] {
        let mut out = [0i64; N];
        let mut i: usize = 0;
        while i < N {
            let a = Self::FACT_U64_0_TO_20[i] as i128;
            let b = Self::FACT_U64_0_TO_20[N - 1 - i] as i128;
            let mut v = a * b;
            if ((N - 1 - i) & 1) == 1 {
                v = -v;
            }
            out[i] = v as i64;
            i += 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{LagrangeHelper, LagrangePolynomial};
    use crate::field::fp128::Fp128 as F;
    use jolt_field::{Field, Ring};

    fn grid_nodes<const N: usize>() -> [F; N] {
        let d = N - 1;
        let start: i64 = -((d / 2) as i64);
        core::array::from_fn(|i| F::from_i64(start + i as i64))
    }

    fn pow_u64(mut base: F, mut exp: u64) -> F {
        if exp == 0 {
            return F::from_u64(1);
        }
        let mut acc = F::from_u64(1);
        while exp > 0 {
            if (exp & 1) == 1 {
                acc *= base;
            }
            base = base * base;
            exp >>= 1;
        }
        acc
    }

    #[test]
    fn closed_form_n1() {
        const N: usize = 1;
        let values = [F::from_u64(7)];
        for k in 0..5u64 {
            let r = F::from_u64(k);
            let l = LagrangePolynomial::<F>::evals::<F, N>(&r);
            assert_eq!(l[0], F::from_u64(1));
            let v = LagrangePolynomial::<F>::evaluate::<F, N>(&values, &r);
            assert_eq!(v, values[0]);
        }
    }

    #[test]
    fn closed_form_n2() {
        // grid {0,1}: L0(r)=1-r, L1(r)=r
        const N: usize = 2;
        let nodes = grid_nodes::<N>();
        for k in 0..7u64 {
            let r = F::from_u64(k);
            let [l0, l1] = LagrangePolynomial::<F>::evals::<F, N>(&r);
            let l0_cf = F::from_u64(1) - r;
            let l1_cf = r;
            assert_eq!(l0, l0_cf);
            assert_eq!(l1, l1_cf);
            // early exit at nodes
            let vals = [F::from_u64(3), F::from_u64(5)];
            for i in 0..N {
                let ri = nodes[i];
                let v = LagrangePolynomial::<F>::evaluate::<F, N>(&vals, &ri);
                assert_eq!(v, vals[i]);
            }
        }
    }

    #[test]
    fn closed_form_n3() {
        // grid {-1,0,1}: L0(r)=r(r-1)/2, L1(r)=1-r^2, L2(r)=r(r+1)/2
        const N: usize = 3;
        let nodes = grid_nodes::<N>();
        let two_inv = F::from_u64(2).inverse().unwrap();
        for k in 0..7u64 {
            let r = F::from_u64(k) - F::from_u64(1); // covers negatives around nodes
            let [l0, l1, l2] = LagrangePolynomial::<F>::evals::<F, N>(&r);
            let l0_cf = (r * (r - F::from_u64(1))) * two_inv;
            let l1_cf = F::from_u64(1) - r * r;
            let l2_cf = (r * (r + F::from_u64(1))) * two_inv;
            assert_eq!(l0, l0_cf);
            assert_eq!(l1, l1_cf);
            assert_eq!(l2, l2_cf);

            // early exit at nodes
            let vals = [F::from_u64(11), F::from_u64(13), F::from_u64(17)];
            for i in 0..N {
                let ri = nodes[i];
                let v = LagrangePolynomial::<F>::evaluate::<F, N>(&vals, &ri);
                assert_eq!(v, vals[i]);
            }
        }
    }

    #[test]
    fn basis_properties_and_monomials() {
        // N set
        let ns = [1usize, 2, 3, 4, 5, 8, 11, 20];
        for &n in &ns {
            // partition of unity and delta at nodes
            let nodes: Vec<F> = {
                let d = n - 1;
                let start: i64 = -((d / 2) as i64);
                (0..n).map(|i| F::from_i64(start + i as i64)).collect()
            };

            // integers in [-3..3]
            for t in -3..=3 {
                let r = F::from_i64(t);
                let basis = match n {
                    1 => LagrangePolynomial::<F>::evals::<F, 1>(&r).to_vec(),
                    2 => LagrangePolynomial::<F>::evals::<F, 2>(&r).to_vec(),
                    3 => LagrangePolynomial::<F>::evals::<F, 3>(&r).to_vec(),
                    4 => LagrangePolynomial::<F>::evals::<F, 4>(&r).to_vec(),
                    5 => LagrangePolynomial::<F>::evals::<F, 5>(&r).to_vec(),
                    8 => LagrangePolynomial::<F>::evals::<F, 8>(&r).to_vec(),
                    11 => LagrangePolynomial::<F>::evals::<F, 11>(&r).to_vec(),
                    20 => LagrangePolynomial::<F>::evals::<F, 20>(&r).to_vec(),
                    _ => unreachable!(),
                };
                let sum: F = basis.iter().copied().sum();
                assert_eq!(sum, F::from_u64(1));
            }

            // delta at nodes
            for (i, &xi) in nodes.iter().enumerate() {
                let basis = match n {
                    1 => LagrangePolynomial::<F>::evals::<F, 1>(&xi).to_vec(),
                    2 => LagrangePolynomial::<F>::evals::<F, 2>(&xi).to_vec(),
                    3 => LagrangePolynomial::<F>::evals::<F, 3>(&xi).to_vec(),
                    4 => LagrangePolynomial::<F>::evals::<F, 4>(&xi).to_vec(),
                    5 => LagrangePolynomial::<F>::evals::<F, 5>(&xi).to_vec(),
                    8 => LagrangePolynomial::<F>::evals::<F, 8>(&xi).to_vec(),
                    11 => LagrangePolynomial::<F>::evals::<F, 11>(&xi).to_vec(),
                    20 => LagrangePolynomial::<F>::evals::<F, 20>(&xi).to_vec(),
                    _ => unreachable!(),
                };
                for (j, &bj) in basis.iter().enumerate() {
                    if i == j {
                        assert_eq!(bj, F::from_u64(1));
                    } else {
                        assert_eq!(bj, F::from_u64(0));
                    }
                }
            }

            // monomial reproduction
            for m in 0..n {
                let d = n - 1;
                let start: i64 = -((d / 2) as i64);
                let mut vals_vec = Vec::with_capacity(n);
                for i in 0..n {
                    let xi = F::from_i64(start + i as i64);
                    vals_vec.push(pow_u64(xi, m as u64));
                }
                for t in -2..=2 {
                    let r = F::from_i64(t);
                    let lhs = match n {
                        1 => LagrangePolynomial::<F>::evaluate::<F, 1>(
                            &vals_vec.clone().try_into().unwrap(),
                            &r,
                        ),
                        2 => LagrangePolynomial::<F>::evaluate::<F, 2>(
                            &vals_vec.clone().try_into().unwrap(),
                            &r,
                        ),
                        3 => LagrangePolynomial::<F>::evaluate::<F, 3>(
                            &vals_vec.clone().try_into().unwrap(),
                            &r,
                        ),
                        4 => LagrangePolynomial::<F>::evaluate::<F, 4>(
                            &vals_vec.clone().try_into().unwrap(),
                            &r,
                        ),
                        5 => LagrangePolynomial::<F>::evaluate::<F, 5>(
                            &vals_vec.clone().try_into().unwrap(),
                            &r,
                        ),
                        8 => LagrangePolynomial::<F>::evaluate::<F, 8>(
                            &vals_vec.clone().try_into().unwrap(),
                            &r,
                        ),
                        11 => LagrangePolynomial::<F>::evaluate::<F, 11>(
                            &vals_vec.clone().try_into().unwrap(),
                            &r,
                        ),
                        20 => LagrangePolynomial::<F>::evaluate::<F, 20>(
                            &vals_vec.clone().try_into().unwrap(),
                            &r,
                        ),
                        _ => unreachable!(),
                    };
                    let rhs = pow_u64(r, m as u64);
                    assert_eq!(lhs, rhs);
                }
            }
        }
    }

    #[test]
    fn lagrange_kernel_matches_evals_and_nodes() {
        fn check_kernel_for<const N: usize>() {
            for r_int in -2..=2 {
                for s_int in -2..=2 {
                    let r = F::from_i64(r_int);
                    let s = F::from_i64(s_int);
                    let k = LagrangePolynomial::<F>::lagrange_kernel::<F, N>(&r, &s);
                    let br = LagrangePolynomial::<F>::evals::<F, N>(&r);
                    let bs = LagrangePolynomial::<F>::evals::<F, N>(&s);
                    let mut dot = F::from_u64(0);
                    for i in 0..N {
                        dot += br[i] * bs[i];
                    }
                    assert_eq!(k, dot);
                    let ks = LagrangePolynomial::<F>::lagrange_kernel::<F, N>(&s, &r);
                    assert_eq!(k, ks);
                }
            }
            let nodes = grid_nodes::<N>();
            for i in 0..N {
                for j in 0..N {
                    let k = LagrangePolynomial::<F>::lagrange_kernel::<F, N>(&nodes[i], &nodes[j]);
                    if i == j {
                        assert_eq!(k, F::from_u64(1));
                    } else {
                        assert_eq!(k, F::from_u64(0));
                    }
                }
            }
        }

        check_kernel_for::<1>();
        check_kernel_for::<2>();
        check_kernel_for::<3>();
        check_kernel_for::<8>();
        check_kernel_for::<11>();
        check_kernel_for::<20>();
    }

    #[test]
    fn factorial_and_denominator_row() {
        let fact = |n: usize| -> u64 { (1..=n as u64).product::<u64>().max(1) };
        for n in 0..=12usize {
            assert_eq!(LagrangeHelper::fact(n), fact(n));
        }

        // den_row_i64 checks for fixed sizes
        fn check_den_row<const N: usize>() {
            let den = LagrangeHelper::den_row_i64::<N>();
            for i in 0..N {
                let sign = if ((N - 1 - i) & 1) == 1 { -1i64 } else { 1i64 };
                let expected = (LagrangeHelper::fact(i) as i128
                    * LagrangeHelper::fact(N - 1 - i) as i128
                    * sign as i128) as i64;
                assert_eq!(den[i], expected);
            }
        }
        check_den_row::<1>();
        check_den_row::<2>();
        check_den_row::<3>();
        check_den_row::<5>();
        check_den_row::<8>();
        check_den_row::<11>();
        check_den_row::<20>();
    }
}
