//! Univariate polynomials in coefficient form: [`UnivariatePoly`], its compressed round-message
//! form [`CompressedPoly`] (linear term omitted, recovered from the sumcheck hint), and
//! [`OmittedConstantPoly`]. Sumcheck proofs carry [`CompressedPoly`]s and evaluation reduction
//! carries a [`UnivariatePoly`], both through the `CanonicalSerialize` impls below.
//!
//! The types are bounded by upstream's `jolt_field::Field` because the external `akita-*` crates
//! bind them through the `jolt-poly` compatibility package.

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use allocative::Allocative;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::utils::serialization::deserialize_vec;

/// Shared interface for univariate polynomial types.
///
/// Provides the minimal common API between [`UnivariatePoly`] (full coefficient
/// form) and [`CompressedPoly`](CompressedPoly) (linear term omitted). Evaluation and coefficient
/// access are deliberately left as inherent methods because the two representations
/// require different calling conventions (compressed evaluation needs an external
/// hint value).
pub trait UnivariatePolynomial<F: Field>: Send + Sync {
    /// Degree of the polynomial, or 0 for the zero polynomial.
    fn degree(&self) -> usize;
}

/// Univariate polynomial in coefficient form: $p(x) = \sum_{i=0}^{d} c_i x^i$.
///
/// Coefficients are stored in ascending degree order: `coefficients[i]` is the
/// coefficient of $x^i$. An empty coefficient vector represents the zero polynomial.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Allocative)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct UnivariatePoly<F: Field> {
    coefficients: Vec<F>,
}

impl<F: Field> UnivariatePolynomial<F> for UnivariatePoly<F> {
    fn degree(&self) -> usize {
        if self.coefficients.is_empty() {
            0
        } else {
            self.coefficients.len() - 1
        }
    }
}

impl<F: Field> UnivariatePoly<F> {
    /// Creates a polynomial from coefficients in ascending degree order.
    pub fn new(coefficients: Vec<F>) -> Self {
        Self { coefficients }
    }

    pub fn zero() -> Self {
        Self {
            coefficients: Vec::new(),
        }
    }

    /// Evaluates $p(\text{point})$ using Horner's method.
    ///
    /// Computes $c_d + x(c_{d-1} + x(c_{d-2} + \cdots))$ in $O(d)$ multiplications.
    #[inline]
    #[expect(clippy::unwrap_used)]
    pub fn evaluate(&self, point: F) -> F {
        if self.coefficients.is_empty() {
            return F::zero();
        }
        self.coefficients
            .iter()
            .rev()
            .copied()
            .reduce(|acc, c| acc * point + c)
            .unwrap()
    }

    /// Lagrange interpolation from a set of $(x_i, y_i)$ pairs.
    ///
    /// Given $n$ distinct points, produces the unique polynomial of degree $\le n-1$
    /// passing through all of them using the standard $O(n^2)$ algorithm.
    ///
    /// # Panics
    /// Panics if `points` is empty.
    #[expect(clippy::expect_used)]
    pub fn interpolate(points: &[(F, F)]) -> Self {
        assert!(!points.is_empty(), "cannot interpolate zero points");

        let n = points.len();
        let mut result = vec![F::zero(); n];

        for j in 0..n {
            let mut basis = vec![F::zero(); n];
            basis[0] = F::one();

            let mut basis_len = 1;
            for m in 0..n {
                if m == j {
                    continue;
                }
                let denom = (points[j].0 - points[m].0)
                    .inverse()
                    .expect("interpolation points must be distinct");
                let neg_xm = -points[m].0;

                for k in (1..=basis_len).rev() {
                    basis[k] = basis[k - 1] + basis[k] * neg_xm;
                }
                basis[0] *= neg_xm;
                basis_len += 1;

                for coeff in basis.iter_mut().take(basis_len) {
                    *coeff *= denom;
                }
            }

            for k in 0..n {
                result[k] += points[j].1 * basis[k];
            }
        }

        Self {
            coefficients: result,
        }
    }

    /// Coefficients in ascending degree order: index $i$ holds the coefficient of $x^i$.
    pub fn coefficients(&self) -> &[F] {
        &self.coefficients
    }

    pub fn into_coefficients(self) -> Vec<F> {
        self.coefficients
    }

    /// Removes trailing zero coefficients while retaining one coefficient for
    /// a nonempty zero polynomial. An empty polynomial stays empty.
    ///
    /// Interpolation deliberately preserves its input width; callers that need
    /// the shortest coefficient form can request it explicitly with this method.
    pub fn trim_trailing_zeros(&mut self) {
        while self.coefficients.len() > 1 && self.coefficients.last().is_some_and(|c| c.is_zero()) {
            let _ = self.coefficients.pop();
        }
    }

    /// Interpolates a polynomial from evaluations at $\{0, 1, \ldots, n-1\}$.
    ///
    /// Given evaluations $[f(0), f(1), \ldots, f(n-1)]$, returns the unique polynomial
    /// of degree $\le n-1$ matching those values. This is a convenience wrapper around
    /// [`interpolate`](Self::interpolate) for the common integer-domain case.
    ///
    /// # Panics
    /// Panics if `evals` is empty.
    pub fn interpolate_over_integers(evals: &[F]) -> Self {
        assert!(!evals.is_empty(), "cannot interpolate zero evaluations");
        let points: Vec<(F, F)> = evals
            .iter()
            .enumerate()
            .map(|(i, &y)| (F::from_u64(i as u64), y))
            .collect();
        Self::interpolate(&points)
    }

    /// Compresses the polynomial by omitting the linear term.
    ///
    /// The resulting [`CompressedPoly`](CompressedPoly) stores
    /// `[c0, c2, c3, ...]`, saving one field element in proof serialization.
    /// The linear term can be recovered given the hint value `f(0) + f(1)`.
    ///
    /// An empty polynomial stores `[0]`; a constant stores `[c0]`. Both
    /// compressed forms have degree bound one because the hint can reconstruct
    /// a nonzero linear term.
    pub fn compress(&self) -> CompressedPoly<F> {
        let Some(&constant) = self.coefficients.first() else {
            return CompressedPoly::new(vec![F::zero()]);
        };
        let mut coeffs = Vec::with_capacity(self.coefficients.len().saturating_sub(1).max(1));
        coeffs.push(constant);
        if self.coefficients.len() > 2 {
            coeffs.extend_from_slice(&self.coefficients[2..]);
        }
        CompressedPoly::new(coeffs)
    }

    /// Interpolates from evaluations at `0, 1, 2, ..., n-1` using Newton forward
    /// differences in O(n²) field operations. The returned coefficient count is
    /// exactly `evals.len()`, including any trailing zero coefficients. Empty
    /// evaluations produce the empty zero polynomial.
    ///
    /// # Panics
    /// Panics if a required factorial has no inverse in the field (for example,
    /// if the evaluation domain exceeds the field characteristic).
    #[expect(clippy::expect_used)]
    pub fn from_evals(evals: &[F]) -> Self {
        let n = evals.len();
        if n == 0 {
            return Self::zero();
        }

        let mut table = evals.to_vec();
        let mut differences = Vec::with_capacity(n);
        for width in (1..=n).rev() {
            differences.push(table[0]);
            for j in 0..width - 1 {
                table[j] = table[j + 1] - table[j];
            }
        }

        let mut factorial = F::one();
        for (k, difference) in differences.iter_mut().enumerate().skip(1) {
            factorial *= F::from_u64(k as u64);
            *difference *= factorial
                .inverse()
                .expect("field characteristic too small for interpolation");
        }

        let mut coefficients = vec![differences[n - 1]];
        for k in (0..n - 1).rev() {
            let shift = F::from_u64(k as u64);
            let mut expanded = vec![F::zero(); coefficients.len() + 1];
            expanded[0] = differences[k];
            for (i, &coefficient) in coefficients.iter().enumerate() {
                expanded[i] -= shift * coefficient;
                expanded[i + 1] += coefficient;
            }
            coefficients = expanded;
        }
        Self { coefficients }
    }

    /// Interpolates from evaluations at `[0, 2, 3, ..., n-1]` with the hint `p(0) + p(1)`.
    ///
    /// Recovers `p(1) = hint - p(0)` and then interpolates over the full set `{0, 1, ..., n-1}`.
    ///
    /// # Panics
    /// Panics if `evals` is empty.
    pub fn from_evals_and_hint(hint: F, evals: &[F]) -> Self {
        assert!(!evals.is_empty(), "cannot interpolate zero evaluations");
        let mut full = evals.to_vec();
        let eval_at_1 = hint - full[0];
        full.insert(1, eval_at_1);
        Self::from_evals(&full)
    }

    /// Interpolates from evaluations at `[0, 1, ..., degree-1, ∞]`.
    ///
    /// The last entry is interpreted as the evaluation at infinity, i.e., the leading
    /// coefficient of the polynomial. Subtracts the leading monomial and
    /// interpolates the remainder with Newton forward differences.
    ///
    /// # Panics
    /// Panics if `evals` is empty.
    pub fn from_evals_toom(evals: &[F]) -> Self {
        let n = evals.len();
        assert!(n > 0, "cannot interpolate zero evaluations");
        let degree = n - 1;
        let leading = evals[degree];
        let finite: Vec<F> = evals[..degree]
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let point = F::from_u64(index as u64);
                let high_power = (0..degree).fold(F::one(), |power, _| power * point);
                *value - leading * high_power
            })
            .collect();
        let mut coefficients = Self::from_evals(&finite).into_coefficients();
        coefficients.push(leading);
        Self { coefficients }
    }

    /// Returns `true` if all coefficients are zero (or the vector is empty).
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty() || self.coefficients.iter().all(|c| *c == F::zero())
    }
}

impl<F: Field> Neg for UnivariatePoly<F> {
    type Output = Self;

    fn neg(mut self) -> Self {
        for c in &mut self.coefficients {
            *c = -*c;
        }
        self
    }
}

impl<F: Field> Add for UnivariatePoly<F> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        self += &rhs;
        self
    }
}

impl<F: Field> Add for &UnivariatePoly<F> {
    type Output = UnivariatePoly<F>;

    fn add(self, rhs: Self) -> UnivariatePoly<F> {
        let (longer, shorter) = if self.coefficients.len() >= rhs.coefficients.len() {
            (&self.coefficients, &rhs.coefficients)
        } else {
            (&rhs.coefficients, &self.coefficients)
        };
        let mut coeffs = longer.clone();
        for (a, b) in coeffs.iter_mut().zip(shorter) {
            *a += *b;
        }
        UnivariatePoly::new(coeffs)
    }
}

impl<F: Field> AddAssign<&Self> for UnivariatePoly<F> {
    fn add_assign(&mut self, rhs: &Self) {
        if rhs.coefficients.len() > self.coefficients.len() {
            self.coefficients.resize(rhs.coefficients.len(), F::zero());
        }
        for (a, b) in self.coefficients.iter_mut().zip(&rhs.coefficients) {
            *a += *b;
        }
    }
}

impl<F: Field> Sub for UnivariatePoly<F> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self {
        self -= &rhs;
        self
    }
}

impl<F: Field> Sub for &UnivariatePoly<F> {
    type Output = UnivariatePoly<F>;

    fn sub(self, rhs: Self) -> UnivariatePoly<F> {
        let max_len = self.coefficients.len().max(rhs.coefficients.len());
        let mut coeffs = vec![F::zero(); max_len];
        for (i, c) in self.coefficients.iter().enumerate() {
            coeffs[i] += *c;
        }
        for (i, c) in rhs.coefficients.iter().enumerate() {
            coeffs[i] -= *c;
        }
        UnivariatePoly::new(coeffs)
    }
}

impl<F: Field> SubAssign<&Self> for UnivariatePoly<F> {
    fn sub_assign(&mut self, rhs: &Self) {
        if rhs.coefficients.len() > self.coefficients.len() {
            self.coefficients.resize(rhs.coefficients.len(), F::zero());
        }
        for (a, b) in self.coefficients.iter_mut().zip(&rhs.coefficients) {
            *a -= *b;
        }
    }
}

impl<F: Field> Mul<F> for UnivariatePoly<F> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self {
        self *= rhs;
        self
    }
}

impl<F: Field> Mul<F> for &UnivariatePoly<F> {
    type Output = UnivariatePoly<F>;

    fn mul(self, rhs: F) -> UnivariatePoly<F> {
        UnivariatePoly::new(self.coefficients.iter().map(|c| *c * rhs).collect())
    }
}

impl<F: Field> MulAssign<F> for UnivariatePoly<F> {
    fn mul_assign(&mut self, rhs: F) {
        for c in &mut self.coefficients {
            *c *= rhs;
        }
    }
}

/// Compressed univariate polynomial: stores `[c0, c2, c3, ...]` with the
/// linear coefficient `c1` omitted.
///
/// Given the hint `h = f(0) + f(1)`, the linear term is recovered as:
/// `c1 = h - 2*c0 - c2 - c3 - ...`
///
/// This saves one field element per sumcheck round polynomial in proof
/// serialization (32 bytes for BN254).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct CompressedPoly<F: Field> {
    coeffs_except_linear_term: Vec<F>,
}

impl<F: Field> UnivariatePolynomial<F> for CompressedPoly<F> {
    /// Degree of the polynomial.
    ///
    /// A degree-d polynomial has d+1 coefficients; the compressed form stores
    /// d of them (all except the linear term), so `stored_len == degree`.
    fn degree(&self) -> usize {
        self.coeffs_except_linear_term.len()
    }
}

impl<F: Field> CompressedPoly<F> {
    /// Creates a compressed polynomial from the stored coefficients `[c0, c2, c3, ...]`.
    pub fn new(coeffs_except_linear_term: Vec<F>) -> Self {
        Self {
            coeffs_except_linear_term,
        }
    }

    /// The stored coefficients `[c0, c2, c3, ...]` (linear term omitted).
    pub fn coeffs_except_linear_term(&self) -> &[F] {
        &self.coeffs_except_linear_term
    }

    pub fn is_empty(&self) -> bool {
        self.coeffs_except_linear_term.is_empty()
    }

    /// Recovers the omitted linear term from the hint `h = f(0) + f(1)`.
    ///
    /// `c1 = h - 2*c0 - c2 - c3 - ...`
    #[inline]
    fn recover_linear_term(&self, hint: F) -> F {
        // Deserialized proofs can carry an empty coefficient vector; fail with
        // a clear contract violation instead of an index panic. Callers on
        // untrusted data must reject empty polynomials first (`is_empty`).
        assert!(
            !self.coeffs_except_linear_term.is_empty(),
            "cannot evaluate an empty compressed polynomial"
        );
        let c0 = self.coeffs_except_linear_term[0];
        let mut linear_term = hint - c0 - c0;
        for &c in &self.coeffs_except_linear_term[1..] {
            linear_term -= c;
        }
        linear_term
    }

    /// Evaluates the polynomial at `point` using the hint `h = f(0) + f(1)`.
    ///
    /// Recovers the linear term, then evaluates via ascending-power accumulation
    /// in O(d) multiplications.
    ///
    /// # Panics
    /// Panics if the polynomial is empty ([`is_empty`](Self::is_empty)).
    #[inline]
    pub fn evaluate_with_hint(&self, hint: F, point: F) -> F {
        self.eval_from_hint(&hint, &point)
    }

    /// Like [`evaluate_with_hint`](Self::evaluate_with_hint) but takes hint and
    /// point by reference, avoiding a copy at call sites that already hold references.
    ///
    /// # Panics
    /// Panics if the polynomial is empty ([`is_empty`](Self::is_empty)).
    #[inline]
    pub fn eval_from_hint(&self, hint: &F, point: &F) -> F {
        let linear_term = self.recover_linear_term(*hint);

        let mut x_pow = *point;
        let mut sum = self.coeffs_except_linear_term[0] + *point * linear_term;
        for &c in &self.coeffs_except_linear_term[1..] {
            x_pow *= *point;
            sum += c * x_pow;
        }
        sum
    }
}

/// Stores `[q1, q2, ..., qd]` for an inner polynomial
/// `q(X) = q0 + q1 X + ... + qd X^d`.
///
/// The constant coefficient is supplied by the protocol using this polynomial.
/// Stored trailing zeros are retained because the payload length can determine
/// the expected round-message shape. An empty payload represents an inner
/// constant polynomial.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct OmittedConstantPoly<F: Field> {
    coefficients: Vec<F>,
}

impl<F: Field> OmittedConstantPoly<F> {
    /// Constructs from the stored nonconstant coefficients `[q1, ..., qd]`.
    pub fn new(coefficients: Vec<F>) -> Self {
        Self { coefficients }
    }

    /// Constructs from the full inner polynomial coefficients `[q0, ..., qd]`.
    pub fn from_q_coefficients(coefficients: Vec<F>) -> Self {
        Self::new(coefficients.into_iter().skip(1).collect())
    }

    /// Returns the stored nonconstant coefficients, including trailing zeros.
    pub fn coefficients(&self) -> &[F] {
        &self.coefficients
    }

    pub fn into_coefficients(self) -> Vec<F> {
        self.coefficients
    }

    /// Returns the stored degree bound of the inner polynomial.
    pub fn degree(&self) -> usize {
        self.coefficients.len()
    }

    /// Returns `q(1) - q0`.
    pub fn nonconstant_term_sum_at_one(&self) -> F {
        self.coefficients.iter().copied().sum()
    }

    /// Returns `q(point) - q0` using Horner's method.
    pub fn evaluate_nonconstant_terms(&self, point: F) -> F {
        self.coefficients
            .iter()
            .rev()
            .copied()
            .fold(F::zero(), |acc, coefficient| acc * point + coefficient)
            * point
    }
}

// Proof wire form: the coefficient vector as an arkworks `Vec`. Its length is prover-supplied, so
// it decodes through `deserialize_vec`.

impl<F: Field + CanonicalSerialize> CanonicalSerialize for UnivariatePoly<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.coefficients.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coefficients.serialized_size(compress)
    }
}

impl<F: Field + CanonicalDeserialize> Valid for UnivariatePoly<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.coefficients.check()
    }
}

impl<F: Field + CanonicalDeserialize> CanonicalDeserialize for UnivariatePoly<F> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self::new(deserialize_vec(reader, compress, validate)?))
    }
}

impl<F: Field + CanonicalSerialize> CanonicalSerialize for CompressedPoly<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.coeffs_except_linear_term
            .serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coeffs_except_linear_term.serialized_size(compress)
    }
}

impl<F: Field + CanonicalDeserialize> Valid for CompressedPoly<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.coeffs_except_linear_term.check()
    }
}

impl<F: Field + CanonicalDeserialize> CanonicalDeserialize for CompressedPoly<F> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self::new(deserialize_vec(reader, compress, validate)?))
    }
}

/// The product, with trailing zero coefficients trimmed (a nonempty zero product keeps one
/// coefficient). Evaluation reduction's `h` is built from these products and travels in the
/// proof, so the trimming is part of the wire form.
impl<F: Field> Mul<&UnivariatePoly<F>> for &UnivariatePoly<F> {
    type Output = UnivariatePoly<F>;

    fn mul(self, rhs: &UnivariatePoly<F>) -> UnivariatePoly<F> {
        if self.coefficients.is_empty() || rhs.coefficients.is_empty() {
            return UnivariatePoly::zero();
        }
        let mut coefficients =
            vec![F::zero(); self.coefficients.len() + rhs.coefficients.len() - 1];
        for (i, lhs) in self.coefficients.iter().enumerate() {
            for (j, rhs) in rhs.coefficients.iter().enumerate() {
                coefficients[i + j] += *lhs * *rhs;
            }
        }
        let mut product = UnivariatePoly::new(coefficients);
        product.trim_trailing_zeros();
        product
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::field::fp128::Fp128 as Fr;
    use jolt_field::Ring;
    use num_traits::{One, Zero};

    #[test]
    fn horner_known_polynomial() {
        let p = UnivariatePoly::new(vec![Fr::from_u64(3), Fr::from_u64(2), Fr::from_u64(1)]);
        assert_eq!(p.evaluate(Fr::from_u64(0)), Fr::from_u64(3));
        assert_eq!(p.evaluate(Fr::from_u64(1)), Fr::from_u64(6));
        assert_eq!(p.evaluate(Fr::from_u64(2)), Fr::from_u64(11));
    }

    #[test]
    fn degree_is_correct() {
        let p = UnivariatePoly::<Fr>::zero();
        assert_eq!(p.degree(), 0);

        let p = UnivariatePoly::new(vec![Fr::from_u64(5)]);
        assert_eq!(p.degree(), 0);

        let p = UnivariatePoly::new(vec![Fr::from_u64(1), Fr::from_u64(1), Fr::from_u64(1)]);
        assert_eq!(p.degree(), 2);
    }

    #[test]
    fn zero_polynomial_evaluates_to_zero() {
        let p = UnivariatePoly::<Fr>::zero();
        assert!(p.evaluate(Fr::from_u64(42)).is_zero());
    }

    #[test]
    fn interpolate_linear() {
        let points = vec![
            (Fr::from_u64(0), Fr::from_u64(1)),
            (Fr::from_u64(1), Fr::from_u64(3)),
        ];
        let p = UnivariatePoly::interpolate(&points);
        assert_eq!(p.evaluate(Fr::from_u64(5)), Fr::from_u64(11));
    }

    #[test]
    fn serde_round_trip() {
        let p = UnivariatePoly::new(vec![Fr::from_u64(3), Fr::from_u64(2), Fr::from_u64(1)]);
        let bytes = bincode::serde::encode_to_vec(&p, bincode::config::standard()).unwrap();
        let recovered: UnivariatePoly<Fr> =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .unwrap()
                .0;
        assert_eq!(p, recovered);
    }

    #[test]
    fn interpolate_over_integers_matches_evaluations() {
        let evals: Vec<Fr> = vec![
            Fr::from_u64(7),
            Fr::from_u64(3),
            Fr::from_u64(11),
            Fr::from_u64(2),
        ];
        let poly = UnivariatePoly::interpolate_over_integers(&evals);

        for (i, &expected) in evals.iter().enumerate() {
            let x = Fr::from_u64(i as u64);
            assert_eq!(poly.evaluate(x), expected, "mismatch at x={i}");
        }
    }

    #[test]
    fn interpolate_single_point_constant() {
        let c = Fr::from_u64(7);
        let poly = UnivariatePoly::interpolate(&[(Fr::from_u64(0), c)]);
        assert_eq!(poly.evaluate(Fr::from_u64(0)), c);
        assert_eq!(poly.evaluate(Fr::from_u64(99)), c);
        assert_eq!(poly.degree(), 0);
    }

    #[test]
    fn add_polynomials() {
        let a = UnivariatePoly::new(vec![Fr::from_u64(1), Fr::from_u64(2)]);
        let b = UnivariatePoly::new(vec![Fr::from_u64(3), Fr::from_u64(1), Fr::from_u64(5)]);
        let sum = &a + &b;
        assert_eq!(
            sum,
            UnivariatePoly::new(vec![Fr::from_u64(4), Fr::from_u64(3), Fr::from_u64(5)])
        );
    }

    #[test]
    fn add_assign_extends_shorter() {
        let mut a = UnivariatePoly::new(vec![Fr::from_u64(1)]);
        let b = UnivariatePoly::new(vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(4)]);
        a += &b;
        assert_eq!(
            a,
            UnivariatePoly::new(vec![Fr::from_u64(3), Fr::from_u64(3), Fr::from_u64(4)])
        );
    }

    #[test]
    fn sub_polynomials() {
        let a = UnivariatePoly::new(vec![Fr::from_u64(5), Fr::from_u64(3)]);
        let b = UnivariatePoly::new(vec![Fr::from_u64(2), Fr::from_u64(1)]);
        let diff = a - b;
        assert_eq!(
            diff,
            UnivariatePoly::new(vec![Fr::from_u64(3), Fr::from_u64(2)])
        );
    }

    #[test]
    fn neg_polynomial() {
        let p = UnivariatePoly::new(vec![Fr::from_u64(1), Fr::from_u64(2)]);
        let neg_p = -p.clone();
        let sum = p + neg_p;
        assert!(sum.is_zero());
    }

    #[test]
    fn scalar_mul() {
        let p = UnivariatePoly::new(vec![Fr::from_u64(1), Fr::from_u64(2)]);
        let scaled = &p * Fr::from_u64(3);
        assert_eq!(
            scaled,
            UnivariatePoly::new(vec![Fr::from_u64(3), Fr::from_u64(6)])
        );
    }

    #[test]
    fn scalar_mul_assign() {
        let mut p = UnivariatePoly::new(vec![Fr::from_u64(2), Fr::from_u64(4)]);
        p *= Fr::from_u64(5);
        assert_eq!(
            p,
            UnivariatePoly::new(vec![Fr::from_u64(10), Fr::from_u64(20)])
        );
    }

    #[test]
    fn from_evals_cubic() {
        let evals = vec![
            Fr::from_u64(1),
            Fr::from_u64(7),
            Fr::from_u64(23),
            Fr::from_u64(55),
        ];
        let poly = UnivariatePoly::from_evals(&evals);
        assert_eq!(poly.coefficients[0], Fr::from_u64(1));
        assert_eq!(poly.coefficients[1], Fr::from_u64(3));
        assert_eq!(poly.coefficients[2], Fr::from_u64(2));
        assert_eq!(poly.coefficients[3], Fr::from_u64(1));
    }

    #[test]
    fn from_evals_matches_interpolate_over_integers() {
        let evals: Vec<Fr> = vec![
            Fr::from_u64(7),
            Fr::from_u64(3),
            Fr::from_u64(11),
            Fr::from_u64(2),
        ];
        let via_evals = UnivariatePoly::from_evals(&evals);
        let via_lagrange = UnivariatePoly::interpolate_over_integers(&evals);

        for i in 0..10u64 {
            let x = Fr::from_u64(i);
            assert_eq!(
                via_evals.evaluate(x),
                via_lagrange.evaluate(x),
                "mismatch at x={i}"
            );
        }
    }

    #[test]
    fn from_evals_and_hint() {
        let hint = Fr::from_u64(7);
        let evals = vec![Fr::from_u64(1), Fr::from_u64(15)];
        let poly = UnivariatePoly::from_evals_and_hint(hint, &evals);

        assert_eq!(poly.evaluate(Fr::from_u64(0)), Fr::from_u64(1));
        assert_eq!(poly.evaluate(Fr::from_u64(1)), Fr::from_u64(6));
        assert_eq!(poly.evaluate(Fr::from_u64(2)), Fr::from_u64(15));
    }

    #[test]
    #[should_panic(expected = "cannot interpolate zero evaluations")]
    fn from_evals_and_hint_rejects_empty() {
        let _ = UnivariatePoly::<Fr>::from_evals_and_hint(Fr::one(), &[]);
    }

    #[test]
    #[should_panic(expected = "cannot interpolate zero evaluations")]
    fn from_evals_toom_rejects_empty() {
        let _ = UnivariatePoly::<Fr>::from_evals_toom(&[]);
    }

    #[test]
    fn toom_recovers_coefficients_and_preserves_zero_leading_terms() {
        for degree in [0, 1, 4, 8, 32] {
            for leading in [Fr::zero(), Fr::from_u64(19)] {
                let mut coefficients: Vec<Fr> = (0..=degree)
                    .map(|index| Fr::from_u64(index as u64 + 3))
                    .collect();
                coefficients[degree] = leading;
                let expected = UnivariatePoly::new(coefficients);
                let mut samples: Vec<Fr> = (0..degree)
                    .map(|index| expected.evaluate(Fr::from_u64(index as u64)))
                    .collect();
                samples.push(leading);
                assert_eq!(UnivariatePoly::from_evals_toom(&samples), expected);
            }
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod compressed_tests {
    use super::*;
    use crate::field::fp128::Fp128 as Fr;
    use jolt_field::Ring;
    use num_traits::{One, Zero};

    fn poly_and_hint(coeffs: Vec<Fr>) -> (UnivariatePoly<Fr>, Fr) {
        let p = UnivariatePoly::new(coeffs);
        let hint = p.evaluate(Fr::zero()) + p.evaluate(Fr::one());
        (p, hint)
    }

    #[test]
    fn compress_linear_polynomial() {
        let (p, hint) = poly_and_hint(vec![Fr::from_u64(5), Fr::from_u64(7)]);
        let compressed = p.compress();

        assert_eq!(compressed.degree(), 1);
        assert_eq!(compressed.coeffs_except_linear_term().len(), 1);

        let x = Fr::from_u64(3);
        assert_eq!(compressed.evaluate_with_hint(hint, x), p.evaluate(x));
    }

    #[test]
    fn compress_cubic_polynomial() {
        let (p, hint) = poly_and_hint(vec![
            Fr::from_u64(1),
            Fr::from_u64(3),
            Fr::from_u64(2),
            Fr::from_u64(1),
        ]);
        let compressed = p.compress();

        assert_eq!(compressed.degree(), 3);
        assert_eq!(compressed.coeffs_except_linear_term().len(), 3);

        for i in 0..10 {
            let x = Fr::from_u64(i);
            assert_eq!(
                compressed.evaluate_with_hint(hint, x),
                p.evaluate(x),
                "mismatch at x={i}"
            );
        }
    }

    /// The proof wire form is the coefficient vector as an arkworks `Vec`, and decodes back.
    #[test]
    fn canonical_serialization_is_the_coefficient_vector() {
        use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
        let (p, _) = poly_and_hint(vec![
            Fr::from_u64(1),
            Fr::from_u64(3),
            Fr::from_u64(2),
            Fr::from_u64(1),
        ]);
        let compressed = p.compress();
        for (bytes, expected) in [
            (wire(&p), wire(&p.coefficients().to_vec())),
            (
                wire(&compressed),
                wire(&compressed.coeffs_except_linear_term().to_vec()),
            ),
        ] {
            assert_eq!(bytes, expected);
        }
        assert_eq!(
            UnivariatePoly::<Fr>::deserialize_compressed(&wire(&p)[..]).unwrap(),
            p
        );
        assert_eq!(
            CompressedPoly::<Fr>::deserialize_compressed(&wire(&compressed)[..]).unwrap(),
            compressed
        );

        fn wire(value: &impl CanonicalSerialize) -> Vec<u8> {
            let mut bytes = Vec::new();
            value.serialize_compressed(&mut bytes).unwrap();
            bytes
        }
    }

    #[test]
    fn serde_round_trip() {
        let (p, _) = poly_and_hint(vec![
            Fr::from_u64(1),
            Fr::from_u64(3),
            Fr::from_u64(2),
            Fr::from_u64(1),
        ]);
        let compressed = p.compress();
        let bytes =
            bincode::serde::encode_to_vec(&compressed, bincode::config::standard()).unwrap();
        let recovered: CompressedPoly<Fr> =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .unwrap()
                .0;
        assert_eq!(compressed, recovered);
    }

    #[test]
    #[should_panic(expected = "cannot evaluate an empty compressed polynomial")]
    fn empty_compressed_poly_evaluation_rejected() {
        let empty = CompressedPoly::<Fr>::new(vec![]);
        let _ = empty.evaluate_with_hint(Fr::one(), Fr::one());
    }

    #[test]
    fn degree_matches_standard() {
        for deg in 1..=5 {
            let coeffs: Vec<Fr> = (0..=deg).map(|i| Fr::from_u64(i as u64 + 1)).collect();
            let p = UnivariatePoly::new(coeffs);
            let compressed = p.compress();
            assert_eq!(
                compressed.degree(),
                p.degree(),
                "degree mismatch for deg={deg}"
            );
        }
    }
}
