//! Round polynomial representation.
//!
//! This module owns only the univariate round-message shape:
//!
//! ```text
//! g_i(x) = c_0 + c_1 x + ... + c_d x^d
//! ```
//!
//! It does not know how `g_i` was computed from a relation.  That separation is
//! deliberate: relation-specific sumcheck code should produce coefficients, and
//! this crate should handle how those coefficients are committed and linked.

/// Coefficients of one sumcheck round polynomial in monomial basis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundPolynomial<F> {
    pub coeffs: Vec<F>,
}

impl<F> RoundPolynomial<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        Self { coeffs }
    }

    pub fn degree(&self) -> Option<usize> {
        self.coeffs.len().checked_sub(1)
    }
}
