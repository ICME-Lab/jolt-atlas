//! Pedersen-style vector commitments for round polynomial coefficients.
//!
//! The concrete implementation is intentionally left out for now.  The intended
//! commitment shape is:
//!
//! ```text
//! C = <coeffs, G> + rho * H
//! ```
//!
//! A production version should either reuse the existing `joltworks` Pedersen
//! generators or wrap them here.  The important design point for this crate is
//! not the curve API; it is that round-to-round checks are linear combinations
//! of these commitments.

use ark_ec::CurveGroup;

/// Commitment to one round polynomial's coefficient vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoeffCommitment<G: CurveGroup> {
    pub point: G,
}

/// Blinding randomness attached to a coefficient-vector commitment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoeffBlinding<F> {
    pub rho: F,
}
