//! Committed round-polynomial sumcheck protocol skeleton.
//!
//! This module should become the entry point.  The verifier-facing protocol is:
//!
//! 1. absorb commitments to all round polynomials;
//! 2. derive the sumcheck challenges;
//! 3. verify each middle transition by commitment linear relations;
//! 4. verify endpoint openings.
//!
//! The middle transition for degree `d` is:
//!
//! ```text
//! <C_i,     (1, r_i, r_i^2, ..., r_i^d)>
//!   ==
//! <C_{i+1}, (2, 1,   1,     ..., 1)>
//! ```
//!
//! Blinding randomness must be generated so the same linear relation also holds
//! on the blinding side.  A conservative implementation should use a slack
//! blinding commitment instead of forcing that constraint onto `rho_0`.

use ark_ec::CurveGroup;

use crate::pedersen::CoeffCommitment;

/// Public endpoint values that are intentionally revealed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckEndpointClaims<F> {
    pub initial_sum: F,
    pub final_eval: F,
}

/// Proof shell for a committed-round-polynomial sumcheck.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedRoundProof<F, G: CurveGroup> {
    pub round_commitments: Vec<CoeffCommitment<G>>,
    pub endpoints: SumcheckEndpointClaims<F>,
}
