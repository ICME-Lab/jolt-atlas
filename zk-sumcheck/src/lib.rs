//! Zero-knowledge sumcheck experiments.
//!
//! This crate is intentionally small for now.  The goal is to isolate the
//! committed-round-polynomial design before wiring it into the larger Qwen
//! prover.
//!
//! The core idea is:
//!
//! ```text
//! g_i(x) = c_{i,0} + c_{i,1} x + ... + c_{i,d} x^d
//!
//! g_i(r_i) = g_{i+1}(0) + g_{i+1}(1)
//! ```
//!
//! If the coefficient vectors are hidden behind Pedersen-style vector
//! commitments, the middle rounds can be checked as commitment linear
//! relations:
//!
//! ```text
//! <C_i,     (1, r_i, r_i^2, ..., r_i^d)>
//!   ==
//! <C_{i+1}, (2, 1,   1,     ..., 1)>
//! ```
//!
//! Only the endpoints need to be opened:
//!
//! ```text
//! g_0(0) + g_0(1)
//! g_k(r_k)
//! ```
//!
//! The implementation is not filled in yet.  The module layout is the intended
//! reading order.

pub mod pedersen;
pub mod round_poly;
pub mod protocol;

pub use protocol::{CommittedRoundProof, SumcheckEndpointClaims};
