//! The symbolic-relation layer: claim expressions ([`Expr`] over openings, derived values and
//! challenges), the [`SymbolicSumcheck`] description every atlas sumcheck implements, and the
//! claim-data traits. [`Int`] is the coefficient ring of an expression stated once for every
//! field.
//!
//! Owned by atlas. It was taken from upstream `a16z/jolt`'s `jolt-claims` (`claims`,
//! `claim_data`, `symbolic`, `ops`, `sumcheck`) at rev `f3265b5a`, without the zkVM protocol
//! graph, the derives and the RISC-V crates they need. Changes are made here; upstream is a
//! reference, consulted when its relation layer moves.

// The verifier's runtime closure: stricter panic and unsafe discipline than the workspace
// lints.
#![forbid(unsafe_code)]
#![deny(
    clippy::get_unwrap,
    clippy::string_slice,
    clippy::fallible_impl_from,
    clippy::mem_forget,
    clippy::exit,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::host_endian_bytes
)]
// wildcard_enum_match_arm is omitted: claim resolvers and test evaluation maps
// are fail-closed by construction (unmatched ids return Err/None/zero, never a
// wrong success path), so the lint would be annotation churn with no catch.

mod claim_data;
mod claims;
mod int;
mod ops;
mod sumcheck;
mod symbolic;

pub use claim_data::{
    ChallengeDrawError, InputClaims, MissingOpeningValue, NoChallenges, NoInputs, NoOutputs,
    OutputClaims, SumcheckChallenges,
};
pub use claims::{challenge, constant, derived, opening, Expr, Source, Term};
pub use int::Int;
pub use sumcheck::SumcheckDomain;
pub use symbolic::SymbolicSumcheck;
