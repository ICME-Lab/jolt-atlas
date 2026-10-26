//! Jolt Atlas Verifier - the verifier side of the ONNX proof system.
//!
//! This crate owns the proof object ([`ONNXProof`] with its
//! [`ReducedOpeningProof`], [`Claims`] and [`ProofId`]/[`ProofType`] keys), its
//! wire format ([`proof_serialization`]), and verification ([`onnx_proof`]:
//! [`onnx_proof::verify`] and [`ONNXProof::verify`], the per-op verifiers, the
//! relations they derive from, and the verifier preprocessing). It depends on
//! no prover crate: `cargo check -p jolt-atlas-verifier` builds it in
//! isolation, and `clippy.toml` disallows the vendored crates' prover items.
//! Every proof field is private; `jolt-atlas-core` builds proofs through
//! `from_parts` and reads them through `&` accessors. The `test-util` feature
//! adds `&mut` accessors and the `probe` self-check state for the soundness
//! harness.

#![allow(
    clippy::len_without_is_empty,
    clippy::needless_range_loop,
    clippy::new_without_default,
    non_snake_case,
    type_alias_bounds
)]
#![warn(missing_docs)]

pub mod node_view;
pub mod onnx_proof;
#[cfg(feature = "test-util")]
pub mod probe;
mod proof;
pub mod proof_serialization;
mod types;
pub mod utils;

pub use proof::{ONNXProof, ReducedOpeningProof};
pub use types::{Claims, ProofId, ProofType};
