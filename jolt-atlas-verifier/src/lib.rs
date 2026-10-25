//! Jolt Atlas Verifier - the verifier side of the ONNX proof system.
//!
//! This crate owns the proof object ([`ONNXProof`] with its
//! [`ReducedOpeningProof`], [`Claims`] and [`ProofId`]/[`ProofType`] keys) and
//! its wire format ([`proof_serialization`]). It depends on no prover crate:
//! `cargo check -p jolt-atlas-verifier` builds it in isolation. Every proof
//! field is private; `jolt-atlas-core` builds proofs through `from_parts` and
//! reads them through `&` accessors. The `test-util` feature adds `&mut`
//! accessors for the soundness harness.

#![warn(missing_docs)]

mod proof;
pub mod proof_serialization;
mod types;

pub use proof::{ONNXProof, ReducedOpeningProof};
pub use types::{Claims, ProofId, ProofType};
