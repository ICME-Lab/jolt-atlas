//! Jolt Atlas Verifier - the verifier side of the ONNX proof system.
//!
//! This crate owns the proof ([`ONNXProof`], the byte string of the prover's transcript) and
//! verification ([`onnx_proof`]: [`onnx_proof::verify`] and [`ONNXProof::verify`], the
//! schedule of what is checked, the relations it is derived from, and the verifier
//! preprocessing). It depends on no prover crate: `cargo check -p jolt-atlas-verifier` builds
//! it in isolation, and `clippy.toml` disallows the vendored crates' prover items. The
//! `test-util` feature adds accessors to the verifier's accumulator and transcript for tests.

#![allow(
    clippy::len_without_is_empty,
    clippy::needless_range_loop,
    clippy::new_without_default,
    non_snake_case,
    type_alias_bounds
)]
#![warn(missing_docs)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::unimplemented,
    clippy::todo,
    clippy::indexing_slicing
)]

pub mod node_view;
pub mod onnx_proof;
mod proof;
pub mod utils;

pub use proof::ONNXProof;
