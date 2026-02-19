//! Jolt Atlas Core - Proofs for ONNX Neural Networks
//!
//! This crate provides the core implementation of proofs for ONNX neural network
//! computations. It uses the sum-check protocol to generate succinct proofs that a neural network
//! was executed correctly on given inputs without revealing the inputs or intermediate values.
//!
//! The main entry point is the [`onnx_proof`] module, which contains the prover and verifier
//! implementations for ONNX models.

#![allow(
    clippy::len_without_is_empty,
    clippy::needless_range_loop,
    clippy::new_without_default,
    non_snake_case,
    type_alias_bounds
)]
#![warn(missing_docs)]

pub mod onnx_proof;
pub mod utils;
