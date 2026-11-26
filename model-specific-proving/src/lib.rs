//! Model-specific fixed-point proofs built with the Jolt crates.
//!
//! Proofs use World BitZ for polynomial commitments.

pub mod layers;
pub mod models;
pub mod protocol;
#[cfg(feature = "safetensors-weights")]
pub mod weights;
