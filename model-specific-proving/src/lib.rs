//! Model-specific fixed-point proofs built with the Jolt crates.
//!
//! Enable `akita` or `bitz` to select a commitment scheme.

pub mod layers;
pub mod models;
pub mod protocol;
#[cfg(feature = "safetensors-weights")]
pub mod weights;
