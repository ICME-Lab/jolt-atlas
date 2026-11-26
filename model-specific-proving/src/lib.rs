//! Model-specific fixed-point proofs built with the Jolt crates.
//!
//! Enable `bitz` to prove and verify with World BitZ.

pub mod layers;
pub mod models;
pub mod protocol;
#[cfg(feature = "safetensors-weights")]
pub mod weights;
