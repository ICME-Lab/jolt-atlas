//! Neural teleportation lookup tables for activation functions.
//!
//! This module contains lookup tables for activation functions (tanh, erf, etc.)
//! that are proven using small table lookups. Unlike the tables in lookup_tables/,
//! these don't use prefix-suffix decomposition and are designed for smaller
//! table sizes typical of activation functions.

pub use jolt_atlas_verifier::onnx_proof::neural_teleport::*;
pub mod cos;
pub mod division;
pub mod sin;
pub mod small_table;
pub mod trig_downscale;
pub mod utils;
