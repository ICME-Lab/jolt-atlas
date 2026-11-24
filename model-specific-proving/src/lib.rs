//! Model-specific fixed-point proofs built with the Jolt crates.
//!
//! Enable `akita` to use the Akita commitment scheme.

pub mod layers;
pub mod models;
pub mod protocol;
#[cfg(feature = "safetensors-weights")]
pub mod weights;

#[doc(hidden)]
pub mod proofs {
    pub use crate::{layers::*, protocol::*};
}
