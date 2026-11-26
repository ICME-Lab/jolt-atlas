//! Model-specific proof relations and their reductions.

pub mod bitz;
pub mod field;
mod fold;
pub mod norm;

mod advice;
mod attention;
mod channel;
pub mod layers;
mod linear;
mod lookup;
mod mlp;
pub mod models;
pub mod projection;
pub mod protocol;
pub mod residual;
mod rounding;
mod softmax;
mod tensor;
mod top_k;

mod tiles;
#[cfg(feature = "safetensors-weights")]
pub mod weights;
