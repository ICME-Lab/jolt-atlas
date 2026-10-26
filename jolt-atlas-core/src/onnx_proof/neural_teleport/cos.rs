//! Cosine lookup table for neural teleportation, sized to the downscaled remainder
//! (see [`super::trig_downscale`]); rescaled back up to `2^MODEL_SCALE` precision.

pub use jolt_atlas_verifier::onnx_proof::neural_teleport::cos::*;
