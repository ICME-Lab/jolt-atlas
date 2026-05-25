// Minimal compile surface while the layer IOP is being rewritten around
// `Claim + Poly`. Re-enable the remaining ops as they are migrated.
pub mod floor;
pub mod hadamard_mul;
pub mod matadd;
pub mod matmul;
pub mod poly_bridge;
pub mod pv_matmul;
pub mod qk_score;
pub mod rms_norm;
pub mod rope;
pub mod round;
pub mod silu;
pub mod softmax;
