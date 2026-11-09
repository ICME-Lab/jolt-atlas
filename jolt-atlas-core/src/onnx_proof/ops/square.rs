use atlas_trace::op::Square;
pub use jolt_atlas_verifier::onnx_proof::ops::square::*;

crate::impl_plan_op_prover!(Square);
