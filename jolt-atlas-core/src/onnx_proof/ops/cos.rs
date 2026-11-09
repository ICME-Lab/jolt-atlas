use atlas_trace::op::Cos;
pub use jolt_atlas_verifier::onnx_proof::ops::cos::*;

crate::impl_plan_op_prover!(Cos);
