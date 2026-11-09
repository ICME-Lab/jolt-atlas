use atlas_trace::op::Tanh;
pub use jolt_atlas_verifier::onnx_proof::ops::tanh::*;

crate::impl_plan_op_prover!(Tanh);
