use atlas_trace::op::Sigmoid;
pub use jolt_atlas_verifier::onnx_proof::ops::sigmoid::*;

crate::impl_plan_op_prover!(Sigmoid);
