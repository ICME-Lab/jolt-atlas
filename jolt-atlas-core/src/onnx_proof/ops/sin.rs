use atlas_trace::op::Sin;
pub use jolt_atlas_verifier::onnx_proof::ops::sin::*;

crate::impl_plan_op_prover!(Sin);
