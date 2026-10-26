//! Exp-LUT materialization and small helpers for the softmax sumchecks; the
//! witness itself is the trace-captured [`atlas_trace::SoftmaxWitness`].

pub use atlas_trace::semantics::tables::ExpLutDecomposed;
pub use jolt_atlas_verifier::onnx_proof::ops::softmax_last_axis::witness::*;
