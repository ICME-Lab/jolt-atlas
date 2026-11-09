use atlas_trace::op::Sum;

/// Axis-wise sum implementations for sumcheck protocol.
pub mod axis;

crate::impl_plan_op_prover!(Sum);
