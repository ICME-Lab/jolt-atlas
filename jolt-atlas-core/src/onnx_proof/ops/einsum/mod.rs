use atlas_trace::op::Einsum;
pub use jolt_atlas_verifier::onnx_proof::ops::einsum::*;

pub mod abmk_abnk_abmn;
/// Shared sumcheck engine for einsum contraction patterns.
pub mod dot;
/// `k,nk->n` layout (vector-matrix multiply).
pub mod k_nk_n;
/// `mk,kn->mn` layout (matrix-matrix multiply).
pub mod mk_kn_mn;

crate::impl_plan_op_prover!(Einsum);
