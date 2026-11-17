//! The prover's side of the clamped activations
//! ([`jolt_atlas_verifier::onnx_proof::ops::activation_clamped`]): the clamped input the small
//! table is read at.

pub use jolt_atlas_verifier::onnx_proof::ops::activation_clamped::*;

use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use atlas_trace::{trace::LayerData, Tensor, Trace};

/// A node's input clamped to the signed `vars`-bit range: the addresses its small table is
/// read at.
pub fn clamped_input(node: &ComputationNode, trace: &Trace, vars: usize) -> Tensor<i32> {
    let LayerData { operands, .. } = TraceView::layer_data(trace, node);
    let clamp_bound = 1i32 << (vars - 1);
    operands[0].map(|x| x.clamp(-clamp_bound, clamp_bound - 1))
}
