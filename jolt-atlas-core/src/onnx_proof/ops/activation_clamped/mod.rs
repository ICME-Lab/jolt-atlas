//! The prover's side of the clamped activations
//! ([`jolt_atlas_verifier::onnx_proof::ops::activation_clamped`]): the clamped input the small
//! table is read at, and the input clamp's lookup witness.

pub use jolt_atlas_verifier::onnx_proof::ops::activation_clamped::*;

use crate::node_view::ComputationNode;
use crate::onnx_proof::op_lookups::{DefaultLookupOperands, LookupWitness};
use crate::onnx_proof::trace_view::TraceView;
use crate::utils::compute_lookup_indices;
use atlas_algebra::utils::lookup_bits::LookupBits;
use atlas_trace::{trace::LayerData, Tensor, Trace};

/// A node's input clamped to the signed `vars`-bit range: the addresses its small table is
/// read at.
pub fn clamped_input(node: &ComputationNode, trace: &Trace, vars: usize) -> Tensor<i32> {
    let LayerData { operands, .. } = TraceView::layer_data(trace, node);
    let clamp_bound = 1i32 << (vars - 1);
    operands[0].map(|x| x.clamp(-clamp_bound, clamp_bound - 1))
}

impl LookupWitness for ActivationClampOperands {
    fn witness(&self, node: &ComputationNode, trace: &Trace) -> Tensor<i64> {
        LookupWitness::witness(&DefaultLookupOperands, node, trace)
    }

    fn lookup_bits(&self, witness: &Tensor<i64>) -> Vec<LookupBits> {
        let operand = witness.map(|v| v as i32);
        compute_lookup_indices(&operand)
    }
}

impl LookupWitness for crate::onnx_proof::ops::smooth_relu::SmoothReluOperands {
    fn witness(&self, node: &ComputationNode, trace: &Trace) -> Tensor<i64> {
        LookupWitness::witness(&ActivationClampOperands, node, trace)
    }

    fn lookup_bits(&self, witness: &Tensor<i64>) -> Vec<LookupBits> {
        LookupWitness::lookup_bits(&ActivationClampOperands, witness)
    }
}
