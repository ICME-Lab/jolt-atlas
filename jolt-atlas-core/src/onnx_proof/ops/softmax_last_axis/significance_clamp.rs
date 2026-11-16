//! The prover's side of the softmax clamp lookup: its operand `u = c − x` from the trace.

pub use jolt_atlas_verifier::onnx_proof::ops::softmax_last_axis::significance_clamp::*;

use crate::node_view::ComputationNode;
use crate::onnx_proof::op_lookups::{DefaultLookupOperands, LookupWitness};
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::utils::lookup_bits::LookupBits;
use atlas_trace::{Tensor, Trace};

impl LookupWitness for SoftmaxSignificanceClampOperands {
    /// The addresses `u = c − x`, from the node's softmax witness in the trace.
    fn witness(&self, node: &ComputationNode, trace: &Trace) -> Tensor<i64> {
        let w = trace
            .softmax_witness(node.idx)
            .expect("softmax node has a softmax witness in the trace");
        super::addresses(node, w).map(|v| v as i64)
    }

    fn lookup_bits(&self, witness: &Tensor<i64>) -> Vec<LookupBits> {
        crate::onnx_proof::op_lookups::LookupWitness::lookup_bits(&DefaultLookupOperands, witness)
    }
}
