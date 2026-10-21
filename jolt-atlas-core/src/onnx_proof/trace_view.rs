//! Proof-layer view helpers over [`atlas_trace::Trace`].
//!
//! The trace is stored exactly as `atlas_trace` produced it (a `Vec` of node
//! outputs plus sparse aux witnesses); this trait adds the node-view-based
//! accessors the proof layer works with, so no re-keyed copy of the trace is
//! ever built.

use crate::node_view::ComputationNode;
use atlas_trace::trace::{FusedIntermediates, LayerData};
use atlas_trace::{NodeId, SoftmaxWitness, Tensor, Trace};

/// Node-view-based accessors over the raw trace.
pub trait TraceView {
    /// Cached fused-rescale intermediates of a node, if any.
    fn fused_intermediates(&self, idx: usize) -> Option<&FusedIntermediates>;
    /// The softmax witness of a node, if any.
    fn softmax_witness(&self, idx: usize) -> Option<&SoftmaxWitness>;
    /// Operands and output of a node.
    fn layer_data<'a>(&'a self, n: &ComputationNode) -> LayerData<'a>;
    /// Operand tensors of a node.
    fn operand_tensors<'a>(&'a self, n: &ComputationNode) -> Vec<&'a Tensor<i32>>;
    /// Operand tensors with power-of-two padding applied.
    fn padded_operand_tensors(&self, n: &ComputationNode) -> Vec<Tensor<i32>>;
}

impl TraceView for Trace {
    fn fused_intermediates(&self, idx: usize) -> Option<&FusedIntermediates> {
        self.fused(NodeId(idx as u32))
    }

    fn softmax_witness(&self, idx: usize) -> Option<&SoftmaxWitness> {
        self.softmax(NodeId(idx as u32))
    }

    fn layer_data<'a>(&'a self, n: &ComputationNode) -> LayerData<'a> {
        LayerData {
            output: &self.node_outputs[n.idx],
            operands: TraceView::operand_tensors(self, n),
        }
    }

    fn operand_tensors<'a>(&'a self, n: &ComputationNode) -> Vec<&'a Tensor<i32>> {
        n.inputs.iter().map(|&i| &self.node_outputs[i]).collect()
    }

    fn padded_operand_tensors(&self, n: &ComputationNode) -> Vec<Tensor<i32>> {
        TraceView::operand_tensors(self, n)
            .into_iter()
            .map(Tensor::padded_pow2)
            .collect()
    }
}
