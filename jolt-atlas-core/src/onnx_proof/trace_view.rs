//! Proof-layer view helpers over [`atlas_trace::Trace`].
//!
//! The trace is stored exactly as `atlas_trace` produced it (a `Vec` of node
//! outputs plus sparse aux witnesses); this trait adds the node-view-based
//! accessors the proof layer works with, so no re-keyed copy of the trace is
//! ever built.

use crate::node_view::ComputationNode;
use atlas_algebra::field::JoltField;
use atlas_algebra::poly::multilinear::{EvaluateMle, MultilinearPolynomial, PolynomialEvaluation};
use atlas_trace::{
    trace::{FusedIntermediates, LayerData},
    NodeId, RsqrtAdvice, SoftmaxWitness, Tensor, Trace, TrigPeriodAdvice,
};

/// Node-view-based accessors over the raw trace.
pub trait TraceView {
    /// Cached fused-rescale intermediates of a node, if any.
    fn fused_intermediates(&self, idx: usize) -> Option<&FusedIntermediates>;
    /// The softmax witness of a node, if any.
    fn softmax_witness(&self, idx: usize) -> Option<&SoftmaxWitness>;
    /// The floor-division remainder of a `ScalarConstDiv` node. Panics for any other.
    fn division_remainder(&self, n: &ComputationNode) -> &Tensor<i32>;
    /// The division and square-root advice of an `Rsqrt` node. Panics for any other.
    fn rsqrt_advice(&self, n: &ComputationNode) -> &RsqrtAdvice;
    /// The period reduction of a `Sin` or `Cos` node's input. Panics for any other.
    fn trig_period_advice(&self, n: &ComputationNode) -> &TrigPeriodAdvice;
    /// Operands and output of a node.
    fn layer_data<'a>(&'a self, n: &ComputationNode) -> LayerData<'a>;
    /// Operand tensors of a node.
    fn operand_tensors<'a>(&'a self, n: &ComputationNode) -> Vec<&'a Tensor<i32>>;
    /// The value of node `idx` over its padded cells as a polynomial: its wide value if it
    /// has one, its `i32` output otherwise.
    fn value_poly<F: JoltField>(&self, idx: usize) -> MultilinearPolynomial<F>;
    /// The multilinear extension of node `idx`'s value at `point`.
    fn value_at<F: JoltField>(&self, idx: usize, point: &[F]) -> F;
}

impl TraceView for Trace {
    fn fused_intermediates(&self, idx: usize) -> Option<&FusedIntermediates> {
        self.fused(NodeId(idx as u32))
    }

    fn softmax_witness(&self, idx: usize) -> Option<&SoftmaxWitness> {
        self.softmax(NodeId(idx as u32))
    }

    fn division_remainder(&self, n: &ComputationNode) -> &Tensor<i32> {
        self.remainder(NodeId(n.idx as u32)).unwrap_or_else(|| {
            panic!(
                "node {} ({}) has no remainder in the trace",
                n.idx, n.operator
            )
        })
    }

    fn rsqrt_advice(&self, n: &ComputationNode) -> &RsqrtAdvice {
        self.rsqrt(NodeId(n.idx as u32)).unwrap_or_else(|| {
            panic!(
                "node {} ({}) has no rsqrt advice in the trace",
                n.idx, n.operator
            )
        })
    }

    fn trig_period_advice(&self, n: &ComputationNode) -> &TrigPeriodAdvice {
        self.trig_period(NodeId(n.idx as u32)).unwrap_or_else(|| {
            panic!(
                "node {} ({}) has no trig-period advice in the trace",
                n.idx, n.operator
            )
        })
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

    fn value_poly<F: JoltField>(&self, idx: usize) -> MultilinearPolynomial<F> {
        match self.wide(NodeId(idx as u32)) {
            Some(wide) => MultilinearPolynomial::from(wide.padded_pow2().into_data()),
            None => MultilinearPolynomial::from(self.node_outputs[idx].padded_pow2()),
        }
    }

    fn value_at<F: JoltField>(&self, idx: usize, point: &[F]) -> F {
        match self.wide(NodeId(idx as u32)) {
            Some(_) => self.value_poly::<F>(idx).evaluate(point),
            None => self.node_outputs[idx].evaluate_mle(point),
        }
    }
}
