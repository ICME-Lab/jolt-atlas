use atlas_algebra::field::JoltField;
pub use jolt_atlas_verifier::onnx_proof::ops::iff::*;

use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::poly::multilinear::MultilinearPolynomial;
use atlas_trace::{trace::LayerData, Trace};

/// The witnesses of a [`select`]: the mask and the two branches.
pub fn select_witnesses<F: JoltField>(
    node: &ComputationNode,
    trace: &Trace,
) -> Vec<MultilinearPolynomial<F>> {
    let LayerData { operands, .. } = TraceView::layer_data(trace, node);
    let [mask, a, b] = operands[..] else {
        panic!("Expected three operands for Iff operation")
    };
    [mask, a, b]
        .map(|operand| MultilinearPolynomial::from(operand.padded_pow2()))
        .into()
}
