use atlas_algebra::field::JoltField;
pub use jolt_atlas_verifier::onnx_proof::ops::mul::*;

use crate::onnx_proof::relation::OpRel;
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::poly::multilinear::MultilinearPolynomial;
use atlas_trace::Trace;

/// The witnesses of a [`product`]: the operands it reads, each once. Its produced openings
/// are the node's leading operands, in order.
pub fn product_witnesses<F: JoltField>(
    relation: &OpRel<F>,
    trace: &Trace,
) -> Vec<MultilinearPolynomial<F>> {
    relation.node.inputs[..relation.produced.len()]
        .iter()
        .map(|&operand| TraceView::value_poly(trace, operand))
        .collect()
}
