use atlas_algebra::field::JoltField;
pub use jolt_atlas_verifier::onnx_proof::ops::mul::*;

use crate::onnx_proof::relation::OpRel;
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::poly::multilinear::MultilinearPolynomial;
use atlas_trace::{trace::LayerData, Trace};

/// The witnesses of a [`product`]: the operands it reads, each once. Its produced openings
/// are the node's leading operands, in order.
pub fn product_witnesses<F: JoltField>(
    relation: &OpRel<F>,
    trace: &Trace,
) -> Vec<MultilinearPolynomial<F>> {
    let LayerData { operands, .. } = TraceView::layer_data(trace, &relation.node);
    operands[..relation.produced.len()]
        .iter()
        .map(|operand| MultilinearPolynomial::from(operand.padded_pow2()))
        .collect()
}
