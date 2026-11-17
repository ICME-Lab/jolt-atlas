use atlas_algebra::field::JoltField;
pub use jolt_atlas_verifier::onnx_proof::ops::mul::*;

use crate::node_view::ComputationNode;
use crate::onnx_proof::relation::OpRelKind;
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::poly::multilinear::MultilinearPolynomial;
use atlas_trace::{trace::LayerData, Trace};

/// The witnesses of a [`product`]: the operands it reads, each once.
pub fn product_witnesses<F: JoltField>(
    node: &ComputationNode,
    kind: OpRelKind,
    trace: &Trace,
) -> Vec<MultilinearPolynomial<F>> {
    let LayerData { operands, .. } = TraceView::layer_data(trace, node);
    self::operands(kind)
        .into_iter()
        .map(|position| MultilinearPolynomial::from(operands[position].padded_pow2()))
        .collect()
}
