use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::{field::JoltField, poly::multilinear::EvaluateMle};
use atlas_trace::{trace::LayerData, Trace};
pub use jolt_atlas_verifier::onnx_proof::ops::add::*;

/// The claims of an [`accumulation`]: the two operands at the output point.
pub fn accumulation_claims<F: JoltField>(
    node: &ComputationNode,
    trace: &Trace,
    point: &[F],
) -> Vec<F> {
    let LayerData { operands, .. } = TraceView::layer_data(trace, node);
    operands[..2]
        .iter()
        .map(|operand| operand.evaluate_mle(point))
        .collect()
}
