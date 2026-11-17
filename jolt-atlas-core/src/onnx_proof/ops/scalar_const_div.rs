use atlas_algebra::field::JoltField;
pub use jolt_atlas_verifier::onnx_proof::ops::scalar_const_div::*;

use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::poly::multilinear::MultilinearPolynomial;
use atlas_trace::{trace::LayerData, Trace};

/// The witnesses of a [`constant_division`]: the operand and the remainder.
pub fn constant_division_witnesses<F: JoltField>(
    node: &ComputationNode,
    trace: &Trace,
) -> Vec<MultilinearPolynomial<F>> {
    let LayerData { operands, .. } = TraceView::layer_data(trace, node);
    vec![
        MultilinearPolynomial::from(operands[0].padded_pow2()),
        MultilinearPolynomial::from(TraceView::division_remainder(trace, node).padded_pow2()),
    ]
}
