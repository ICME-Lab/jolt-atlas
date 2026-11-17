use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::field::JoltField;
use atlas_algebra::poly::multilinear::MultilinearPolynomial;
use atlas_sumcheck::opening_proof::OpeningAccumulator;
use atlas_trace::{trace::LayerData, Trace};
pub use jolt_atlas_verifier::onnx_proof::ops::div::*;

/// The witnesses of a [`division`]: the operands, the quotient and the remainder.
pub fn division_witnesses<F: JoltField>(
    node: &ComputationNode,
    trace: &Trace,
) -> Vec<MultilinearPolynomial<F>> {
    let LayerData { operands, output } = TraceView::layer_data(trace, node);
    let [left, right] = operands[..] else {
        panic!("Expected two operands for Div operation")
    };
    // Padding lanes divide 0 by 0 and carry no remainder.
    let remainder = TraceView::division_remainder(trace, node).padded_pow2();
    vec![
        MultilinearPolynomial::from(left.padded_pow2()),
        MultilinearPolynomial::from(right.padded_pow2()),
        MultilinearPolynomial::from(output.padded_pow2()),
        MultilinearPolynomial::from(remainder),
    ]
}

/// The claim of a [`division_quotient`]: the committed quotient is the node's output, so its
/// claim is the reduced one.
pub fn division_quotient_claims<F: JoltField>(
    node: &ComputationNode,
    accumulator: &dyn OpeningAccumulator<F>,
) -> Vec<F> {
    vec![
        accumulator
            .get_node_output_opening(node.idx)
            .expect("the prover holds every claim it reads")
            .1,
    ]
}
