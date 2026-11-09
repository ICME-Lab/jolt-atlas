use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, RelationWitnesses, TieClaims};
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::field::JoltField;
use atlas_algebra::poly::multilinear_polynomial::MultilinearPolynomial;
use atlas_sumcheck::opening_proof::OpeningAccumulator;
use atlas_trace::{op::Div, semantics::fixed::rem_floor, trace::LayerData, Trace};
pub use jolt_atlas_verifier::onnx_proof::ops::div::*;

crate::impl_plan_op_prover!(Div);

impl<F: JoltField> RelationWitnesses<F> for DivRelation {
    fn witnesses(&self, trace: &Trace, _setup: &RelationSetup<F>) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, output } = TraceView::layer_data(trace, self.node());
        let [left, right] = operands[..] else {
            panic!("Expected two operands for Div operation")
        };
        let remainder: Vec<i32> = left
            .padded_pow2()
            .iter()
            .zip(right.padded_pow2().iter())
            // Padding lanes divide 0 by 0: no remainder.
            .map(|(&a, &b)| if b == 0 { 0 } else { rem_floor(a, b) })
            .collect();
        vec![
            MultilinearPolynomial::from(left.padded_pow2()),
            MultilinearPolynomial::from(right.padded_pow2()),
            MultilinearPolynomial::from(output.padded_pow2()),
            MultilinearPolynomial::from(remainder),
        ]
    }
}

impl<F: JoltField> TieClaims<F> for DivQuotientRelation {
    fn claims(
        &self,
        _trace: &Trace,
        _setup: &RelationSetup<F>,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        // The committed quotient is the node's output: its claim is the reduced one.
        vec![
            accumulator
                .get_node_output_opening(self.node().idx)
                .expect("the prover holds every claim it reads")
                .1,
        ]
    }
}
