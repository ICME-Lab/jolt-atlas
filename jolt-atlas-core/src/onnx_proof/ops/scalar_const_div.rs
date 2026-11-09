use atlas_algebra::field::JoltField;
pub use jolt_atlas_verifier::onnx_proof::ops::scalar_const_div::*;

use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, RelationWitnesses};
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::poly::multilinear_polynomial::MultilinearPolynomial;
use atlas_trace::{semantics::fixed::rem_floor, trace::LayerData, Trace};

impl<F: JoltField> RelationWitnesses<F> for ConstantDivisionRelation {
    fn witnesses(&self, trace: &Trace, _setup: &RelationSetup<F>) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, self.node());
        let left = operands[0].padded_pow2();
        let divisor = self.divisor().expect("the node is a ScalarConstDiv");
        let remainder: Vec<i32> = left.iter().map(|&a| rem_floor(a, divisor)).collect();
        vec![
            MultilinearPolynomial::from(left),
            MultilinearPolynomial::from(remainder),
        ]
    }
}
