use atlas_algebra::field::JoltField;
use atlas_trace::op::Iff;
pub use jolt_atlas_verifier::onnx_proof::ops::iff::*;

crate::impl_plan_op_prover!(Iff);

use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, RelationWitnesses};
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::poly::multilinear_polynomial::MultilinearPolynomial;
use atlas_trace::{trace::LayerData, Trace};

impl<F: JoltField> RelationWitnesses<F> for IffRelation {
    fn witnesses(&self, trace: &Trace, _setup: &RelationSetup<F>) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, self.node());
        let [mask, a, b] = operands[..] else {
            panic!("Expected three operands for Iff operation")
        };
        [mask, a, b]
            .map(|operand| MultilinearPolynomial::from(operand.padded_pow2()))
            .into()
    }
}
