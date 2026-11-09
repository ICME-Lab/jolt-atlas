use atlas_algebra::field::JoltField;
use atlas_trace::op::Mul;
pub use jolt_atlas_verifier::onnx_proof::ops::mul::*;

crate::impl_plan_op_prover!(Mul);

use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, RelationWitnesses};
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::poly::multilinear_polynomial::MultilinearPolynomial;
use atlas_trace::op::OpSpec;
use atlas_trace::{trace::LayerData, Trace};

impl<F: JoltField, S: OpSpec + ProductShape + 'static> RelationWitnesses<F> for ProductRelation<S> {
    fn witnesses(&self, trace: &Trace, _setup: &RelationSetup<F>) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, self.node());
        Self::operands()
            .into_iter()
            .map(|position| MultilinearPolynomial::from(operands[position].padded_pow2()))
            .collect()
    }
}
