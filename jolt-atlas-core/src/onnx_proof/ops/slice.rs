use atlas_algebra::field::JoltField;
use atlas_trace::op::Slice;
pub use jolt_atlas_verifier::onnx_proof::ops::slice::*;

crate::impl_plan_op_prover!(Slice);

use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, RelationWitnesses};
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::poly::multilinear_polynomial::MultilinearPolynomial;
use atlas_trace::{trace::LayerData, Trace};

impl<F: JoltField> RelationWitnesses<F> for SliceRelation {
    fn witnesses(&self, trace: &Trace, _setup: &RelationSetup<F>) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, self.node());
        vec![MultilinearPolynomial::from(operands[0].padded_pow2())]
    }
}
