//! Reshape proof using an eq-selector over the padded input domain.
//!
//! For each raw input element `t`, we compute the padded output index that the
//! same element occupies after reshape and place `eq(r_output, mapped_index(t))`
//! at the corresponding padded input position. The resulting selector MLE
//! satisfies:
//!
//! `out(r_output) = sum_i selector(i) * input(i)`.

pub use jolt_atlas_verifier::onnx_proof::ops::reshape::*;

use atlas_algebra::field::JoltField;

use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, RelationWitnesses};
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::poly::multilinear_polynomial::MultilinearPolynomial;
use atlas_trace::{trace::LayerData, Trace};

impl<F: JoltField> RelationWitnesses<F> for ReshapeRelation {
    fn witnesses(&self, trace: &Trace, _setup: &RelationSetup<F>) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, self.node());
        vec![MultilinearPolynomial::from(operands[0].padded_pow2())]
    }
}
