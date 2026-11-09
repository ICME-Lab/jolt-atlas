pub use jolt_atlas_verifier::onnx_proof::ops::sum::axis::*;

use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, RelationWitnesses};
use crate::onnx_proof::trace_view::TraceView;
use crate::utils::dims::SumAxis;
use atlas_algebra::{
    field::JoltField,
    poly::{eq_poly::EqPolynomial, multilinear_polynomial::MultilinearPolynomial},
    utils::math::Math,
};
use atlas_trace::{trace::LayerData, Trace};
use common::parallel::par_enabled;
use rayon::prelude::*;

impl<F: JoltField> RelationWitnesses<F> for SumAxisRelation {
    /// The operand folded at the output point along the kept axis, over the summed axis.
    fn witnesses(&self, trace: &Trace, setup: &RelationSetup<F>) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, self.node());
        let operand = operands[0];
        let [m, n] = self.config().operand_dims()[..] else {
            panic!("a sum normalizes to two dims")
        };
        let eq = EqPolynomial::<F>::evals(&setup.incoming_point);
        let folded: Vec<F> = match self.config().axis() {
            SumAxis::Axis0 => {
                debug_assert_eq!(n.log_2(), setup.incoming_point.len());
                (0..m)
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|h| {
                        (0..n)
                            .map(|j| F::from_i32(operand[h * n + j]) * eq[j])
                            .sum()
                    })
                    .collect()
            }
            SumAxis::Axis1 => {
                debug_assert_eq!(m.log_2(), setup.incoming_point.len());
                (0..n)
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|j| {
                        (0..m)
                            .map(|h| F::from_i32(operand[h * n + j]) * eq[h])
                            .sum()
                    })
                    .collect()
            }
        };
        vec![MultilinearPolynomial::from(folded)]
    }
}
