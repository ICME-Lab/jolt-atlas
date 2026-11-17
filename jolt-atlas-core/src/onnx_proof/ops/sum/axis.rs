pub use jolt_atlas_verifier::onnx_proof::ops::sum::axis::*;

use crate::onnx_proof::relation::OpRel;
use crate::onnx_proof::trace_view::TraceView;
use crate::utils::dims::SumAxis;
use atlas_algebra::{
    field::JoltField,
    poly::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
    utils::math::Math,
};
use atlas_trace::{trace::LayerData, Trace};
use common::parallel::par_enabled;
use rayon::prelude::*;

/// The witness of an [`axis_sum`]: the operand folded at the output point `point` along the
/// kept axis, over the summed axis.
pub fn axis_sum_witnesses<F: JoltField>(
    relation: &OpRel<F>,
    trace: &Trace,
    point: &[F],
) -> Vec<MultilinearPolynomial<F>> {
    let LayerData { operands, .. } = TraceView::layer_data(trace, &relation.node);
    let operand = operands[0];
    let config = config(relation).expect("the relation was built for a single-axis sum");
    let [m, n] = config.operand_dims()[..] else {
        panic!("a sum normalizes to two dims")
    };
    let eq = EqPolynomial::<F>::evals(point);
    let folded: Vec<F> = match config.axis() {
        SumAxis::Axis0 => {
            debug_assert_eq!(n.log_2(), point.len());
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
            debug_assert_eq!(m.log_2(), point.len());
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
