use atlas_algebra::field::JoltField;
use atlas_trace::op::Concat;
pub use jolt_atlas_verifier::onnx_proof::ops::concat::*;

crate::impl_plan_op_prover!(Concat);

use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, RelationWitnesses};
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::poly::multilinear_polynomial::MultilinearPolynomial;
use atlas_algebra::utils::math::Math;
use atlas_trace::{trace::LayerData, Trace};
use jolt_claims::SymbolicSumcheck;

impl<F: JoltField> RelationWitnesses<F> for ConcatRelation {
    fn witnesses(&self, trace: &Trace, _setup: &RelationSetup<F>) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, self.node());
        let max_num_vars = self.rounds();
        operands
            .iter()
            .map(|operand| {
                let padded = operand.padded_pow2();
                let extended =
                    extend_input_to_max_domain(padded.data(), padded.len().log_2(), max_num_vars);
                MultilinearPolynomial::from(extended)
            })
            .collect()
    }
}

fn extend_input_to_max_domain<T: Copy>(
    input_padded: &[T],
    input_num_vars: usize,
    max_num_vars: usize,
) -> Vec<T> {
    let pad_repeat = 1usize << (max_num_vars - input_num_vars);
    let mut extended = Vec::with_capacity(input_padded.len() * pad_repeat);
    for value in input_padded {
        for _ in 0..pad_repeat {
            extended.push(*value);
        }
    }
    extended
}
