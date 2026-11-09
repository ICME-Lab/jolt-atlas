use super::*;
use crate::onnx_proof::trace_view::TraceView;
use common::parallel::par_enabled;
pub use jolt_atlas_verifier::onnx_proof::ops::gather::large::*;

pub(crate) fn gather_lookup_indices(
    computation_node: &ComputationNode,
    trace: &Trace,
) -> Vec<usize> {
    let LayerData { operands, .. } = TraceView::layer_data(trace, computation_node);
    let [_, indexes] = operands[..] else {
        panic!("Expected two operands for Gather operation")
    };
    indexes
        .padded_pow2()
        .data()
        .par_iter()
        .with_min_len(par_enabled())
        .map(|&x| x as usize)
        .collect()
}
