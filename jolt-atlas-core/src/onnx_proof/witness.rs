use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
};
use common::CommittedPolynomial;
use joltworks::{
    config::OneHotParams,
    field::JoltField,
    poly::{multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial},
    utils::{lookup_bits::LookupBits, math::Math},
};
use rayon::prelude::*;

use crate::onnx_proof::op_lookups::{
    read_raf_checking::compute_lookup_indices_from_operands, InterleavedBitsMarker,
};

/// Returns a list of symbols representing all committed polynomials.
pub fn node_committed_polynomials(node: &ComputationNode) -> Vec<CommittedPolynomial> {
    let mut polynomials = vec![];
    let one_hot_params = OneHotParams::new(node.num_output_elements().log_2()); // TODO: make more robust
    for i in 0..one_hot_params.instruction_d {
        polynomials.push(CommittedPolynomial::NodeOutputRaD(node.idx, i));
    }
    polynomials
}

pub fn generate_node_output_ra<F>(
    computation_node: &ComputationNode,
    trace: &Trace,
) -> Vec<(CommittedPolynomial, MultilinearPolynomial<F>)>
where
    F: JoltField,
{
    let LayerData {
        output: _,
        operands,
    } = Trace::layer_data(trace, computation_node);
    let is_interleaved_operands = computation_node.is_interleaved_operands();
    let lookup_indices = compute_lookup_indices_from_operands(&operands, is_interleaved_operands);
    let one_hot_params = OneHotParams::new(lookup_indices.len().log_2());
    (0..one_hot_params.instruction_d)
        .map(|d| {
            let poly = generate_node_output_ra_d(&lookup_indices, &one_hot_params, d);
            (
                CommittedPolynomial::NodeOutputRaD(computation_node.idx, d),
                poly,
            )
        })
        .collect()
}

pub fn generate_node_output_ra_d<F>(
    trace: &[LookupBits],
    one_hot_params: &OneHotParams,
    d: usize,
) -> MultilinearPolynomial<F>
where
    F: JoltField,
{
    let addresses: Vec<_> = trace
        .par_iter()
        .map(|lookup_index| Some(one_hot_params.lookup_index_chunk(lookup_index.into(), d)))
        .collect();
    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
        addresses,
        one_hot_params.k_chunk,
    ))
}
