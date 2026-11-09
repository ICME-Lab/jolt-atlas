use super::*;
use crate::onnx_proof::trace_view::TraceView;
use atlas_sumcheck::{
    booleanity::prover::SmallBooleanitySumcheckProver,
    hamming_booleanity::prover::HammingBooleanitySumcheckProver,
    hamming_weight::prover::HammingWeightSumcheckProver,
};
use atlas_trace::op::GatherSmall;
pub use jolt_atlas_verifier::onnx_proof::ops::gather::small::*;

crate::impl_plan_op_prover!(GatherSmall);

pub(crate) fn build_stage2_provers<F: JoltField>(
    computation_node: &ComputationNode,
    trace: &Trace,
    accumulator: &ProverOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> (
    HammingBooleanitySumcheckProver<F>,
    SmallBooleanitySumcheckProver<F>,
) {
    let Op::GatherSmall(gather_op) = &computation_node.operator else {
        panic!("Expected GatherSmall operator")
    };
    let LayerData { operands, .. } = TraceView::layer_data(trace, computation_node);
    let [dict, indexes] = operands[..] else {
        panic!("Expected two operands for Gather operation")
    };
    let num_words = dict.dims()[gather_op.axis];
    let num_lookups = indexes.len();

    let hb_params =
        ra_hamming_bool_params::<F>(computation_node, num_lookups, accumulator, transcript)
            .expect("the prover holds every claim it reads");
    let bool_params = ra_booleanity_params::<F>(
        computation_node,
        num_words,
        num_lookups,
        accumulator,
        transcript,
    )
    .expect("the prover holds every claim it reads");

    let hw = {
        let mut lookup_vec = vec![F::one(); indexes.len()];
        lookup_vec.resize(indexes.len().next_power_of_two(), F::zero());
        lookup_vec
    };
    let ra_evals = compute_ra_evals(&bool_params.r_cycle, indexes, num_words);
    let indexes_u = indexes.iter().map(|&x| Some(x as u16)).collect();

    let hb_sumcheck = HammingBooleanitySumcheckProver::gen(hb_params, vec![hw]);
    let bool_sumcheck =
        SmallBooleanitySumcheckProver::gen(bool_params, vec![ra_evals], vec![indexes_u]);

    (hb_sumcheck, bool_sumcheck)
}

pub(crate) fn build_stage3_prover<F: JoltField>(
    computation_node: &ComputationNode,
    trace: &Trace,
    accumulator: &ProverOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> HammingWeightSumcheckProver<F> {
    let Op::GatherSmall(gather_op) = &computation_node.operator else {
        panic!("Expected GatherSmall operator")
    };
    let LayerData { operands, .. } = TraceView::layer_data(trace, computation_node);
    let [dict, indexes] = operands[..] else {
        panic!("Expected two operands for Gather operation")
    };
    let num_words = dict.dims()[gather_op.axis];

    let hw_params =
        ra_hamming_weight_params::<F>(computation_node, num_words, accumulator, transcript)
            .expect("the prover holds every claim it reads");

    let ra_evals = compute_ra_evals(&hw_params.r_cycle, indexes, num_words);

    HammingWeightSumcheckProver::gen(hw_params, vec![ra_evals])
}
