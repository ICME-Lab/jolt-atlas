use super::*;
use crate::onnx_proof::trace_view::TraceView;
use atlas_sumcheck::opening_proof::prover::ProverOpeningAccumulator;
use atlas_sumcheck::{
    hamming_booleanity::prover::HammingBooleanitySumcheckProver,
    hamming_weight::prover::HammingWeightSumcheckProver,
};
use atlas_transcript::ProvingTranscript;
pub use jolt_atlas_verifier::onnx_proof::ops::gather::small::*;

pub(crate) fn build_stage2_prover<F: JoltField>(
    computation_node: &ComputationNode,
    trace: &Trace,
    accumulator: &ProverOpeningAccumulator<F>,
    transcript: &mut impl ProvingTranscript,
) -> HammingBooleanitySumcheckProver<F> {
    let LayerData { operands, .. } = TraceView::layer_data(trace, computation_node);
    let [_, indexes] = operands[..] else {
        panic!("Expected two operands for Gather operation")
    };
    let num_lookups = indexes.len();

    let hb_params =
        ra_hamming_bool_params::<F>(computation_node, num_lookups, accumulator, transcript)
            .expect("the prover holds every claim it reads");
    let hw = {
        let mut lookup_vec = vec![F::one(); indexes.len()];
        lookup_vec.resize(indexes.len().next_power_of_two(), F::zero());
        lookup_vec
    };
    HammingBooleanitySumcheckProver::gen(hb_params, vec![hw])
}

pub(crate) fn build_stage3_prover<F: JoltField>(
    computation_node: &ComputationNode,
    trace: &Trace,
    accumulator: &ProverOpeningAccumulator<F>,
    transcript: &mut impl ProvingTranscript,
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
