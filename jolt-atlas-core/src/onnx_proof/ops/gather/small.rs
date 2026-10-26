use super::*;
use crate::onnx_proof::trace_view::TraceView;
use atlas_trace::op::GatherSmall;
pub use jolt_atlas_verifier::onnx_proof::ops::gather::small::*;
use joltworks::subprotocols::{
    booleanity::SmallBooleanitySumcheckProver,
    hamming_booleanity::HammingBooleanitySumcheckProver,
    hamming_weight::HammingWeightSumcheckProver,
    sumcheck::{BatchedSumcheck, Sumcheck},
};

impl<F: JoltField, T: Transcript> ProveOp<F, T> for GatherSmall {
    #[tracing::instrument(skip_all, name = "GatherSmall::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        let mut results = Vec::new();

        let params = GatherParams::new(
            node.clone(),
            &prover.preprocessing.model,
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut exec_sumcheck = GatherProver::initialize(
            &prover.trace,
            params,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        let (proof, _) = Sumcheck::prove(
            &mut exec_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), proof));

        let (hb_sumcheck, bool_sumcheck) = build_stage2_provers::<F>(node, prover);
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![Box::new(hb_sumcheck), Box::new(bool_sumcheck)];
        let (stage2_proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::RaOneHotChecks), stage2_proof));

        let mut hw_sumcheck = build_stage3_prover::<F>(node, prover);
        let (hw_proof, _) = Sumcheck::prove(
            &mut hw_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::RaHammingWeight), hw_proof));

        results
    }
}

pub(crate) fn build_stage2_provers<F: JoltField>(
    computation_node: &ComputationNode,
    prover: &mut Prover<F, impl Transcript>,
) -> (
    HammingBooleanitySumcheckProver<F>,
    SmallBooleanitySumcheckProver<F>,
) {
    let Op::GatherSmall(gather_op) = &computation_node.operator else {
        panic!("Expected GatherSmall operator")
    };
    let LayerData { operands, .. } = TraceView::layer_data(&prover.trace, computation_node);
    let [dict, indexes] = operands[..] else {
        panic!("Expected two operands for Gather operation")
    };
    let num_words = dict.dims()[gather_op.axis];
    let num_lookups = indexes.len();

    let hb_params = ra_hamming_bool_params::<F>(
        computation_node,
        num_lookups,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let bool_params = ra_booleanity_params::<F>(
        computation_node,
        num_words,
        num_lookups,
        &prover.accumulator,
        &mut prover.transcript,
    );

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
    prover: &mut Prover<F, impl Transcript>,
) -> HammingWeightSumcheckProver<F> {
    let Op::GatherSmall(gather_op) = &computation_node.operator else {
        panic!("Expected GatherSmall operator")
    };
    let LayerData { operands, .. } = TraceView::layer_data(&prover.trace, computation_node);
    let [dict, indexes] = operands[..] else {
        panic!("Expected two operands for Gather operation")
    };
    let num_words = dict.dims()[gather_op.axis];

    let hw_params = ra_hamming_weight_params::<F>(
        computation_node,
        num_words,
        &prover.accumulator,
        &mut prover.transcript,
    );

    let ra_evals = compute_ra_evals(&hw_params.r_cycle, indexes, num_words);

    HammingWeightSumcheckProver::gen(hw_params, vec![ra_evals])
}
