use super::*;
use crate::onnx_proof::trace_view::TraceView;
use atlas_sumcheck::{
    shout,
    sumcheck::{BatchedSumcheck, Sumcheck},
};
use atlas_trace::op::GatherLarge;
use common::parallel::par_enabled;
pub use jolt_atlas_verifier::onnx_proof::ops::gather::large::*;

impl<F: JoltField, T: Transcript> ProveOp<F, T> for GatherLarge {
    #[tracing::instrument(skip_all, name = "GatherLarge::prove")]
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

        let encoding = GatherRaEncoding::new(node);
        let lookup_indices = gather_lookup_indices(node, &prover.trace);
        let [ra_prover, bool_prover] = shout::ra_onehot_provers(
            &encoding,
            &lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![ra_prover, bool_prover];

        let (stage2_proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );

        results.push((ProofId(node.idx, ProofType::RaOneHotChecks), stage2_proof));

        results
    }
}

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
