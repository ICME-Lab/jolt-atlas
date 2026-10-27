use crate::node_view::ComputationNode;
use crate::onnx_proof::op_lookups::OpLookupProve;
use crate::onnx_proof::ops::ProveOp;
use crate::onnx_proof::{op_lookups::OpLookupProvider, ProofId, ProofType, Prover};
use atlas_algebra::field::JoltField;
use atlas_lookup_tables::clamp::{ClampTable, CLAMP_BOUND};
use atlas_sumcheck::{
    shout,
    sumcheck::{prover::SumcheckInstanceProver, BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
};
use atlas_trace::op::Clamp;
use atlas_transcript::Transcript;
pub use jolt_atlas_verifier::onnx_proof::ops::clamp::*;

use common::consts::XLEN;

impl<F: JoltField, T: Transcript> ProveOp<F, T> for Clamp {
    #[tracing::instrument(skip_all, name = "Clamp::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        debug_assert_eq!(self.bound_log as usize, CLAMP_BOUND);
        let mut results = Vec::new();

        // Execution proof
        let provider: OpLookupProvider<SymmetricClampOperands> =
            OpLookupProvider::new(node.clone());
        let (mut execution_sumcheck, lookup_indices) = provider
            .read_raf_prove::<F, T, ClampTable<XLEN>, XLEN>(
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
            );
        let (execution_proof, _) = Sumcheck::prove(
            &mut execution_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), execution_proof));

        // RaOneHotChecks proof
        let encoding = provider.encoding();

        let [ra_prover, bool_prover] = shout::prover::ra_onehot_provers(
            &encoding,
            &lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![ra_prover, bool_prover];
        let (ra_one_hot_proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((
            ProofId(node.idx, ProofType::RaOneHotChecks),
            ra_one_hot_proof,
        ));

        results
    }
}
