use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::eval_reduction::NodeEvalReductionProve;
use crate::onnx_proof::ops::ProveOp;
use crate::onnx_proof::range_checking::RangeCheckProve;
use crate::{
    onnx_proof::deferred_lookups::{DeferredBatch, DeferredOneHot},
    onnx_proof::{
        ops::eval_reduction::NodeEvalReduction,
        range_checking::{
            range_check_operands::{
                RangeCheckOperands, RiRangeCheckOperands, RsRangeCheckOperands,
            },
            RangeCheckProvider,
        },
        relation::{RelationParams, RelationProver},
        ProofId, ProofType, Prover,
    },
};
use atlas_trace::op::Rsqrt;
use common::consts::XLEN;
pub use jolt_atlas_verifier::onnx_proof::ops::rsqrt::*;
use jolt_claims::SymbolicSumcheck;
use joltworks::{
    field::JoltField,
    lookup_tables::unsigned_less_than::UnsignedLessThanTable,
    subprotocols::{
        evaluation_reduction::EvalReductionProof,
        sumcheck::{Sumcheck, SumcheckInstanceProof},
    },
    transcripts::Transcript,
};

impl<F: JoltField, T: Transcript> ProveOp<F, T> for Rsqrt {
    #[tracing::instrument(skip_all, name = "Rsqrt::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        let mut results = Vec::new();
        // Execution proof
        let params = RelationParams::<F, _>::new(
            RsqrtRelation::new(node.clone()),
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut prover_sumcheck = RelationProver::initialize(&prover.trace, params);
        let (proof, _) = Sumcheck::prove(
            &mut prover_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), proof));

        results
    }

    fn prove_with_reduction(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> (
        EvalReductionProof<F>,
        Vec<(ProofId, SumcheckInstanceProof<F>)>,
    ) {
        let mut proofs = self.prove(node, prover);
        let eval_reduction_proof = NodeEvalReduction::prove(prover, node);
        proofs.extend(prove_range_and_onehot(node, prover));
        (eval_reduction_proof, proofs)
    }
}

fn prove_range_and_onehot<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
    // Both range checks are built now (challenges drawn, operand claims
    // appended) and proven after the node loop in the `RangeCheck` batch; their
    // one-hot checks follow that batch.
    let div_rangecheck_provider = RangeCheckProvider::<RiRangeCheckOperands>::new(node);
    let (div_rc_prover, div_lookup_indices) = div_rangecheck_provider
        .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
    prover.defer(DeferredBatch::RangeCheck, Box::new(div_rc_prover));

    let sqrt_rangecheck_provider = RangeCheckProvider::<RsRangeCheckOperands>::new(node);
    let (sqrt_rc_prover, sqrt_lookup_indices) = sqrt_rangecheck_provider
        .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
    prover.defer(DeferredBatch::RangeCheck, Box::new(sqrt_rc_prover));

    let div_operands = RangeCheckOperands::<RiRangeCheckOperands>::new(node);
    prover.deferred_onehots.push((
        DeferredOneHot::RangeCheck(div_operands.get_encoding(node)),
        div_lookup_indices,
    ));
    let sqrt_operands = RangeCheckOperands::<RsRangeCheckOperands>::new(node);
    prover.deferred_onehots.push((
        DeferredOneHot::RangeCheck(sqrt_operands.get_encoding(node)),
        sqrt_lookup_indices,
    ));

    Vec::new()
}
