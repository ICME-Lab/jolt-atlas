use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::ProveOp;
use crate::onnx_proof::trace_view::TraceView;
use crate::utils::opening_access::IntoProverProvider;
use crate::{
    onnx_proof::{
        clamp_lookups::{is_scalar, prove_clamp_lookup},
        relation::prove_tie,
        ProofId, Prover,
    },
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_algebra::{field::JoltField, poly::multilinear_polynomial::EvaluateMle};
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_trace::{op::Add, trace::LayerData};
use atlas_transcript::Transcript;
pub use jolt_atlas_verifier::onnx_proof::ops::add::*;
use jolt_claims::SymbolicSumcheck;

impl<F: JoltField, T: Transcript> ProveOp<F, T> for Add {
    #[tracing::instrument(skip_all, name = "Add::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        prove_accumulation::<Add, F, T>(node, prover)
    }
}

/// Prove a saturating `left ± right`: the clamp lookup and the accumulation tie, or for a
/// scalar node the operands at the output point.
pub(crate) fn prove_accumulation<S: AccumulationShape, F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
    if is_scalar(node) {
        let (opening_point, _claim) =
            AccOpeningAccessor::new(&prover.accumulator, node).get_reduced_opening();
        let LayerData { operands, .. } = TraceView::layer_data(&prover.trace, node);
        let [left, right] = operands[..] else {
            panic!("Expected two operands for {}", node.operator)
        };
        let left_claim = left.evaluate_mle(&opening_point.r);
        let right_claim = right.evaluate_mle(&opening_point.r);
        let mut provider = AccOpeningAccessor::new(&mut prover.accumulator, node)
            .into_provider(&mut prover.transcript, opening_point);
        provider.append_nodeio(Target::Input(0), left_claim);
        provider.append_nodeio(Target::Input(1), right_claim);
        return Vec::new();
    }
    let results = prove_clamp_lookup(node, prover, None);
    prove_tie(
        &AccumulationRelation::<S>::new(node.clone()),
        &prover.trace,
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    results
}
