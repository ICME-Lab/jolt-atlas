use crate::onnx_proof::ops::eval_reduction::NodeEvalReductionProve;
use crate::onnx_proof::ops::ProveOp;
use crate::onnx_proof::range_checking::RangeCheckProve;
use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, RelationWitnesses, TieClaims};
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    node_view::ComputationNode,
    onnx_proof::{
        ops::eval_reduction::NodeEvalReduction,
        range_checking::{
            range_check_operands::{DivRangeCheckOperands, RangeCheckOperands},
            RangeCheckProvider,
        },
        relation::{prove_tie, RelationParams, RelationProver},
        ProofId, ProofType, Prover,
    },
};
use atlas_algebra::field::JoltField;
use atlas_algebra::poly::multilinear_polynomial::MultilinearPolynomial;
use atlas_lookup_tables::unsigned_less_than::UnsignedLessThanTable;
use atlas_sumcheck::opening_proof::OpeningAccumulator;
use atlas_sumcheck::relation::prover::{prove_relation, prove_relation_batch, DynRelationWitness};
use atlas_sumcheck::{
    evaluation_reduction::EvalReductionProof, shout, sumcheck::SumcheckInstanceProof,
};
use atlas_trace::{op::Div, semantics::fixed::rem_floor, trace::LayerData, Trace};
use atlas_transcript::Transcript;
use common::consts::XLEN;
pub use jolt_atlas_verifier::onnx_proof::ops::div::*;
use jolt_claims::SymbolicSumcheck;

impl<F: JoltField, T: Transcript> ProveOp<F, T> for Div {
    #[tracing::instrument(skip_all, name = "Div::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        let mut results = Vec::new();

        // Execution proof
        let params = RelationParams::<F, _>::new(
            DivRelation::new(node.clone()),
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut exec_sumcheck = RelationProver::initialize(&prover.trace, params);
        let (proof, _) = prove_relation(
            &mut exec_sumcheck,
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

        // Reduce node-output openings first, then bind the reduced output claim
        // as the committed Div quotient opening used by PCS opening verification.
        let eval_reduction_proof = NodeEvalReduction::prove(prover, node);
        prove_tie(
            &DivQuotientRelation::new(node.clone()),
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );

        if node.is_scalar() {
            return (eval_reduction_proof, proofs);
        }

        proofs.extend(prove_range_and_onehot(node, prover));
        (eval_reduction_proof, proofs)
    }
}

fn prove_range_and_onehot<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
    let mut results = Vec::new();

    let rangecheck_provider = RangeCheckProvider::<DivRangeCheckOperands>::new(node);
    let (mut rangecheck_sumcheck, lookup_indices) = rangecheck_provider
        .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
    let (rangecheck_proof, _) = prove_relation(
        &mut rangecheck_sumcheck,
        &mut prover.accumulator,
        &mut prover.transcript,
    );

    results.push((ProofId(node.idx, ProofType::RangeCheck), rangecheck_proof));

    let rc_operands = RangeCheckOperands::<DivRangeCheckOperands>::new(node);
    let encoding = rc_operands.get_encoding(node);

    let [ra_sumcheck, bool_sumcheck] = shout::prover::ra_onehot_provers(
        &encoding,
        &lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );

    let mut instances: Vec<Box<dyn DynRelationWitness<_, _>>> = vec![ra_sumcheck, bool_sumcheck];
    let (ra_one_hot_proof, _) = prove_relation_batch(
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

impl<F: JoltField> RelationWitnesses<F> for DivRelation {
    fn witnesses(&self, trace: &Trace, _setup: &RelationSetup<F>) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, output } = TraceView::layer_data(trace, self.node());
        let [left, right] = operands[..] else {
            panic!("Expected two operands for Div operation")
        };
        let remainder: Vec<i32> = left
            .padded_pow2()
            .iter()
            .zip(right.padded_pow2().iter())
            // Padding lanes divide 0 by 0: no remainder.
            .map(|(&a, &b)| if b == 0 { 0 } else { rem_floor(a, b) })
            .collect();
        vec![
            MultilinearPolynomial::from(left.padded_pow2()),
            MultilinearPolynomial::from(right.padded_pow2()),
            MultilinearPolynomial::from(output.padded_pow2()),
            MultilinearPolynomial::from(remainder),
        ]
    }
}

impl<F: JoltField> TieClaims<F> for DivQuotientRelation {
    fn claims(
        &self,
        _trace: &Trace,
        _setup: &RelationSetup<F>,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        // The committed quotient is the node's output: its claim is the reduced one.
        vec![accumulator.get_node_output_opening(self.node().idx).1]
    }
}
