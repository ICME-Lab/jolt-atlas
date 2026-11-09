use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::ProveOp;
use crate::onnx_proof::{
    relation::{RelationParams, RelationProver},
    ProofId, ProofType, Prover,
};
use atlas_algebra::field::JoltField;
use atlas_sumcheck::relation::prover::prove_relation;
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_trace::op::ScalarConstDiv;
use atlas_transcript::Transcript;
pub use jolt_atlas_verifier::onnx_proof::ops::scalar_const_div::*;
use jolt_claims::SymbolicSumcheck;

impl<F: JoltField, T: Transcript> ProveOp<F, T> for ScalarConstDiv {
    #[tracing::instrument(skip_all, name = "ScalarConstDiv::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        let params = RelationParams::<F, _>::new(
            ConstantDivisionRelation::new(node.clone()),
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut exec_sumcheck = RelationProver::initialize(&prover.trace, params);
        let (proof, _) = prove_relation(
            &mut exec_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        vec![(ProofId(node.idx, ProofType::Execution), proof)]
    }
}

use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, RelationWitnesses};
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::poly::multilinear_polynomial::MultilinearPolynomial;
use atlas_trace::{semantics::fixed::rem_floor, trace::LayerData, Trace};

impl<F: JoltField> RelationWitnesses<F> for ConstantDivisionRelation {
    fn witnesses(&self, trace: &Trace, _setup: &RelationSetup<F>) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, self.node());
        let left = operands[0].padded_pow2();
        let divisor = self.divisor().expect("the node is a ScalarConstDiv");
        let remainder: Vec<i32> = left.iter().map(|&a| rem_floor(a, divisor)).collect();
        vec![
            MultilinearPolynomial::from(left),
            MultilinearPolynomial::from(remainder),
        ]
    }
}
