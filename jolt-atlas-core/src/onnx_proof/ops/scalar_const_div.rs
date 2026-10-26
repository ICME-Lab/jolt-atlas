use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    atlas_relation_types,
    onnx_proof::{
        ops::OperatorProofTrait,
        relation::{
            reduced_output_setup, AtlasRelation, EngineRelation, RelationDerived, RelationExpr,
            RelationId, RelationOpening, RelationParams, RelationProver, RelationSetup,
            RelationVerifier,
        },
        ProofId, ProofType, Prover, Verifier,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_trace::{op::ScalarConstDiv, semantics::fixed::rem_floor, trace::LayerData, Op, Trace};
use common::CommittedPoly;
use jolt_claims::{constant, derived, opening, SymbolicSumcheck};
use jolt_field::Ring;
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{OpeningAccumulator, OpeningId},
    },
    subprotocols::sumcheck::{Sumcheck, SumcheckInstanceProof},
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

crate::bind_relation!(ScalarConstDiv, ConstantDivisionRelation);

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for ScalarConstDiv {
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
        let (proof, _) = Sumcheck::prove(
            &mut exec_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        vec![(ProofId(node.idx, ProofType::Execution), proof)]
    }

    #[tracing::instrument(skip_all, name = "ScalarConstDiv::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let exec_sumcheck = RelationVerifier::<F, _>::new(
            ConstantDivisionRelation::new(node.clone()),
            &verifier.accumulator,
            &mut verifier.transcript,
        );
        Sumcheck::verify(
            proof,
            &exec_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;
        Ok(())
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        vec![CommittedPoly::ScalarConstDivNodeRemainder(node.idx)]
    }
}

/// Division by a constant `d`, `q = ⌊left / d⌋`, as a relation:
///
/// ```text
/// input claim   d · q(r₀)              (the reduced output opening, scaled)
/// output claim  eq(r₀, r′) · (left − rem)
/// ```
///
/// **S ⊆ R.** `atlas_trace::semantics::fixed::floor_div` and `rem_floor` satisfy
/// `left = d·q + rem` at every element. Nothing bounds the committed remainder, so the relation
/// does not pin `q` to the floor: a known soundness gap of this op's protocol, which the
/// relation preserves.
pub struct ConstantDivisionRelation {
    node: ComputationNode,
}

impl ConstantDivisionRelation {
    fn divisor(&self) -> i32 {
        let Op::ScalarConstDiv(op) = &self.node.operator else {
            panic!("Expected ScalarConstDiv operator")
        };
        op.divisor
    }

    fn ids(&self) -> OpeningIdBuilder<'_> {
        OpeningIdBuilder::new(&self.node)
    }
}

impl SymbolicSumcheck for ConstantDivisionRelation {
    atlas_relation_types!();
    type Shape = ComputationNode;

    fn new(node: ComputationNode) -> Self {
        Self { node }
    }

    fn id() -> RelationId {
        RelationId::ConstantDivision
    }

    fn rounds(&self) -> usize {
        self.node.pow2_padded_num_output_elements().log_2()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        constant(F::from_i64(self.divisor() as i64))
            * opening(RelationOpening::ReducedNodeOutput(self.node.idx))
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        let open = |id| opening(RelationOpening::Id(id));
        let left = open(self.ids().nodeio(Target::Input(0)));
        let remainder = open(
            self.ids()
                .advice(CommittedPoly::ScalarConstDivNodeRemainder),
        );
        derived(RelationDerived::Eq) * (left - remainder)
    }
}

impl AtlasRelation for ConstantDivisionRelation {
    type Spec = ScalarConstDiv;

    fn node(&self) -> &ComputationNode {
        &self.node
    }

    fn setup<F: JoltField, T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> RelationSetup<F> {
        reduced_output_setup(accumulator, &self.node)
    }

    fn produced(&self) -> Vec<OpeningId> {
        vec![
            self.ids().nodeio(Target::Input(0)),
            self.ids()
                .advice(CommittedPoly::ScalarConstDivNodeRemainder),
        ]
    }
}

impl EngineRelation for ConstantDivisionRelation {
    fn witnesses<F: JoltField>(
        &self,
        trace: &Trace,
        _setup: &RelationSetup<F>,
    ) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, &self.node);
        let left = operands[0].padded_pow2();
        let divisor = self.divisor();
        let remainder: Vec<i32> = left.iter().map(|&a| rem_floor(a, divisor)).collect();
        vec![
            MultilinearPolynomial::from(left),
            MultilinearPolynomial::from(remainder),
        ]
    }
}
