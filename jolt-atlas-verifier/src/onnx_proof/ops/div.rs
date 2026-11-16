use crate::onnx_proof::ops::OpLayout;
use crate::onnx_proof::plan::{OpPlan, OpRelation, OpTie, Plan, Step};
use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};
use crate::{
    node_view::ComputationNode,
    onnx_proof::{
        relation::{
            AtlasRelation, EngineRelation, RelationChallenge, RelationDerived, RelationExpr,
            RelationId, RelationOpening, RelationSetup, TieRelation,
        },
        ProofType,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::field::Ring;
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_claims::{derived, opening, Expr, NoChallenges, NoInputs, NoOutputs, SymbolicSumcheck};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_trace::op::Div;
use atlas_transcript::Transcript;
use common::{CommittedPoly, VirtualPoly};

impl OpLayout for Div {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        Ok(vec![CommittedPoly::DivNodeQuotient(node.idx)])
    }
}

impl OpPlan for Div {
    fn plan(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        // The division identity runs before the reduction: it adds a claim on the node's own
        // output (the quotient), which the reduction then folds in.
        // The remainder's bound `0 ≤ rem < right` is proven in the packed remainder buckets
        // after the node loop (`global_clamp::node_remainders`).
        Ok(Plan(vec![
            Step::Relation {
                proof: ProofType::Execution,
                relation: OpRelation::Div(DivRelation::new(node.clone())),
            },
            Step::Reduce,
            Step::Tie {
                relation: OpTie::DivQuotient(DivQuotientRelation::new(node.clone())),
            },
        ]))
    }
}

crate::bind_relation!(Div, DivRelation);
crate::bind_relation!(Div, DivQuotientRelation);

/// Floor division `q = ⌊left / right⌋` as a relation, its point `r₀` drawn from the transcript:
///
/// ```text
/// input claim   0
/// output claim  eq(r₀, r′) · (right·q + rem − left)
/// ```
///
/// **S ⊆ R.** `atlas_trace::semantics::fixed::floor_div` and its remainder `rem_floor` (the
/// divisor's sign) satisfy `left = right·q + rem` at every element, so the summand vanishes on
/// the hypercube. The bound on `rem` that makes `q` the floor is the range check below,
/// outside the relation.
pub struct DivRelation {
    node: ComputationNode,
}

impl DivRelation {
    fn ids(&self) -> OpeningIdBuilder<'_> {
        OpeningIdBuilder::new(&self.node)
    }
}

impl SymbolicSumcheck for DivRelation {
    type RelationId = RelationId;
    type OpeningId = RelationOpening;
    type DerivedId = RelationDerived;
    type ChallengeId = RelationChallenge;
    type Shape = ComputationNode;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = NoInputs<C>;
    type Outputs<C> = NoOutputs<C>;

    fn new(node: ComputationNode) -> Self {
        Self { node }
    }

    fn id() -> RelationId {
        RelationId::Division
    }

    fn rounds(&self) -> usize {
        self.node.pow2_padded_num_output_elements().log_2()
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        Expr::zero()
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        let open = |id| opening(RelationOpening::Id(id));
        let left = open(self.ids().nodeio(Target::Input(0)));
        let right = open(self.ids().nodeio(Target::Input(1)));
        let quotient = open(self.ids().nodeio(Target::Current));
        let remainder = open(self.ids().advice(VirtualPoly::DivRemainder));
        derived(RelationDerived::Eq) * (right * quotient + remainder - left)
    }
}

impl Produces for DivRelation {
    fn produced(&self) -> Vec<OpeningId> {
        vec![
            self.ids().nodeio(Target::Input(0)),
            self.ids().nodeio(Target::Input(1)),
            self.ids().nodeio(Target::Current),
            self.ids().advice(VirtualPoly::DivRemainder),
        ]
    }
}

impl<F: JoltField> Relation<F> for DivRelation {}

impl<F: JoltField> SetupRelation<F> for DivRelation {
    fn setup<T: Transcript>(
        &self,
        _accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<RelationSetup<F>, ProofVerifyError> {
        Ok(RelationSetup {
            incoming_point: transcript
                .challenge_vector_optimized::<F>(self.rounds())
                .into_opening(),
            challenges: Vec::new(),
        })
    }
}

impl AtlasRelation for DivRelation {
    type Spec = Div;

    fn node(&self) -> &ComputationNode {
        &self.node
    }
}

impl<F: JoltField> EngineRelation<F> for DivRelation {}

// TODO: Reduce two claims to 1 via 4.5.2 PAZK for Quotient polynomial openings

/// The committed quotient `q` (what the PCS opens) bound to the node's reduced output, a tie:
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  q(r)                   (the committed quotient, at the same point)
/// ```
///
/// **S ⊆ R.** The committed quotient is the node's output tensor.
pub struct DivQuotientRelation {
    node: ComputationNode,
}

impl SymbolicSumcheck for DivQuotientRelation {
    type RelationId = RelationId;
    type OpeningId = RelationOpening;
    type DerivedId = RelationDerived;
    type ChallengeId = RelationChallenge;
    type Shape = ComputationNode;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = NoInputs<C>;
    type Outputs<C> = NoOutputs<C>;

    fn new(node: ComputationNode) -> Self {
        Self { node }
    }

    fn id() -> RelationId {
        RelationId::DivisionQuotient
    }

    fn rounds(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::ReducedNodeOutput(self.node.idx))
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::Id(self.quotient()))
    }
}

impl DivQuotientRelation {
    fn quotient(&self) -> OpeningId {
        OpeningIdBuilder::new(&self.node).advice(CommittedPoly::DivNodeQuotient)
    }
}

impl Produces for DivQuotientRelation {
    fn produced(&self) -> Vec<OpeningId> {
        vec![self.quotient()]
    }
}

impl<F: JoltField> Relation<F> for DivQuotientRelation {}

impl<F: JoltField> SetupRelation<F> for DivQuotientRelation {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> Result<RelationSetup<F>, ProofVerifyError> {
        Ok(RelationSetup {
            incoming_point: accumulator.get_node_output_opening(self.node.idx)?.0.r,
            challenges: Vec::new(),
        })
    }
}

impl AtlasRelation for DivQuotientRelation {
    type Spec = Div;

    fn node(&self) -> &ComputationNode {
        &self.node
    }
}

impl<F: JoltField> TieRelation<F> for DivQuotientRelation {}
