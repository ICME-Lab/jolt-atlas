use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::OpLayout;
use crate::onnx_proof::plan::{OpPlan, OpRelation, Plan, Step};
use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};
use crate::{
    atlas_relation_types,
    onnx_proof::{
        relation::{
            reduced_output_setup, AtlasRelation, EngineRelation, RelationDerived, RelationExpr,
            RelationId, RelationOpening, RelationSetup,
        },
        ProofType,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{
    field::JoltField,
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_trace::{op::ScalarConstDiv, Op};
use atlas_transcript::Transcript;
use common::CommittedPoly;
use jolt_claims::{constant, derived, opening, SymbolicSumcheck};
use jolt_field::Ring;

crate::bind_relation!(ScalarConstDiv, ConstantDivisionRelation);

impl OpLayout for ScalarConstDiv {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        Ok(vec![CommittedPoly::ScalarConstDivNodeRemainder(node.idx)])
    }
}

impl OpPlan for ScalarConstDiv {
    fn plan(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(Plan(vec![
            Step::Reduce,
            Step::Relation {
                proof: ProofType::Execution,
                relation: OpRelation::ConstantDivision(ConstantDivisionRelation::new(node.clone())),
            },
        ]))
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
    /// The node's constant divisor; `None` unless the node is a `ScalarConstDiv`, which the
    /// setup requires.
    divisor: Option<i32>,
}

impl ConstantDivisionRelation {
    /// The constant divisor. `Err` if the node is not a `ScalarConstDiv`.
    pub fn divisor(&self) -> Result<i32, ProofVerifyError> {
        self.divisor.ok_or_else(|| {
            ProofVerifyError::Malformed(format!(
                "model: node {} is {}, not a ScalarConstDiv",
                self.node.idx, self.node.operator
            ))
        })
    }

    fn ids(&self) -> OpeningIdBuilder<'_> {
        OpeningIdBuilder::new(&self.node)
    }
}

impl SymbolicSumcheck for ConstantDivisionRelation {
    atlas_relation_types!();
    type Shape = ComputationNode;

    fn new(node: ComputationNode) -> Self {
        let divisor = match &node.operator {
            Op::ScalarConstDiv(op) => Some(op.divisor),
            _ => None,
        };
        Self { node, divisor }
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
        // The setup rejects a node without a divisor, so the zero is never proven against.
        constant(F::from_i64(self.divisor.unwrap_or(0) as i64))
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

impl Produces for ConstantDivisionRelation {
    fn produced(&self) -> Vec<OpeningId> {
        vec![
            self.ids().nodeio(Target::Input(0)),
            self.ids()
                .advice(CommittedPoly::ScalarConstDivNodeRemainder),
        ]
    }
}

impl<F: JoltField> Relation<F> for ConstantDivisionRelation {}

impl<F: JoltField> SetupRelation<F> for ConstantDivisionRelation {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> Result<RelationSetup<F>, ProofVerifyError> {
        self.divisor()?;
        reduced_output_setup(accumulator, &self.node)
    }
}

impl AtlasRelation for ConstantDivisionRelation {
    type Spec = ScalarConstDiv;

    fn node(&self) -> &ComputationNode {
        &self.node
    }
}

impl<F: JoltField> EngineRelation<F> for ConstantDivisionRelation {}
