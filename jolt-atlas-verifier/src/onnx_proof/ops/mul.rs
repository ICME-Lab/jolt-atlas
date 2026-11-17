use crate::node_view::ComputationNode;
use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};
use crate::{
    onnx_proof::{
        fused_rebase,
        plan::{fused_rescale_plan, OpPlan, OpRelation, Plan},
        relation::{
            AtlasRelation, EngineRelation, RelationChallenge, RelationDerived, RelationExpr,
            RelationId, RelationOpening, RelationSetup,
        },
        ProofType,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::field::Ring;
use atlas_algebra::{
    field::JoltField,
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_claims::{derived, opening, NoChallenges, NoInputs, NoOutputs, SymbolicSumcheck};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_trace::{
    op::{And, Cube, Mul, OpSpec, Square},
    trace::ModelExecutionIO,
    Model,
};
use atlas_transcript::Transcript;
use std::marker::PhantomData;

crate::impl_plan_op!(Mul, MulRelation);

impl OpPlan for Mul {
    fn plan(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(fused_rescale_plan(
            node,
            ProofType::RescaleArith,
            OpRelation::Mul(MulRelation::new(node.clone())),
        ))
    }
}

/// Element-wise multiplication as a relation.
pub type MulRelation = ProductRelation<Mul>;

/// An op whose output is a product of its operands, by operand position.
pub trait ProductShape {
    /// The operand position of each factor: `[0, 1]` for `L·R`, `[0, 0]` for `X²`.
    const FACTORS: &'static [usize];
}

impl ProductShape for Mul {
    const FACTORS: &'static [usize] = &[0, 1];
}

impl ProductShape for And {
    const FACTORS: &'static [usize] = &[0, 1];
}

impl ProductShape for Square {
    const FACTORS: &'static [usize] = &[0, 0];
}

impl ProductShape for Cube {
    const FACTORS: &'static [usize] = &[0, 0, 0];
}

/// An element-wise product of operands (`Mul`, `Square`, `Cube`, and `And` on booleans) as a
/// relation, `Π` running over the op's [`ProductShape::FACTORS`]:
///
/// ```text
/// input claim   out(r₀)·2^S + rem(r₀)  (fused rescale: the output is the quotient)
///               out(r₀)                (raw product: the reduced output opening)
/// output claim  eq(r₀, r′) · Π operand(r′)
/// ```
///
/// The incoming point is the node output's reduced opening point. The range claims of a fused
/// node's remainder and output are a separate protocol ([`fused_rebase`]) after this one.
///
/// **S ⊆ R.** A raw product's output is `Π operand` exactly, so `Σ eq(r₀,·)·Π operand` is the
/// output's MLE at `r₀`. A fused product splits the accumulation `Π operand` with
/// `atlas_trace::semantics::fixed::rebase` into `out·2^S + rem` (Euclidean, `S` the op's
/// rebase bits: twice the scale for `Cube`), and its execution fails when the quotient leaves
/// `i32`, so the same sum equals `out(r₀)·2^S + rem(r₀)` on every execution that exists.
///
/// **No wrap (fused).** The operands are `i32` cells, so `|Π operand| ≤ 2^62` for `Mul` and
/// `Square` and `≤ 2^93` for `Cube`. The range claims give `out ∈ [−2^31, 2^31)` (a model
/// output is a public `i32` tensor) and `rem ∈ [0, 2^S)`, so `|out·2^S + rem| < 2^(32+S)`,
/// at most `2^62` since `S ≤ 30` (the scale is at most 15, `atlas_trace::Scale`). Both sides
/// are integers of magnitude below `2^94`, far below half the 128-bit field modulus, so
/// their equality in the field is their equality over the integers.
pub struct ProductRelation<S> {
    node: ComputationNode,
    _spec: PhantomData<fn() -> S>,
}

impl<S: ProductShape> ProductRelation<S> {
    fn ids(&self) -> OpeningIdBuilder<'_> {
        OpeningIdBuilder::new(&self.node)
    }

    /// The operand positions the product reads, each once, in first-factor order.
    pub fn operands() -> Vec<usize> {
        let mut operands = Vec::new();
        for &position in S::FACTORS {
            if !operands.contains(&position) {
                operands.push(position);
            }
        }
        operands
    }
}

impl<S: OpSpec + ProductShape + 'static> SymbolicSumcheck for ProductRelation<S> {
    type RelationId = RelationId;
    type OpeningId = RelationOpening;
    type DerivedId = RelationDerived;
    type ChallengeId = RelationChallenge;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = NoInputs<C>;
    type Outputs<C> = NoOutputs<C>;

    fn id(&self) -> RelationId {
        RelationId::Product
    }

    fn rounds(&self) -> usize {
        self.node.pow2_padded_num_output_elements().log_2()
    }

    fn degree(&self) -> usize {
        S::FACTORS.len() + 1
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        fused_rebase::input_expression(&self.node)
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        S::FACTORS
            .iter()
            .fold(derived(RelationDerived::Eq), |product, &position| {
                product
                    * opening(RelationOpening::Id(
                        self.ids().nodeio(Target::Input(position)),
                    ))
            })
    }
}

impl<S: OpSpec + ProductShape + 'static> ProductRelation<S> {
    /// The relation for the given shape.
    pub fn new(node: ComputationNode) -> Self {
        Self {
            node,
            _spec: PhantomData,
        }
    }
}

impl<S: OpSpec + ProductShape + 'static> Produces for ProductRelation<S> {
    fn produced(&self) -> Vec<OpeningId> {
        Self::operands()
            .into_iter()
            .map(|position| self.ids().nodeio(Target::Input(position)))
            .collect()
    }
}

impl<F: JoltField, S: OpSpec + ProductShape + 'static> Relation<F> for ProductRelation<S> {}

impl<F: JoltField, S: OpSpec + ProductShape + 'static> SetupRelation<F> for ProductRelation<S> {
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

impl<S: OpSpec + ProductShape + 'static> AtlasRelation for ProductRelation<S> {
    type Spec = S;

    fn node(&self) -> &ComputationNode {
        &self.node
    }
}

impl<F: JoltField, S: OpSpec + ProductShape + 'static> EngineRelation<F> for ProductRelation<S> {}
