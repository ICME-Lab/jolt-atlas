use crate::node_view::ComputationNode;
use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};
use crate::{
    onnx_proof::{
        clamp_lookups::is_scalar,
        plan::{clamp_steps, OpPlan, OpTie, Plan, Step},
        relation::{
            AtlasRelation, RelationChallenge, RelationDerived, RelationExpr, RelationId,
            RelationOpening, RelationSetup, TieRelation,
        },
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_trace::{
    op::{Add, OpSpec, Sub},
    trace::ModelExecutionIO,
    Model,
};
use atlas_transcript::Transcript;
use common::VirtualPoly;
use jolt_claims::{constant, opening, NoChallenges, NoInputs, NoOutputs, SymbolicSumcheck};
use jolt_field::Ring;
use std::marker::PhantomData;

crate::impl_plan_op!(Add, AccumulationRelation<Add>);

/// Proof for saturating element-wise addition: `output = SatClamp(left + right)`.
///
/// The accumulation is widened to `i64` (`acc = left + right`) and then clamped
/// to `i32`, so `Add` is no longer a pure linear identity. The plan ([`accumulation_plan`]) has
/// two parts:
///
/// 1. **Clamp** ([`clamp_steps`]): the accumulation ([`VirtualPoly::ClampAcc`]) is opened at
///    the node's output point `r`. A model output whose public value is unsaturated then ties
///    `acc(r) = output(r)`; every other node joins its packed clamp bucket, which proves
///    `output = SatClamp(acc)` after the node loop
///    ([`clamp_split`](crate::onnx_proof::clamp_split)).
/// 2. **Operand tie** ([`AccumulationRelation`], no sumcheck): the prover opens
///    `left`/`right` at the same output point `r`, and the verifier checks
///    `left(r) + right(r) == acc(r)`.
///
/// A scalar node skips (1): its operands open in the clear and the verifier
/// recomputes the clamp.
///
/// The accumulation is re-executed from the operands at proving time
/// ([`clamp_intermediate`](crate::onnx_proof::clamp_lookups)); it is not stored
/// in the trace.
impl OpPlan for Add {
    fn plan(
        &self,
        node: &ComputationNode,
        model: &Model,
        io: &ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(accumulation_plan::<Add>(node, model, io))
    }
}

/// A saturating `left ± right`.
pub trait AccumulationShape: OpSpec + 'static {
    /// `+1` for `Add`, `−1` for `Sub`.
    const SIGN: i64;

    /// The plan step tying the pre-clamp accumulation to the operands.
    fn accumulation_tie(node: &ComputationNode) -> OpTie;
}

impl AccumulationShape for Add {
    const SIGN: i64 = 1;

    fn accumulation_tie(node: &ComputationNode) -> OpTie {
        OpTie::AddAccumulation(AccumulationRelation::new(node.clone()))
    }
}

impl AccumulationShape for Sub {
    const SIGN: i64 = -1;

    fn accumulation_tie(node: &ComputationNode) -> OpTie {
        OpTie::SubAccumulation(AccumulationRelation::new(node.clone()))
    }
}

/// The plan of a saturating `left ± right`: the evaluation reduction, then the clamp steps and
/// the accumulation tie, or for a scalar node the clamp recomputed from the operands in the
/// clear.
pub fn accumulation_plan<S: AccumulationShape>(
    node: &ComputationNode,
    model: &Model,
    io: &ModelExecutionIO,
) -> Plan {
    let mut steps = vec![Step::Reduce];
    if is_scalar(node) {
        steps.push(Step::Tie {
            relation: OpTie::ScalarAccumulation {
                node: node.clone(),
                sign: S::SIGN,
            },
        });
    } else {
        steps.extend(clamp_steps(node, model, io));
        steps.push(Step::Tie {
            relation: S::accumulation_tie(node),
        });
    }
    Plan(steps)
}

/// The pre-clamp accumulation of a non-scalar saturating `left ± right`, a tie at the output
/// point `r`:
///
/// ```text
/// input claim   acc(r)                 (the clamp lookup's raf opening, `ClampAcc`)
/// output claim  left(r) ± right(r)
/// ```
///
/// **S ⊆ R.** The op accumulates `left ± right` in `i64` before the clamp, and the clamp
/// lookup's `acc` is that accumulation, so the identity holds at every element.
pub struct AccumulationRelation<S> {
    node: ComputationNode,
    _spec: PhantomData<fn() -> S>,
}

impl<S> AccumulationRelation<S> {
    fn ids(&self) -> OpeningIdBuilder<'_> {
        OpeningIdBuilder::new(&self.node)
    }
}

impl<S: AccumulationShape> SymbolicSumcheck for AccumulationRelation<S> {
    type RelationId = RelationId;
    type OpeningId = RelationOpening;
    type DerivedId = RelationDerived;
    type ChallengeId = RelationChallenge;
    type Shape = ComputationNode;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = NoInputs<C>;
    type Outputs<C> = NoOutputs<C>;

    fn new(node: ComputationNode) -> Self {
        Self {
            node,
            _spec: PhantomData,
        }
    }

    fn id() -> RelationId {
        RelationId::Accumulation
    }

    fn rounds(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::Id(
            self.ids().advice(VirtualPoly::ClampAcc),
        ))
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        let [left, right] = [0, 1].map(|position| {
            opening(RelationOpening::Id(
                self.ids().nodeio(Target::Input(position)),
            ))
        });
        left + constant(F::from_i64(S::SIGN)) * right
    }
}

impl<S: AccumulationShape> Produces for AccumulationRelation<S> {
    fn produced(&self) -> Vec<OpeningId> {
        vec![
            self.ids().nodeio(Target::Input(0)),
            self.ids().nodeio(Target::Input(1)),
        ]
    }
}

impl<F: JoltField, S: AccumulationShape> Relation<F> for AccumulationRelation<S> {}

impl<F: JoltField, S: AccumulationShape> SetupRelation<F> for AccumulationRelation<S> {
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

impl<S: AccumulationShape> AtlasRelation for AccumulationRelation<S> {
    type Spec = S;

    fn node(&self) -> &ComputationNode {
        &self.node
    }
}

impl<F: JoltField, S: AccumulationShape> TieRelation<F> for AccumulationRelation<S> {}
