use crate::node_view::ComputationNode;
use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};
use crate::{
    onnx_proof::{
        plan::{OpPlan, OpTie, Plan, Step},
        relation::{
            AtlasRelation, RelationChallenge, RelationDerived, RelationExpr, RelationId,
            RelationOpening, RelationSetup, TieRelation,
        },
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::field::Ring;
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_claims::{constant, opening, NoChallenges, NoInputs, NoOutputs, SymbolicSumcheck};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_trace::{
    op::{Add, OpSpec, Sub},
    trace::ModelExecutionIO,
    Model,
};
use atlas_transcript::Transcript;
use std::marker::PhantomData;

crate::impl_plan_op!(Add, AccumulationRelation<Add>);

/// Proof for exact element-wise addition: `output = left + right`, where a sum outside `i32`
/// is an execution error. The plan ([`accumulation_plan`]) has two parts:
///
/// 1. **Evaluation reduction**: the node's output claims are reduced to one opening at a
///    point `r`.
/// 2. **Operand tie** ([`AccumulationRelation`], no sumcheck): the prover opens
///    `left`/`right` at `r`, and the verifier checks `left(r) + right(r) == output(r)`.
///
/// Unless the node is a model output, its output is shown to lie in `i32` by a range claim in
/// the packed range buckets, proven after the node loop
/// ([`global_clamp`](crate::onnx_proof::global_clamp)).
impl OpPlan for Add {
    fn plan(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(accumulation_plan::<Add>(node))
    }
}

/// An exact `left ± right`.
pub trait AccumulationShape: OpSpec + 'static {
    /// `+1` for `Add`, `−1` for `Sub`.
    const SIGN: i64;

    /// The plan step tying the output to the operands.
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

/// The plan of `left ± right`: the evaluation reduction, then the tie of the output to the
/// operands. The output is range-checked in the packed buckets after the node loop.
pub fn accumulation_plan<S: AccumulationShape>(node: &ComputationNode) -> Plan {
    Plan(vec![
        Step::Reduce,
        Step::Tie {
            relation: S::accumulation_tie(node),
        },
    ])
}

/// `out = left ± right` as a tie at the output point `r`: no sumcheck, the identity holds at
/// `r` because it holds on the hypercube and both sides are multilinear.
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  left(r) ± right(r)
/// ```
///
/// **S ⊆ R.** The op computes `left ± right` exactly and its execution fails when a result
/// leaves `i32`, so the identity holds at every element of every execution that exists.
///
/// **No wrap.** The operands are `i32` cells, so `|left ± right| ≤ 2^32`, and `out` lies in
/// `[−2^31, 2^31)` by its range claim (a model output by being a public `i32` tensor). Both
/// sides are integers of magnitude below `2^33`, far below half the 128-bit field modulus,
/// so their equality in the field is their equality over the integers.
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
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = NoInputs<C>;
    type Outputs<C> = NoOutputs<C>;

    fn id(&self) -> RelationId {
        RelationId::Accumulation
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
        let [left, right] = [0, 1].map(|position| {
            opening(RelationOpening::Id(
                self.ids().nodeio(Target::Input(position)),
            ))
        });
        left + constant(F::from_i64(S::SIGN)) * right
    }
}

impl<S: AccumulationShape> AccumulationRelation<S> {
    /// The relation for the given shape.
    pub fn new(node: ComputationNode) -> Self {
        Self {
            node,
            _spec: PhantomData,
        }
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
