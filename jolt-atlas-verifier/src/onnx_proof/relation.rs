//! Op protocols written as symbolic relations ([`atlas_claims::SymbolicSumcheck`]) and the engine
//! that proves and verifies them. The relation core (ids, setup, [`RelationVerifier`]) is
//! `atlas_sumcheck::relation`, shared with the subprotocol relations; this module binds a
//! relation to the op it proves.
//!
//! A relation declares, over atlas openings, its **input-claim expression** (the claimed sum,
//! from openings at the incoming point `r₀`) and its **output expression** (the expected final
//! claim once every round is bound), a sum of products of produced openings with constant,
//! challenge or derived coefficients. It also says where each produced opening sits given the
//! sumcheck challenges. Prover and verifier both read the claim algebra from the relation; neither
//! restates it. For an op proven this way the relation is the protocol: the verifier is
//! [`RelationVerifier`], derived from it, and no hand-written verifier exists.
//!
//! Most relations have the engine's shape ([`EngineRelation`]): every output term is
//! `eq(r₀, r′)` times produced openings at the one low-to-high bound point `r′`. The engine,
//! [`RelationProver`], owns that eq factor (an upstream Gruen split eq over `r₀`), binds every
//! produced opening's witness and appends the openings in declared order. A relation of another
//! shape keeps a specialized prover, which reads its input claim and appends its openings
//! through [`relation_input_claim`] and [`append_produced`]; such a prover is completeness-only
//! and is checked against the relation's reference messages (`relation_prover_equiv_*`).
//!
//! The claim store stays the atlas accumulator: a relation's openings are accumulator
//! [`OpeningId`]s or a node output's reduced opening, the instance the multi-consumer evaluation
//! reduction produces for it.

use crate::node_view::ComputationNode;
use atlas_algebra::field::{IntoOpening, Ring};
use atlas_algebra::{
    field::JoltField,
    poly::{
        multilinear::BindingOrder,
        point::{OpeningPoint, BIG_ENDIAN},
    },
    utils::errors::ProofVerifyError,
};
use atlas_claims::{Expr, Int, SymbolicSumcheck};
use atlas_sumcheck::opening_proof::{
    OpeningAccumulator, OpeningId, PolynomialId, VerifierOpeningAccumulator,
};
pub use atlas_sumcheck::relation::*;
use atlas_trace::op::OpSpec;
use atlas_transcript::Transcript;

/// An op protocol as a symbolic relation on one computation node.
///
/// Each relation documents its **S ⊆ R** argument: every execution of the op's semantics
/// (`atlas_trace::semantics`, through [`Self::Spec`]) satisfies the relation, so an honest prover
/// always succeeds. That is a review obligation; the compiler checks only the op binding.
pub trait AtlasRelation:
    SymbolicSumcheck<
        RelationId = RelationId,
        OpeningId = RelationOpening,
        DerivedId = RelationDerived,
        ChallengeId = RelationChallenge,
    > + Send
    + Sync
{
    /// The op this relation proves. The dispatch macros require it to be the op the relation is
    /// dispatched for.
    type Spec: OpSpec + 'static;

    /// The node the relation proves.
    fn node(&self) -> &ComputationNode;
}

/// The node a relation proves. Temporary: it lets the relation types and [`OpRel`] share the
/// generic drivers until every relation is an [`OpRel`].
pub trait RelationNode {
    /// The node the relation proves.
    fn node(&self) -> &ComputationNode;

    /// The relation's kind, when it is an [`OpRel`].
    fn kind(&self) -> Option<OpRelKind>;

    /// The name of the op type the relation is bound to, when it is a relation type.
    fn spec_name(&self) -> Option<&'static str>;
}

impl<R: AtlasRelation> RelationNode for R {
    fn node(&self) -> &ComputationNode {
        AtlasRelation::node(self)
    }

    fn kind(&self) -> Option<OpRelKind> {
        None
    }

    fn spec_name(&self) -> Option<&'static str> {
        Some(std::any::type_name::<R::Spec>())
    }
}

impl<F: JoltField> RelationNode for OpRel<F> {
    fn node(&self) -> &ComputationNode {
        &self.node
    }

    fn kind(&self) -> Option<OpRelKind> {
        Some(self.kind)
    }

    fn spec_name(&self) -> Option<&'static str> {
        None
    }
}

/// A relation the engine proves, binding low-to-high with its produced openings at the bound
/// point (or a prefix of it), in one of two shapes:
///
/// * **eq-weighted**: each output term is [`RelationDerived::Eq`] times at most three produced
///   openings, with constant or challenge coefficients; the engine folds over a Gruen split eq;
/// * **dense**: no term has an eq factor; each term multiplies produced openings and public
///   tables ([`RelationDerived::Table`]), and the engine sends the message's evaluations at
///   `0, 2, …, degree` with the claim as hint.
pub trait EngineRelation<F: JoltField>: RelationNode + SetupRelation<F> {
    /// The order a dense relation binds its variables in (an eq-weighted one binds
    /// low-to-high).
    fn binding_order(&self) -> BindingOrder {
        BindingOrder::LowToHigh
    }
}

/// The setup of a relation whose incoming point is `node`'s reduced output point, with no
/// challenges. `Err` if the accumulator lacks the node's reduced opening.
pub fn reduced_output_setup<F: JoltField>(
    accumulator: &dyn OpeningAccumulator<F>,
    node: &ComputationNode,
) -> Result<RelationSetup<F>, ProofVerifyError> {
    Ok(RelationSetup {
        incoming_point: accumulator.get_node_output_opening(node.idx)?.0.r,
        challenges: Vec::new(),
    })
}

/// The [`SymbolicSumcheck`] items every atlas relation shares: the atlas id types and no
/// upstream claim structs (openings resolve through the accumulator).
#[macro_export]
macro_rules! atlas_relation_types {
    () => {
        type RelationId = $crate::onnx_proof::relation::RelationId;
        type OpeningId = $crate::onnx_proof::relation::RelationOpening;
        type DerivedId = $crate::onnx_proof::relation::RelationDerived;
        type ChallengeId = $crate::onnx_proof::relation::RelationChallenge;
        type Challenges<C> = atlas_claims::NoChallenges<C>;
        type Inputs<C> = atlas_claims::NoInputs<C>;
        type Outputs<C> = atlas_claims::NoOutputs<C>;
    };
}

/// A zero-round relation (a tie): its produced openings sit at points derived from the
/// incoming point, and the verifier checks the input claim against the output expression there.
pub trait TieRelation<F: JoltField>: RelationNode + SetupRelation<F> {}

/// Verify a zero-round relation (a tie): append its produced openings at the incoming point and
/// require the input claim to equal the output expression.
pub fn verify_tie<F: JoltField, T: Transcript, R: RelationNode + SetupRelation<F>>(
    relation: &R,
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<(), ProofVerifyError> {
    if relation.rounds() != 0 {
        return Err(ProofVerifyError::Malformed(format!(
            "{:?} on node {} is not a tie: it has {} rounds",
            relation.id(),
            relation.node().idx,
            relation.rounds()
        )));
    }
    let setup = relation.setup(&*accumulator, transcript)?;
    let points = relation.opening_points(&setup, &[])?;
    for (id, point) in relation.produced().into_iter().zip(points) {
        match id.polynomial {
            PolynomialId::Virtual(_) => accumulator.append_virtual(
                transcript,
                id,
                OpeningPoint::<BIG_ENDIAN, F>::new(point),
            )?,
            PolynomialId::Committed(_) => accumulator.append_dense(transcript, id, point)?,
        }
    }
    let input = relation_input_claim(relation, &setup, &*accumulator)?;
    let output = evaluate_expression(
        relation.output_expression::<F>(),
        &*accumulator,
        |challenge| setup.challenge(challenge),
        |derived| relation.derive(derived, &setup, &[]),
    )?;
    if input != output {
        return Err(ProofVerifyError::InvalidOpeningProof(format!(
            "{:?} tie on node {} does not hold",
            relation.id(),
            relation.node().idx
        )));
    }
    Ok(())
}

/// An op relation's expression: integer coefficients, read by any field
/// ([`Expr::cast`](atlas_claims::Expr::cast)).
pub type OpExpr = Expr<Int, RelationOpening, RelationDerived, RelationChallenge>;

/// Which op relation an [`OpRel`] is: the prover's dispatch key, and the declaration of the
/// ops the relation may serve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpRelKind {
    /// `And`: the product of two booleans.
    And,
    /// `Mul`: `left·right`.
    Mul,
    /// `Square`: `x²`.
    Square,
    /// `Cube`: `x³`.
    Cube,
    /// `Add`: the output is `left + right`.
    AddAccumulation,
    /// `Sub`: the output is `left − right`.
    SubAccumulation,
}

impl OpRelKind {
    /// The name the relation is registered under.
    pub fn relation_id(self) -> RelationId {
        match self {
            Self::And | Self::Mul | Self::Square | Self::Cube => RelationId::Product,
            Self::AddAccumulation | Self::SubAccumulation => RelationId::Accumulation,
        }
    }

    /// Whether a relation of this kind may be built for a node of operator `op`.
    pub fn serves(self, op: &atlas_trace::Op) -> bool {
        use atlas_trace::Op;
        match self {
            Self::And => matches!(op, Op::And(_)),
            Self::Mul => matches!(op, Op::Mul(_)),
            Self::Square => matches!(op, Op::Square(_)),
            Self::Cube => matches!(op, Op::Cube(_)),
            Self::AddAccumulation => matches!(op, Op::Add(_)),
            Self::SubAccumulation => matches!(op, Op::Sub(_)),
        }
    }
}

/// Where a relation's incoming point comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Incoming {
    /// The node's reduced output point.
    ReducedOutput,
    /// `rounds` challenges drawn from the transcript.
    Fresh,
}

/// An advice opening a relation's setup appends after its draws, at coordinates `range` of the
/// incoming point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetupOpen {
    /// The opening appended.
    pub id: OpeningId,
    /// The coordinates of the incoming point it is opened at.
    pub range: std::ops::Range<usize>,
}

/// The opening points of a relation with its own placement rule.
pub type PointsFn<F> =
    fn(&OpRel<F>, &RelationSetup<F>, &[F]) -> Result<Vec<Vec<F>>, ProofVerifyError>;
/// The derived scalars of a relation with its own.
pub type DeriveFn<F> =
    fn(&OpRel<F>, &RelationDerived, &RelationSetup<F>, &[F]) -> Result<F, ProofVerifyError>;
/// Further setup entries computed from the drawn ones and from held claims.
pub type FinishSetupFn<F> = fn(
    &OpRel<F>,
    &dyn OpeningAccumulator<F>,
    &mut RelationSetup<F>,
) -> Result<(), ProofVerifyError>;

/// What a relation computes that is not the default. Each member is a function, never a
/// closure: a relation captures no state.
pub struct Custom<F: JoltField> {
    /// The point of each produced opening; the default is the low-to-high bound point for
    /// all, which for a tie is the incoming point.
    pub points: Option<PointsFn<F>>,
    /// A derived scalar of the output expression; the default derives
    /// [`RelationDerived::Eq`] only.
    pub derive: Option<DeriveFn<F>>,
    /// Entries added to the setup after the draws. It has no transcript.
    pub finish_setup: Option<FinishSetupFn<F>>,
}

impl<F: JoltField> Clone for Custom<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: JoltField> Copy for Custom<F> {}

impl<F: JoltField> Default for Custom<F> {
    fn default() -> Self {
        Self {
            points: None,
            derive: None,
            finish_setup: None,
        }
    }
}

/// An op's protocol as a value: the six facts of a relation (rounds, degree, the input and
/// output expressions, the produced openings, the incoming point) and what its setup does.
/// Each op builds its own with a constructor function, which documents the S ⊆ R argument:
/// every execution of the op's semantics (`atlas_trace::semantics`) satisfies the relation.
#[derive(Clone)]
pub struct OpRel<F: JoltField> {
    /// Which relation this is.
    pub kind: OpRelKind,
    /// The node proven.
    pub node: ComputationNode,
    /// The sumcheck's round count; zero for a tie.
    pub rounds: usize,
    /// The per-round degree bound.
    pub degree: usize,
    /// The claimed sum, over openings held before the sumcheck and the setup's coefficients.
    pub input: OpExpr,
    /// The expected final claim, over the produced openings.
    pub output: OpExpr,
    /// The openings produced, in append order.
    pub produced: Vec<OpeningId>,
    /// Where the incoming point comes from.
    pub incoming: Incoming,
    /// The challenges the setup draws after the incoming point, one scalar each, in order.
    pub draws: Vec<RelationChallenge>,
    /// The advice openings the setup appends after the draws.
    pub setup_opens: Vec<SetupOpen>,
    /// The order a dense relation binds its variables in (an eq-weighted one binds
    /// low-to-high).
    pub binding: BindingOrder,
    /// What is not the default.
    pub custom: Custom<F>,
}

impl<F: JoltField> OpRel<F> {
    /// A relation of `kind` on `node` with no rounds, no openings and the default setup: the
    /// base a constructor fills in.
    pub fn base(kind: OpRelKind, node: &ComputationNode) -> Self {
        Self {
            kind,
            node: node.clone(),
            rounds: 0,
            degree: 0,
            input: OpExpr::zero(),
            output: OpExpr::zero(),
            produced: Vec::new(),
            incoming: Incoming::ReducedOutput,
            draws: Vec::new(),
            setup_opens: Vec::new(),
            binding: BindingOrder::LowToHigh,
            custom: Custom::default(),
        }
    }
}

impl<F: JoltField> SymbolicSumcheck for OpRel<F> {
    atlas_sumcheck::relation_types!();

    fn id(&self) -> RelationId {
        self.kind.relation_id()
    }

    fn rounds(&self) -> usize {
        self.rounds
    }

    fn degree(&self) -> usize {
        self.degree
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        self.input.cast()
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        self.output.cast()
    }
}

impl<F: JoltField> Produces for OpRel<F> {
    fn produced(&self) -> Vec<OpeningId> {
        self.produced.clone()
    }
}

impl<F: JoltField> Relation<F> for OpRel<F> {
    fn opening_points(
        &self,
        setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Result<Vec<Vec<F>>, ProofVerifyError> {
        if let Some(points) = self.custom.points {
            return points(self, setup, challenges);
        }
        let point = if self.rounds == 0 {
            setup.incoming_point.clone()
        } else {
            low_to_high_point(challenges)
        };
        Ok(vec![point; self.produced.len()])
    }

    fn derive(
        &self,
        id: &RelationDerived,
        setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Result<F, ProofVerifyError> {
        match (self.custom.derive, id) {
            (Some(derive), _) => derive(self, id, setup, challenges),
            (None, RelationDerived::Eq) => Ok(atlas_algebra::poly::eq::EqPolynomial::mle(
                &setup.incoming_point,
                &low_to_high_point(challenges),
            )),
            (None, other) => Err(not_derived(other)),
        }
    }
}

impl<F: JoltField> SetupRelation<F> for OpRel<F> {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<RelationSetup<F>, ProofVerifyError> {
        let incoming_point = match self.incoming {
            Incoming::ReducedOutput => accumulator.get_node_output_opening(self.node.idx)?.0.r,
            Incoming::Fresh => transcript
                .challenge_vector_optimized::<F>(self.rounds)
                .into_opening(),
        };
        let challenges = self
            .draws
            .iter()
            .map(|id| (*id, transcript.challenge_scalar()))
            .collect();
        let mut setup = RelationSetup {
            incoming_point,
            challenges,
        };
        if let Some(finish_setup) = self.custom.finish_setup {
            finish_setup(self, accumulator, &mut setup)?;
        }
        Ok(setup)
    }
}

impl<F: JoltField> EngineRelation<F> for OpRel<F> {
    fn binding_order(&self) -> BindingOrder {
        self.binding
    }
}

impl<F: JoltField> TieRelation<F> for OpRel<F> {}
