//! Op protocols written as symbolic relations ([`jolt_claims::SymbolicSumcheck`]) and the engine
//! that proves and verifies them.
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
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialEvaluation},
        opening_point::{OpeningPoint, BIG_ENDIAN},
    },
    utils::errors::ProofVerifyError,
};
use atlas_sumcheck::{
    opening_proof::{OpeningAccumulator, OpeningId, PolynomialId, VerifierOpeningAccumulator},
    sumcheck_verifier::SumcheckInstanceVerifier,
};
use atlas_trace::{op::OpSpec, Trace};
use atlas_transcript::Transcript;
use jolt_claims::{Expr, SymbolicSumcheck};

/// An opening a relation reads or produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RelationOpening {
    /// An opening held in the accumulator under its id.
    Id(OpeningId),
    /// A node output's reduced opening: the instance the evaluation reduction produced from
    /// every consumer's claim on that output.
    ReducedNodeOutput(usize),
}

/// A verifier-computed scalar in a relation's output expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelationDerived {
    /// `eq(r₀, r′)` at the low-to-high bound point: the engine-owned factor of every output term
    /// of an [`EngineRelation`].
    Eq,
    /// The batch eq factor of an einsum contraction layout (one when it has none).
    ContractionEq,
    /// The MLE of public data (a model constant, a public input) at the incoming point.
    Public,
    /// The MLE of a broadcast's all-ones replication tensor at its replicated coordinates.
    BroadcastOnes,
    /// The MLE of the relation's public table `i` ([`AtlasRelation::table`]) at the
    /// low-to-high bound point.
    Table(usize),
    /// `eq(r₀, r′_retained)`: eq between the incoming point and the leading (retained)
    /// coordinates of a high-to-low reduction's bound point.
    RetainedEq,
}

/// A Fiat-Shamir challenge a relation draws in its [`AtlasRelation::setup`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelationChallenge {
    /// The coefficient folding two identities into one sumcheck.
    Gamma,
}

/// The relation names op protocols are registered under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelationId {
    /// Element-wise product of operands (`Mul`, `Square`, `Cube`, and `And` on booleans).
    Product,
    /// The fused reciprocal square root.
    Rsqrt,
    /// An einsum contraction.
    Contraction,
    /// Floor division `left = right·q + rem`.
    Division,
    /// A committed division quotient equal to the node's reduced output.
    DivisionQuotient,
    /// Selection `mask·a + (1 − mask)·b`.
    Select,
    /// The pre-clamp accumulation of a saturating add or subtract, `acc = left ± right`.
    Accumulation,
    /// An output equal to `±` its operand at the same point (`Identity`, `Neg`).
    SignedCopy,
    /// An output that is zero everywhere (`IsNan` on finite fixed-point data).
    ZeroOutput,
    /// An output equal to public data (`Constant`, `Input`).
    PublicValue,
    /// An output that permutes its operand's axes.
    AxisPermutation,
    /// An output replicating its operand along broadcast axes.
    Broadcast,
    /// Division by a constant `left = d·q + rem`.
    ConstantDivision,
    /// A reshape: the operand's elements at new padded positions.
    Reshape,
    /// A slice: a window of the operand.
    Slice,
    /// A concatenation of operands along an axis.
    Concat,
    /// A sum along one axis.
    AxisSum,
    /// The mean of squares along the reduced axes, rescaled.
    MeanOfSquares,
}

/// A relation's expression over atlas openings.
pub type RelationExpr<F> = Expr<F, RelationOpening, RelationDerived, RelationChallenge>;

/// What a relation's setup fixes before its sumcheck: the incoming point and the challenges.
#[derive(Clone)]
pub struct RelationSetup<F> {
    /// The incoming point `r₀` (big-endian).
    pub incoming_point: Vec<F>,
    /// The drawn challenges, by id.
    pub challenges: Vec<(RelationChallenge, F)>,
}

impl<F: Copy> RelationSetup<F> {
    /// A drawn challenge's value.
    pub fn challenge(&self, id: &RelationChallenge) -> F {
        self.challenges
            .iter()
            .find(|(drawn, _)| drawn == id)
            .unwrap_or_else(|| panic!("challenge {id:?} was not drawn"))
            .1
    }
}

/// The big-endian point a low-to-high sumcheck binds: its challenges, last first.
pub fn low_to_high_point<F: Copy>(challenges: &[F]) -> Vec<F> {
    challenges.iter().rev().copied().collect()
}

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

    /// The incoming point and challenges, in protocol order: read from the accumulator, drawn
    /// from the transcript, or both. Prover and verifier run it at the same transcript state.
    fn setup<F: JoltField, T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut T,
    ) -> RelationSetup<F>;

    /// The openings the relation produces, in append (Fiat-Shamir) order. Every one appears in
    /// the output expression; an opening may repeat (an op whose operands are one node).
    fn produced(&self) -> Vec<OpeningId>;

    /// The big-endian point of each produced opening, aligned with [`Self::produced`], given the
    /// sumcheck challenges in round order. Defaults to the low-to-high bound point for all, which
    /// for a zero-round relation (a tie) is the incoming point.
    fn opening_points<F: JoltField>(
        &self,
        setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Vec<Vec<F>> {
        let point = if self.rounds() == 0 {
            setup.incoming_point.clone()
        } else {
            low_to_high_point(challenges)
        };
        vec![point; self.produced().len()]
    }

    /// A derived scalar of the output expression, given the sumcheck challenges in round order.
    fn derive<F: JoltField>(
        &self,
        id: &RelationDerived,
        setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> F {
        match id {
            RelationDerived::Eq => {
                EqPolynomial::mle(&setup.incoming_point, &low_to_high_point(challenges))
            }
            RelationDerived::Table(index) => MultilinearPolynomial::from(self.table(*index, setup))
                .evaluate(&low_to_high_point(challenges)),
            other => panic!("{other:?} is not derived by this relation"),
        }
    }

    /// Public table `index`, over the sumcheck's hypercube: data both sides build from the
    /// setup (a selector placing eq evaluations at the cells an op moves), read by
    /// [`RelationDerived::Table`].
    fn table<F: JoltField>(&self, index: usize, _setup: &RelationSetup<F>) -> Vec<F> {
        panic!("this relation has no table {index}")
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
pub trait EngineRelation: AtlasRelation {
    /// The witness behind each produced opening, aligned with [`AtlasRelation::produced`]: a
    /// polynomial over the sumcheck's hypercube whose final bound value is the opening's claim.
    fn witnesses<F: JoltField>(
        &self,
        trace: &Trace,
        setup: &RelationSetup<F>,
    ) -> Vec<MultilinearPolynomial<F>>;

    /// The order a dense relation binds its variables in (an eq-weighted one binds
    /// low-to-high).
    fn binding_order(&self) -> BindingOrder {
        BindingOrder::LowToHigh
    }
}

/// The setup of a relation whose incoming point is `node`'s reduced output point, with no
/// challenges.
pub fn reduced_output_setup<F: JoltField>(
    accumulator: &dyn OpeningAccumulator<F>,
    node: &ComputationNode,
) -> RelationSetup<F> {
    RelationSetup {
        incoming_point: accumulator.get_node_output_opening(node.idx).0.r,
        challenges: Vec::new(),
    }
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
        type Challenges<F> = jolt_claims::NoChallenges<F>;
        type Inputs<C> = jolt_claims::NoInputs<C>;
        type Outputs<C> = jolt_claims::NoOutputs<C>;
    };
}

/// The value of an opening the relation reads, from the accumulator.
fn resolve<F: JoltField>(accumulator: &dyn OpeningAccumulator<F>, opening: &RelationOpening) -> F {
    match *opening {
        RelationOpening::Id(id) => match id.polynomial {
            PolynomialId::Virtual(_) => accumulator.get_virtual_polynomial_opening(id).1,
            PolynomialId::Committed(_) => accumulator.get_committed_polynomial_opening(id).1,
        },
        RelationOpening::ReducedNodeOutput(node) => accumulator.get_node_output_opening(node).1,
    }
}

/// The relation's input claim: its input expression over the accumulator's openings.
pub fn relation_input_claim<F: JoltField, R: AtlasRelation>(
    relation: &R,
    setup: &RelationSetup<F>,
    accumulator: &dyn OpeningAccumulator<F>,
) -> F {
    relation.input_expression::<F>().evaluate(
        |opening| resolve(accumulator, opening),
        |challenge| setup.challenge(challenge),
        |derived| panic!("the input claim is taken before binding: no {derived:?} term"),
    )
}

/// A zero-round relation (a tie): its produced openings sit at points derived from the
/// incoming point, and the verifier checks the input claim against the output expression there.
pub trait TieRelation: AtlasRelation {
    /// The prover's values of the produced openings, aligned with [`AtlasRelation::produced`]
    /// at their [`opening_points`](AtlasRelation::opening_points): read off the trace, or taken
    /// from claims the accumulator already holds where the tie fixes them.
    fn claims<F: JoltField>(
        &self,
        trace: &Trace,
        setup: &RelationSetup<F>,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F>;
}

/// Verify a zero-round relation (a tie): append its produced openings at the incoming point and
/// require the input claim to equal the output expression.
pub fn verify_tie<F: JoltField, T: Transcript, R: AtlasRelation>(
    relation: &R,
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<(), ProofVerifyError> {
    assert_eq!(relation.rounds(), 0, "a tie has no rounds");
    let setup = relation.setup(&*accumulator, transcript);
    let points = relation.opening_points(&setup, &[]);
    for (id, point) in relation.produced().into_iter().zip(points) {
        match id.polynomial {
            PolynomialId::Virtual(_) => accumulator.append_virtual(
                transcript,
                id,
                OpeningPoint::<BIG_ENDIAN, F>::new(point),
            ),
            PolynomialId::Committed(_) => accumulator.append_dense(transcript, id, point),
        }
    }
    let input = relation_input_claim(relation, &setup, &*accumulator);
    let output = relation.output_expression::<F>().evaluate(
        |opening| resolve(&*accumulator, opening),
        |challenge| setup.challenge(challenge),
        |derived| relation.derive(derived, &setup, &[]),
    );
    if input != output {
        return Err(ProofVerifyError::InvalidOpeningProof(format!(
            "{:?} tie on node {} does not hold",
            R::id(),
            relation.node().idx
        )));
    }
    Ok(())
}

/// A relation with its setup done: what both sides start from.
pub struct RelationParams<F: JoltField, R: AtlasRelation> {
    relation: R,
    setup: RelationSetup<F>,
}

impl<F: JoltField, R: AtlasRelation> RelationParams<F, R> {
    /// `relation`, set up against the accumulator and transcript.
    pub fn new<T: Transcript>(
        relation: R,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        let setup = relation.setup(accumulator, transcript);
        Self::at(relation, setup)
    }

    /// The relation with an explicit setup.
    pub fn at(relation: R, setup: RelationSetup<F>) -> Self {
        Self { relation, setup }
    }

    /// The relation and its setup.
    pub fn into_parts(self) -> (R, RelationSetup<F>) {
        (self.relation, self.setup)
    }
}

/// The verifier of a relation: its output expression at the bound point.
pub struct RelationVerifier<F: JoltField, R: AtlasRelation> {
    relation: R,
    setup: RelationSetup<F>,
}

impl<F: JoltField, R: AtlasRelation> RelationVerifier<F, R> {
    /// The verifier for `relation`, set up against the accumulator and transcript.
    pub fn new<T: Transcript>(
        relation: R,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        let (relation, setup) = RelationParams::new(relation, accumulator, transcript).into_parts();
        Self { relation, setup }
    }
}

impl<F: JoltField, T: Transcript, R: AtlasRelation> SumcheckInstanceVerifier<F, T>
    for RelationVerifier<F, R>
{
    fn degree(&self) -> usize {
        self.relation.degree()
    }

    fn num_rounds(&self) -> usize {
        self.relation.rounds()
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        relation_input_claim(&self.relation, &self.setup, accumulator)
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let challenges: Vec<F> = sumcheck_challenges.to_vec().into_opening();
        self.relation.output_expression::<F>().evaluate(
            |opening| resolve(accumulator, opening),
            |challenge| self.setup.challenge(challenge),
            |derived| self.relation.derive(derived, &self.setup, &challenges),
        )
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let points = self
            .relation
            .opening_points(&self.setup, &sumcheck_challenges.to_vec().into_opening());
        for (id, point) in self.relation.produced().into_iter().zip(points) {
            match id.polynomial {
                PolynomialId::Virtual(_) => accumulator.append_virtual(
                    transcript,
                    id,
                    OpeningPoint::<BIG_ENDIAN, F>::new(point),
                ),
                PolynomialId::Committed(_) => accumulator.append_dense(transcript, id, point),
            }
        }
    }
}
