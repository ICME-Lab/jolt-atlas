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
use atlas_trace::{op::OpSpec, Trace};
use jolt_claims::{Expr, Source, SymbolicSumcheck};
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, PolynomialId, ProverOpeningAccumulator,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        split_eq_poly::SplitEqUnreducedFold,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
        upstream_sumcheck::round_check_failed,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

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
            other => panic!("{other:?} is not derived by this relation"),
        }
    }
}

/// A relation the engine proves: each output term is [`RelationDerived::Eq`] times at most three
/// produced openings at the default bound point, with constant or challenge coefficients.
pub trait EngineRelation: AtlasRelation {
    /// The witness behind each produced opening, aligned with [`AtlasRelation::produced`].
    fn witnesses<F: JoltField>(&self, trace: &Trace) -> Vec<MultilinearPolynomial<F>>;
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

/// Append the relation's produced openings, with their `claims`, at their points.
pub fn append_produced<F: JoltField, T: Transcript, R: AtlasRelation>(
    relation: &R,
    setup: &RelationSetup<F>,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
    sumcheck_challenges: &[F::Challenge],
    claims: &[F],
) {
    let points = relation.opening_points(setup, &sumcheck_challenges.to_vec().into_opening());
    for ((&id, point), &claim) in relation.produced().iter().zip(points).zip(claims) {
        match id.polynomial {
            PolynomialId::Virtual(_) => accumulator.append_virtual(
                transcript,
                id,
                OpeningPoint::<BIG_ENDIAN, F>::new(point),
                claim,
            ),
            PolynomialId::Committed(_) => accumulator.append_dense(transcript, id, point, claim),
        }
    }
}

/// Prove a zero-round relation (a tie): append its produced openings, read off its witnesses
/// at the incoming point. Nothing else enters the transcript; [`verify_tie`] checks the input
/// claim against the output expression.
pub fn prove_tie<F: JoltField, T: Transcript, R: EngineRelation>(
    relation: &R,
    trace: &Trace,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
) {
    assert_eq!(relation.rounds(), 0, "a tie has no rounds");
    note_dispatch(relation);
    let setup = relation.setup(&*accumulator, transcript);
    let points = relation.opening_points(&setup, &[]);
    let claims: Vec<F> = relation
        .witnesses::<F>(trace)
        .iter()
        .zip(&points)
        .map(|(witness, point)| witness.evaluate(point))
        .collect();
    append_produced(relation, &setup, accumulator, transcript, &[], &claims);
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

/// Tell the test suite which relation ran for a node.
pub(crate) fn note_dispatch<R: AtlasRelation>(relation: &R) {
    #[cfg(test)]
    crate::onnx_proof::testing::relations::record_dispatch::<R>(relation.node());
    #[cfg(not(test))]
    let _ = relation;
}

/// One output term with the eq factor taken out and its challenges folded into the
/// coefficient: `coefficient · Π wires`.
struct SummandTerm<F> {
    coefficient: F,
    unit: bool,
    wires: Vec<usize>,
}

impl<F: JoltField> SummandTerm<F> {
    fn scaled(&self, value: F) -> F {
        if self.unit {
            value
        } else {
            self.coefficient * value
        }
    }
}

/// The output expression as the prover's summand: each term must carry the eq factor exactly
/// once and otherwise multiply produced openings (indices into [`AtlasRelation::produced`]) and
/// drawn challenges.
fn compile_summand<F: JoltField>(
    expr: RelationExpr<F>,
    produced: &[OpeningId],
    setup: &RelationSetup<F>,
) -> Vec<SummandTerm<F>> {
    expr.terms
        .into_iter()
        .map(|term| {
            let mut coefficient = term.coefficient;
            let mut eq_factors = 0;
            let mut wires = Vec::new();
            for factor in term.factors {
                match factor {
                    Source::Derived(RelationDerived::Eq) => eq_factors += 1,
                    Source::Derived(other) => panic!("the engine derives only Eq, not {other:?}"),
                    Source::Challenge(challenge) => coefficient *= setup.challenge(&challenge),
                    Source::Opening(RelationOpening::Id(id)) => wires.push(
                        produced
                            .iter()
                            .position(|p| *p == id)
                            .unwrap_or_else(|| panic!("{id:?} is not a produced opening")),
                    ),
                    Source::Opening(RelationOpening::ReducedNodeOutput(node)) => {
                        panic!("node {node}'s reduced output is read, never produced")
                    }
                }
            }
            assert_eq!(eq_factors, 1, "each output term carries the eq factor once");
            assert!(wires.len() <= 3, "summand terms are at most cubic");
            SummandTerm {
                unit: coefficient == F::one(),
                coefficient,
                wires,
            }
        })
        .collect()
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

/// The engine's prover for an [`EngineRelation`]: its summand folded over the split eq.
pub struct RelationProver<F: JoltField, R: EngineRelation> {
    relation: R,
    setup: RelationSetup<F>,
    eq: GruenSplitEqPolynomial<F>,
    summand: Vec<SummandTerm<F>>,
    /// The most produced openings in one term: the summand's degree in each round variable,
    /// apart from eq.
    inner_degree: usize,
    wires: Vec<MultilinearPolynomial<F>>,
}

impl<F: JoltField, R: EngineRelation> RelationProver<F, R> {
    /// The prover for `params`, its witnesses read from `trace`.
    #[tracing::instrument(skip_all, name = "RelationProver::initialize")]
    pub fn initialize(trace: &Trace, params: RelationParams<F, R>) -> Self {
        let (relation, setup) = params.into_parts();
        note_dispatch(&relation);
        let produced = relation.produced();
        let summand = compile_summand(relation.output_expression::<F>(), &produced, &setup);
        let inner_degree = summand.iter().map(|t| t.wires.len()).max().unwrap_or(0);
        assert_eq!(
            relation.degree(),
            inner_degree.max(1) + 1,
            "an engine relation's degree is its summand's, eq included"
        );
        let wires = relation.witnesses(trace);
        assert_eq!(
            wires.len(),
            produced.len(),
            "one witness per produced opening"
        );
        Self {
            eq: GruenSplitEqPolynomial::new(
                &setup.incoming_point,
                jolt_poly::BindingOrder::LowToHigh,
            ),
            relation,
            setup,
            summand,
            inner_degree,
            wires,
        }
    }

    fn bind(&mut self, r: F) {
        let r_j = F::challenge_from_field(r);
        self.eq.bind(r);
        for wire in &mut self.wires {
            wire.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }
}

impl<F: JoltField, R: EngineRelation> ProveRounds<F> for RelationProver<F, R> {
    fn num_rounds(&self) -> usize {
        self.relation.rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        let Self {
            eq,
            summand,
            wires,
            inner_degree,
            ..
        } = self;
        let (summand, wires) = (&*summand, &*wires);
        let at = |g: usize, x: usize, wire: usize| wires[wire].get_bound_coeff(2 * g + x);
        let at_endpoint = |x: usize| {
            move |g: usize| -> F {
                summand
                    .iter()
                    .map(|term| {
                        term.scaled(
                            term.wires
                                .iter()
                                .fold(F::one(), |acc, &w| acc * at(g, x, w)),
                        )
                    })
                    .sum()
            }
        };
        let message = if *inner_degree <= 1 {
            // q linear: its value at 1 (and at 0 when the eq factor vanishes there).
            eq.gruen_round_deg_2(&at_endpoint(1), &at_endpoint(0), previous_claim)
        } else if *inner_degree == 2 {
            // q quadratic: its constant and X² coefficient.
            eq.gruen_round_deg_3(
                &|g| {
                    let mut constant = F::zero();
                    let mut quadratic = F::zero();
                    for term in summand.iter() {
                        match term.wires[..] {
                            [] => constant += term.coefficient,
                            [a] => constant += term.scaled(at(g, 0, a)),
                            [a, b] => {
                                let (a0, b0) = (at(g, 0, a), at(g, 0, b));
                                constant += term.scaled(a0 * b0);
                                quadratic += term.scaled((at(g, 1, a) - a0) * (at(g, 1, b) - b0));
                            }
                            _ => unreachable!("inner degree is at most two"),
                        }
                    }
                    [constant, quadratic]
                },
                &at_endpoint(1),
                previous_claim,
            )
        } else {
            // q cubic: its values at 1 and 2 and its X³ coefficient.
            eq.gruen_round_from_evals::<3>(
                &|g| {
                    let mut evals = [F::zero(); 3];
                    for term in summand.iter() {
                        let (mut one, mut two, mut slopes) = (F::one(), F::one(), F::one());
                        for &w in &term.wires {
                            let (p0, p1) = (at(g, 0, w), at(g, 1, w));
                            let slope = p1 - p0;
                            one *= p1;
                            two *= p1 + slope;
                            slopes *= slope;
                        }
                        evals[0] += term.scaled(one);
                        evals[1] += term.scaled(two);
                        if term.wires.len() == 3 {
                            evals[2] += term.scaled(slopes);
                        }
                    }
                    evals
                },
                &at_endpoint(0),
                previous_claim,
            )
        };
        message.map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField, T: Transcript, R: EngineRelation> SumcheckInstanceProver<F, T>
    for RelationProver<F, R>
{
    fn degree(&self) -> usize {
        self.relation.degree()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        relation_input_claim(&self.relation, &self.setup, accumulator)
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let claims: Vec<F> = self.wires.iter().map(|w| w.final_claim()).collect();
        append_produced(
            &self.relation,
            &self.setup,
            accumulator,
            transcript,
            sumcheck_challenges,
            &claims,
        );
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
