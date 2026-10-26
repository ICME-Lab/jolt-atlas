//! Op protocols written as symbolic relations ([`jolt_claims::SymbolicSumcheck`]) and the engine
//! that proves and verifies them.
//!
//! A relation declares, over atlas openings, its **input-claim expression** (the claimed sum,
//! from openings at the incoming point `r₀`) and its **output expression** (the expected final
//! claim at the bound point `r′`): `eq(r₀, r′)` times a sum of products of produced openings.
//! The engine owns the eq factor (an upstream Gruen split eq over `r₀`), binds every produced
//! opening's witness low-to-high, and appends the produced openings in the relation's declared
//! order. Prover and verifier both read the claim algebra from the relation; neither restates it.
//!
//! The claim store stays the atlas accumulator: a relation's openings are accumulator
//! [`OpeningId`]s or a node output's reduced opening, the instance the multi-consumer evaluation
//! reduction produces for it.

use crate::node_view::ComputationNode;
use atlas_trace::Trace;
use jolt_claims::{Expr, Source, SymbolicSumcheck};
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, PolynomialId, ProverOpeningAccumulator,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        split_eq_poly::SplitEqUnreducedFold,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
        upstream_sumcheck::round_check_failed,
    },
    transcripts::Transcript,
};
use std::marker::PhantomData;

/// An opening a relation reads or produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RelationOpening {
    /// An opening held in the accumulator under its id.
    Id(OpeningId),
    /// A node output's reduced opening: the instance the evaluation reduction produced from
    /// every consumer's claim on that output.
    ReducedNodeOutput(usize),
}

/// A verifier-computed scalar in a relation's expressions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelationDerived {
    /// `eq(r₀, r′)`, the engine-owned factor of every output term.
    Eq,
}

/// A relation's Fiat-Shamir challenges (none yet).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelationChallenge {}

/// The relation names an op protocol is registered under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelationId {
    /// Element-wise product (`Mul`, and `And` on booleans).
    Mul,
}

/// A relation's expression over atlas openings.
pub type RelationExpr<F> = Expr<F, RelationOpening, RelationDerived, RelationChallenge>;

/// An op protocol as a symbolic relation on one computation node.
pub trait AtlasRelation:
    SymbolicSumcheck<
        RelationId = RelationId,
        OpeningId = RelationOpening,
        DerivedId = RelationDerived,
        ChallengeId = RelationChallenge,
        Shape = ComputationNode,
    > + Send
    + Sync
{
    /// The incoming point `r₀` (big-endian) the engine's eq factor sits on.
    fn incoming_point<F: JoltField>(&self, accumulator: &dyn OpeningAccumulator<F>) -> Vec<F>;

    /// The openings the relation produces, in append (Fiat-Shamir) order. Every one appears in
    /// the output expression; an opening may repeat (an op whose operands are one node).
    fn produced(&self) -> Vec<OpeningId>;

    /// The witness behind each produced opening, aligned with [`Self::produced`].
    fn witnesses<F: JoltField>(&self, trace: &Trace) -> Vec<MultilinearPolynomial<F>>;
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
fn input_claim<F: JoltField, R: AtlasRelation>(
    relation: &R,
    accumulator: &dyn OpeningAccumulator<F>,
) -> F {
    relation.input_expression::<F>().evaluate(
        |opening| resolve(accumulator, opening),
        |challenge| match *challenge {},
        |derived| match derived {
            RelationDerived::Eq => panic!("the input claim is taken before binding: no eq term"),
        },
    )
}

/// The bound point `r′` (big-endian) of a low-to-high sumcheck.
fn bound_point<F: JoltField>(challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
    OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec().into_opening()).match_endianness()
}

/// One output term with the eq factor taken out: `coefficient · Π wires`.
struct SummandTerm<F> {
    coefficient: F,
    unit: bool,
    wires: Vec<usize>,
}

/// The output expression as the prover's summand: each term must carry the eq factor exactly
/// once and otherwise multiply produced openings (indices into [`AtlasRelation::produced`]).
fn compile_summand<F: JoltField>(
    expr: RelationExpr<F>,
    produced: &[OpeningId],
) -> Vec<SummandTerm<F>> {
    expr.terms
        .into_iter()
        .map(|term| {
            let mut eq_factors = 0;
            let mut wires = Vec::new();
            for factor in term.factors {
                match factor {
                    Source::Derived(RelationDerived::Eq) => eq_factors += 1,
                    Source::Opening(RelationOpening::Id(id)) => wires.push(
                        produced
                            .iter()
                            .position(|p| *p == id)
                            .unwrap_or_else(|| panic!("{id:?} is not a produced opening")),
                    ),
                    Source::Opening(RelationOpening::ReducedNodeOutput(node)) => {
                        panic!("node {node}'s reduced output is read, never produced")
                    }
                    Source::Challenge(challenge) => match challenge {},
                }
            }
            assert_eq!(eq_factors, 1, "each output term carries the eq factor once");
            assert!(wires.len() <= 2, "summand terms are at most quadratic");
            SummandTerm {
                unit: term.coefficient == F::one(),
                coefficient: term.coefficient,
                wires,
            }
        })
        .collect()
}

/// A relation with its incoming point read off the accumulator: what both sides start from.
pub struct RelationParams<F: JoltField, R: AtlasRelation> {
    relation: R,
    incoming_point: Vec<F>,
}

impl<F: JoltField, R: AtlasRelation> RelationParams<F, R> {
    /// The relation on `node`, at the incoming point the accumulator holds for it.
    pub fn new(node: ComputationNode, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let relation = R::new(node);
        let incoming_point = relation.incoming_point(accumulator);
        Self::at(relation, incoming_point)
    }

    /// The relation at an explicit incoming point.
    pub fn at(relation: R, incoming_point: Vec<F>) -> Self {
        Self {
            relation,
            incoming_point,
        }
    }
}

/// The prover side of a relation: its summand folded over the engine's split eq.
pub struct RelationProver<F: JoltField, R: AtlasRelation> {
    relation: R,
    produced: Vec<OpeningId>,
    eq: GruenSplitEqPolynomial<F>,
    summand: Vec<SummandTerm<F>>,
    wires: Vec<MultilinearPolynomial<F>>,
}

impl<F: JoltField, R: AtlasRelation> RelationProver<F, R> {
    /// The prover for `params`, its witnesses read from `trace`.
    #[tracing::instrument(skip_all, name = "RelationProver::initialize")]
    pub fn initialize(trace: &Trace, params: RelationParams<F, R>) -> Self {
        let RelationParams {
            relation,
            incoming_point,
        } = params;
        let produced = relation.produced();
        let summand = compile_summand(relation.output_expression::<F>(), &produced);
        let wires = relation.witnesses(trace);
        assert_eq!(
            wires.len(),
            produced.len(),
            "one witness per produced opening"
        );
        Self {
            eq: GruenSplitEqPolynomial::new(&incoming_point, jolt_poly::BindingOrder::LowToHigh),
            relation,
            produced,
            summand,
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

impl<F: JoltField, R: AtlasRelation> ProveRounds<F> for RelationProver<F, R> {
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
            eq, summand, wires, ..
        } = self;
        let scaled = |term: &SummandTerm<F>, value: F| {
            if term.unit {
                value
            } else {
                term.coefficient * value
            }
        };
        eq.gruen_round_deg_3(
            &|g| {
                let mut constant = F::zero();
                let mut quadratic = F::zero();
                for term in summand.iter() {
                    match term.wires[..] {
                        [] => constant += term.coefficient,
                        [a] => constant += scaled(term, wires[a].get_bound_coeff(2 * g)),
                        [a, b] => {
                            let a0 = wires[a].get_bound_coeff(2 * g);
                            let b0 = wires[b].get_bound_coeff(2 * g);
                            let a_slope = wires[a].get_bound_coeff(2 * g + 1) - a0;
                            let b_slope = wires[b].get_bound_coeff(2 * g + 1) - b0;
                            constant += scaled(term, a0 * b0);
                            quadratic += scaled(term, a_slope * b_slope);
                        }
                        _ => unreachable!("summand terms are at most quadratic"),
                    }
                }
                [constant, quadratic]
            },
            &|g| {
                summand
                    .iter()
                    .map(|term| {
                        let product = term.wires.iter().fold(F::one(), |acc, &w| {
                            acc * wires[w].get_bound_coeff(2 * g + 1)
                        });
                        term.coefficient * product
                    })
                    .sum()
            },
            previous_claim,
        )
        .map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField, T: Transcript, R: AtlasRelation> SumcheckInstanceProver<F, T>
    for RelationProver<F, R>
{
    fn degree(&self) -> usize {
        self.relation.degree()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        input_claim(&self.relation, accumulator)
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let point = bound_point::<F>(sumcheck_challenges);
        for (id, wire) in self.produced.iter().zip(&self.wires) {
            accumulator.append_virtual(transcript, *id, point.clone(), wire.final_claim());
        }
    }
}

/// The verifier side of a relation: its output expression at the bound point.
pub struct RelationVerifier<F: JoltField, R: AtlasRelation> {
    relation: R,
    incoming_point: Vec<F>,
    _field: PhantomData<F>,
}

impl<F: JoltField, R: AtlasRelation> RelationVerifier<F, R> {
    /// The verifier for the relation on `node`.
    pub fn new(node: ComputationNode, accumulator: &VerifierOpeningAccumulator<F>) -> Self {
        let RelationParams {
            relation,
            incoming_point,
        } = RelationParams::new(node, accumulator);
        Self {
            relation,
            incoming_point,
            _field: PhantomData,
        }
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
        input_claim(&self.relation, accumulator)
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let point = bound_point::<F>(sumcheck_challenges);
        let eq = EqPolynomial::mle(&self.incoming_point, &point.r);
        self.relation.output_expression::<F>().evaluate(
            |opening| resolve(accumulator, opening),
            |challenge| match *challenge {},
            |derived| match derived {
                RelationDerived::Eq => eq,
            },
        )
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let point = bound_point::<F>(sumcheck_challenges);
        for id in self.relation.produced() {
            accumulator.append_virtual(transcript, id, point.clone());
        }
    }
}
