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

use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_point::{OpeningPoint, BIG_ENDIAN},
        split_eq_poly::SplitEqUnreducedFold,
    },
};
use atlas_sumcheck::{
    opening_proof::{prover::ProverOpeningAccumulator, OpeningId, PolynomialId},
    sumcheck::prover::SumcheckInstanceProver,
    upstream_sumcheck::round_check_failed,
};
use atlas_trace::Trace;
use atlas_transcript::Transcript;
use common::parallel::par_enabled;
pub use jolt_atlas_verifier::onnx_proof::relation::*;
use jolt_claims::Source;
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rayon::prelude::*;

/// The [`SymbolicSumcheck`] items every atlas relation shares: the atlas id types and no
/// upstream claim structs (openings resolve through the accumulator).
#[macro_export]
macro_rules! atlas_relation_types {
    () => {
        type RelationId = $crate::onnx_proof::relation::RelationId;
        type OpeningId = $crate::onnx_proof::relation::RelationOpening;
        type DerivedId = $crate::onnx_proof::relation::RelationDerived;
        type ChallengeId = $crate::onnx_proof::relation::RelationChallenge;
        type Challenges<C> = jolt_claims::NoChallenges<C>;
        type Inputs<C> = jolt_claims::NoInputs<C>;
        type Outputs<C> = jolt_claims::NoOutputs<C>;
    };
}

/// Append the relation's produced openings, with their `claims`, at their points.
pub fn append_produced<F: JoltField, T: Transcript, R: AtlasRelation + Relation<F>>(
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

/// Prove a tie: append its produced openings with their [`TieRelation::claims`]. Nothing else
/// enters the transcript; [`verify_tie`] checks the input claim against the output expression.
pub fn prove_tie<F: JoltField, T: Transcript, R: TieRelation<F>>(
    relation: &R,
    trace: &Trace,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
) {
    assert_eq!(relation.rounds(), 0, "a tie has no rounds");
    note_dispatch(relation);
    let setup = relation.setup(&*accumulator, transcript);
    let claims = relation.claims(trace, &setup, &*accumulator);
    append_produced(relation, &setup, accumulator, transcript, &[], &claims);
}

/// Tell the test suite which relation ran for a node.
pub(crate) fn note_dispatch<R: AtlasRelation>(relation: &R) {
    #[cfg(test)]
    crate::onnx_proof::testing::relations::record_dispatch::<R>(relation.node());
    #[cfg(not(test))]
    let _ = relation;
}

/// The engine's prover for an [`EngineRelation`]: its summand folded over the split eq.
pub struct RelationProver<F: JoltField, R: EngineRelation<F>> {
    relation: R,
    setup: RelationSetup<F>,
    /// The Gruen split eq of an eq-weighted relation; `None` for a dense one.
    eq: Option<GruenSplitEqPolynomial<F>>,
    summand: Vec<SummandTerm<F>>,
    /// How many of the wires are produced openings' witnesses; the rest are public tables.
    witness_count: usize,
    /// The most produced openings in one term: the summand's degree in each round variable,
    /// apart from eq.
    inner_degree: usize,
    wires: Vec<MultilinearPolynomial<F>>,
}

impl<F: JoltField, R: EngineRelation<F>> RelationProver<F, R> {
    /// The prover for `params`, its witnesses read from `trace`.
    #[tracing::instrument(skip_all, name = "RelationProver::initialize")]
    pub fn initialize(trace: &Trace, params: RelationParams<F, R>) -> Self {
        let (relation, setup) = params.into_parts();
        note_dispatch(&relation);
        let produced = relation.produced();
        let mut tables = Vec::new();
        let (summand, eq_weighted) = compile_summand(
            relation.output_expression::<F>(),
            &produced,
            &mut tables,
            &setup,
        );
        let inner_degree = summand.iter().map(|t| t.wires.len()).max().unwrap_or(0);
        let expected_degree = if eq_weighted {
            inner_degree.max(1) + 1
        } else {
            inner_degree
        };
        assert_eq!(
            relation.degree(),
            expected_degree,
            "an engine relation's degree is its summand's"
        );
        let mut wires = relation.witnesses(trace, &setup);
        assert_eq!(
            wires.len(),
            produced.len(),
            "one witness per produced opening"
        );
        wires.extend(
            tables
                .iter()
                .map(|&index| MultilinearPolynomial::from(relation.table(index, &setup))),
        );
        assert!(
            !eq_weighted || relation.binding_order() == BindingOrder::LowToHigh,
            "an eq-weighted relation binds low-to-high"
        );
        let eq = eq_weighted.then(|| {
            GruenSplitEqPolynomial::new(&setup.incoming_point, jolt_poly::BindingOrder::LowToHigh)
        });
        Self {
            eq,
            relation,
            setup,
            summand,
            witness_count: produced.len(),
            inner_degree,
            wires,
        }
    }

    fn bind(&mut self, r: F) {
        let r_j = F::challenge_from_field(r);
        if let Some(eq) = &mut self.eq {
            eq.bind(r);
        }
        let order = self.relation.binding_order();
        for wire in &mut self.wires {
            wire.bind_parallel(r_j, order);
        }
    }
}

impl<F: JoltField, R: EngineRelation<F>> ProveRounds<F> for RelationProver<F, R> {
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
        let Some(eq) = eq else {
            let order = self.relation.binding_order();
            return Ok(match self.relation.degree() {
                1 => dense_message::<F, 1>(summand, wires, order, previous_claim),
                2 => dense_message::<F, 2>(summand, wires, order, previous_claim),
                3 => dense_message::<F, 3>(summand, wires, order, previous_claim),
                degree => panic!("dense relations of degree {degree} are not supported"),
            });
        };
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

impl<F: JoltField, T: Transcript, R: EngineRelation<F>> SumcheckInstanceProver<F, T>
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
        let claims: Vec<F> = self.wires[..self.witness_count]
            .iter()
            .map(|w| w.final_claim())
            .collect();
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

/// One output term with the eq factor taken out and its challenges folded into the
/// coefficient: `coefficient · Π wires`. A wire indexes the produced openings' witnesses, then
/// the public tables after them.
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
    tables: &mut Vec<usize>,
    setup: &RelationSetup<F>,
) -> (Vec<SummandTerm<F>>, bool) {
    let mut eq_weighted = None;
    let terms = expr
        .terms
        .into_iter()
        .map(|term| {
            let mut coefficient = term.coefficient;
            let mut eq_factors = 0;
            let mut wires = Vec::new();
            for factor in term.factors {
                match factor {
                    Source::Derived(RelationDerived::Eq) => eq_factors += 1,
                    Source::Derived(RelationDerived::Table(index)) => {
                        let position =
                            tables.iter().position(|&t| t == index).unwrap_or_else(|| {
                                tables.push(index);
                                tables.len() - 1
                            });
                        wires.push(produced.len() + position);
                    }
                    Source::Derived(other) => panic!("the engine cannot derive {other:?}"),
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
            assert!(
                eq_factors <= 1,
                "an output term carries the eq factor at most once"
            );
            let term_eq_weighted = eq_factors == 1;
            assert_eq!(
                *eq_weighted.get_or_insert(term_eq_weighted),
                term_eq_weighted,
                "either every output term carries the eq factor or none does"
            );
            assert!(wires.len() <= 3, "summand terms are at most cubic");
            SummandTerm {
                unit: coefficient == F::one(),
                coefficient,
                wires,
            }
        })
        .collect();
    (terms, eq_weighted.unwrap_or(true))
}

/// A dense relation's round message: `Σ_i Σ_terms coefficient · Π wires(i, X)` at
/// `X = 0, 2, …, D`, interpolated with the claim as hint.
fn dense_message<F: JoltField, const D: usize>(
    summand: &[SummandTerm<F>],
    wires: &[MultilinearPolynomial<F>],
    order: BindingOrder,
    previous_claim: F,
) -> UnivariatePoly<F> {
    let half = wires[0].len() / 2;
    let evals: [F; D] = (0..half)
        .into_par_iter()
        .with_min_len(par_enabled())
        .map(|i| {
            let mut evals = [F::zero(); D];
            for term in summand {
                let mut product: Option<[F; D]> = None;
                for &w in &term.wires {
                    let wire = wires[w].sumcheck_evals_array::<D>(i, order);
                    product = Some(match product {
                        None => wire,
                        Some(p) => std::array::from_fn(|k| p[k] * wire[k]),
                    });
                }
                let product = product.unwrap_or([F::one(); D]);
                for k in 0..D {
                    evals[k] += term.scaled(product[k]);
                }
            }
            evals
        })
        .reduce(
            || [F::zero(); D],
            |a, b| std::array::from_fn(|k| a[k] + b[k]),
        );
    UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
}
