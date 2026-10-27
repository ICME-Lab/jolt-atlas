//! Op protocols written as symbolic relations ([`jolt_claims::SymbolicSumcheck`]) and the engine
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
use atlas_algebra::poly::multilinear_polynomial::MultilinearPolynomial;
use atlas_algebra::{
    field::JoltField,
    poly::{
        multilinear_polynomial::BindingOrder,
        opening_point::{OpeningPoint, BIG_ENDIAN},
    },
    utils::errors::ProofVerifyError,
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, PolynomialId, VerifierOpeningAccumulator};
pub use atlas_sumcheck::relation::*;
use atlas_trace::{op::OpSpec, Trace};
use atlas_transcript::Transcript;
use jolt_claims::SymbolicSumcheck;

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

/// A relation the engine proves, binding low-to-high with its produced openings at the bound
/// point (or a prefix of it), in one of two shapes:
///
/// * **eq-weighted**: each output term is [`RelationDerived::Eq`] times at most three produced
///   openings, with constant or challenge coefficients; the engine folds over a Gruen split eq;
/// * **dense**: no term has an eq factor; each term multiplies produced openings and public
///   tables ([`RelationDerived::Table`]), and the engine sends the message's evaluations at
///   `0, 2, …, degree` with the claim as hint.
pub trait EngineRelation<F: JoltField>: AtlasRelation + SetupRelation<F> {
    /// The witness behind each produced opening, aligned with [`AtlasRelation::produced`]: a
    /// polynomial over the sumcheck's hypercube whose final bound value is the opening's claim.
    fn witnesses(&self, trace: &Trace, setup: &RelationSetup<F>) -> Vec<MultilinearPolynomial<F>>;

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
        type Challenges<C> = jolt_claims::NoChallenges<C>;
        type Inputs<C> = jolt_claims::NoInputs<C>;
        type Outputs<C> = jolt_claims::NoOutputs<C>;
    };
}

/// A zero-round relation (a tie): its produced openings sit at points derived from the
/// incoming point, and the verifier checks the input claim against the output expression there.
pub trait TieRelation<F: JoltField>: AtlasRelation + SetupRelation<F> {
    /// The prover's values of the produced openings, aligned with [`AtlasRelation::produced`]
    /// at their [`opening_points`](AtlasRelation::opening_points): read off the trace, or taken
    /// from claims the accumulator already holds where the tie fixes them.
    fn claims(
        &self,
        trace: &Trace,
        setup: &RelationSetup<F>,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F>;
}

/// Verify a zero-round relation (a tie): append its produced openings at the incoming point and
/// require the input claim to equal the output expression.
pub fn verify_tie<F: JoltField, T: Transcript, R: AtlasRelation + SetupRelation<F>>(
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
