//! Sumcheck instances written as symbolic relations ([`jolt_claims::SymbolicSumcheck`]) over atlas
//! openings, and the verifier derived from them.
//!
//! A relation declares its **input-claim expression** (the claimed sum, from openings at the
//! incoming point and the setup's coefficients) and its **output expression** (the expected final
//! claim once every round is bound), a sum of products of produced openings with constant,
//! challenge or derived coefficients. It also says where each produced opening sits given the
//! sumcheck challenges. The verifier of a relation is [`RelationVerifier`]; no hand-written
//! verifier restates the algebra.
//!
//! A relation's [`RelationSetup`] carries every field value the algebra reads that is not an
//! opening: the incoming point and the coefficients. An op relation draws it itself
//! ([`SetupRelation`]); a subprotocol relation (a one-hot check) is handed it by its caller, which
//! drew the points and batching coefficients for the protocol it serves.
//!
//! The claim store is the atlas accumulator: a relation's openings are accumulator
//! [`OpeningId`]s or a node output's reduced opening, the instance the multi-consumer evaluation
//! reduction produces for it.

/// The batched sumcheck over relations: the verifier's driver of the upstream engine.
pub mod batch;
/// The prover side: witnesses that compute their own rounds for a relation.
pub mod prover;

pub use batch::{verify_relation, verify_relation_batch, DynRelationVerifier};

use crate::opening_proof::{
    OpeningAccumulator, OpeningId, PolynomialId, SumcheckId, VerifierOpeningAccumulator,
};
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_point::{OpeningPoint, BIG_ENDIAN},
    },
    utils::errors::ProofVerifyError,
};
use atlas_transcript::Transcript;
use common::CommittedPoly;
use jolt_claims::{Expr, Source, SymbolicSumcheck};

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
    /// of an engine relation.
    Eq,
    /// The batch eq factor of an einsum contraction layout (one when it has none).
    ContractionEq,
    /// The MLE of public data (a model constant, a public input) at the incoming point.
    Public,
    /// The MLE of a broadcast's all-ones replication tensor at its replicated coordinates.
    BroadcastOnes,
    /// The MLE of the relation's public table `i` ([`Relation::table`]) at the low-to-high
    /// bound point.
    Table(usize),
    /// `eq(r₀, r′_retained)`: eq between the incoming point and the leading (retained)
    /// coordinates of a high-to-low reduction's bound point.
    RetainedEq,
    /// A one-hot check's eq factor over its address-then-cycle point (see the relation).
    OneHotEq,
    /// A one-hot check's eq factor over its cycle coordinates alone.
    CycleEq,
    /// The address index `Σ_j 2^j·r_j` at the bound address coordinates.
    AddressIndex,
    /// The identity polynomial `Σ_j 2^j·r_j` over a read-raf check's address point.
    Identity,
    /// A read-raf check's read-address claim at its bound address point.
    Raf,
    /// The MLE of verifier-known per-row data at the row (leading) coordinates of the bound
    /// point.
    RowPublic,
    /// The MLE of a public one-hot indicator at the bound point.
    Indicator,
}

/// A coefficient of a relation: a Fiat-Shamir challenge drawn in its setup, or a field value its
/// caller supplies with the setup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelationChallenge {
    /// The coefficient folding two identities into one sumcheck.
    Gamma,
    /// The `i`-th batching coefficient of a batched check.
    Batching(usize),
    /// The `i`-th coefficient of a second batched check sharing the sumcheck.
    SecondBatching(usize),
    /// The `i`-th weight of a linear term.
    Weight(usize),
    /// The constant of a claimed sum.
    ClaimConstant,
    /// The coefficient of the `i`-th opening in a claimed sum.
    ClaimCoefficient(usize),
    /// A claimed sum supplied as a value.
    Claim,
    /// A second claim supplied as a value.
    SecondClaim,
}

/// The names relations are registered under.
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
    /// One-hot chunks are boolean (with an optional linear check sharing their opening).
    Booleanity,
    /// Hamming-weight indicators are boolean.
    HammingBooleanity,
    /// One-hot chunks have Hamming weight one.
    HammingWeight,
    /// A virtual read-address polynomial is the product of its one-hot chunks.
    RaVirtualization,
    /// A lookup into a small table: read value plus `γ`·read address.
    ReadRaf,
    /// A prefix-suffix lookup into a large table: read value plus `γ`·read address.
    PrefixSuffixReadRaf,
    /// A value is in range: its one-hot address decomposition recovers it.
    IdentityRangeCheck,
    /// A clamp bucket's split of its accumulations into output, slack and saturation indicator.
    ClampSplit,
    /// A packed bucket's values opened at one point.
    PackedValue,
    /// A lookup into a small public table (a teleported trig op, a clamped activation).
    SmallTable,
    /// A gather's read of its dictionary.
    Gather,
    /// Softmax: the output as the exponents times the rows' inverse sums.
    SoftmaxRecipProduct,
    /// Softmax: the row sums of the exponents.
    SoftmaxExpSum,
    /// Softmax: the row maxima read at the argmax.
    SoftmaxMax,
    /// Softmax: the exponent as the product of its two table digits.
    SoftmaxExpProduct,
    /// A group of committed openings reduced to one claim for the joint opening proof.
    OpeningReduction,
}

/// A relation's expression over atlas openings.
pub type RelationExpr<F> = Expr<F, RelationOpening, RelationDerived, RelationChallenge>;

/// What a relation's setup fixes before its sumcheck: the incoming point and the coefficients.
#[derive(Clone)]
pub struct RelationSetup<F> {
    /// The incoming point `r₀` (big-endian).
    pub incoming_point: Vec<F>,
    /// The coefficients, by id.
    pub challenges: Vec<(RelationChallenge, F)>,
}

impl<F: Copy> RelationSetup<F> {
    /// A coefficient's value.
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

/// Committed openings appended together as one sparse one-hot group, at a point `address ‖ cycle`
/// whose first `address_vars` coordinates are the address.
pub struct SparseGroup {
    /// The group's polynomials, in produced order.
    pub polynomials: Vec<CommittedPoly>,
    /// The sumcheck the openings come from.
    pub sumcheck: SumcheckId,
    /// Address variables of the group's point.
    pub address_vars: usize,
}

/// Each group with its point: the point of its first produced opening (a group's openings share
/// one point). `Err` if a group has no point.
pub fn group_points<F>(
    groups: Vec<SparseGroup>,
    points: Vec<Vec<F>>,
) -> Result<Vec<(SparseGroup, Vec<F>)>, ProofVerifyError> {
    let mut points = points.into_iter();
    groups
        .into_iter()
        .map(|group| {
            let mut members = points.by_ref().take(group.polynomials.len());
            let point = members.next().ok_or_else(|| {
                ProofVerifyError::Malformed("a sparse group has no produced opening point".into())
            })?;
            members.for_each(drop);
            Ok((group, point))
        })
        .collect()
}

/// The error of a relation asked for a derived scalar its expressions do not name.
pub fn not_derived(id: &RelationDerived) -> ProofVerifyError {
    ProofVerifyError::Malformed(format!("{id:?} is not derived by this relation"))
}

/// The openings a relation produces: field-independent, so the relation's expressions name them.
pub trait Produces {
    /// The openings the relation produces, in append (Fiat-Shamir) order. Every one appears in
    /// the output expression; an opening may repeat (an op whose operands are one node).
    fn produced(&self) -> Vec<OpeningId>;
}

/// A sumcheck instance as a symbolic relation over atlas openings, over the field `F` (as
/// upstream's `ConcreteSumcheck<F>`: a relation may carry field data its derived values read).
pub trait Relation<F: JoltField>:
    SymbolicSumcheck<
        RelationId = RelationId,
        OpeningId = RelationOpening,
        DerivedId = RelationDerived,
        ChallengeId = RelationChallenge,
    > + Produces
    + Send
    + Sync
{
    /// The big-endian point of each produced opening, aligned with [`Self::produced`], given the
    /// sumcheck challenges in round order. Defaults to the low-to-high bound point for all, which
    /// for a zero-round relation (a tie) is the incoming point. `Err` if the setup or the
    /// challenges do not have the shape the relation places its openings by.
    fn opening_points(
        &self,
        setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Result<Vec<Vec<F>>, ProofVerifyError> {
        let point = if self.rounds() == 0 {
            setup.incoming_point.clone()
        } else {
            low_to_high_point(challenges)
        };
        Ok(vec![point; self.produced().len()])
    }

    /// The produced openings as sparse one-hot groups, in produced order, when they are appended
    /// that way; `None` appends them one by one.
    fn sparse_groups(&self) -> Option<Vec<SparseGroup>> {
        None
    }

    /// Append the produced openings at their points, in Fiat-Shamir order: one append per
    /// opening, or one per [`Self::sparse_groups`] group.
    fn append_produced_requests<T: Transcript>(
        &self,
        setup: &RelationSetup<F>,
        challenges: &[F],
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        let points = self.opening_points(setup, challenges)?;
        if let Some(groups) = self.sparse_groups() {
            for (group, point) in group_points(groups, points)? {
                accumulator.append_sparse(transcript, group.polynomials, group.sumcheck, point)?;
            }
            return Ok(());
        }
        for (id, point) in self.produced().into_iter().zip(points) {
            match id.polynomial {
                PolynomialId::Virtual(_) => accumulator.append_virtual(
                    transcript,
                    id,
                    OpeningPoint::<BIG_ENDIAN, F>::new(point),
                )?,
                PolynomialId::Committed(_) => accumulator.append_dense(transcript, id, point)?,
            }
        }
        Ok(())
    }

    /// A derived scalar of the output expression, given the sumcheck challenges in round order.
    /// `Err` for an id the relation does not derive, or one it cannot compute from this setup.
    fn derive(
        &self,
        id: &RelationDerived,
        setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Result<F, ProofVerifyError> {
        Ok(match id {
            RelationDerived::Eq => {
                EqPolynomial::mle(&setup.incoming_point, &low_to_high_point(challenges))
            }
            RelationDerived::Table(index) => {
                MultilinearPolynomial::from(self.table(*index, setup)?)
                    .evaluate(&low_to_high_point(challenges))
            }
            other => return Err(not_derived(other)),
        })
    }

    /// Public table `index`, over the sumcheck's hypercube: data both sides build from the
    /// setup (a selector placing eq evaluations at the cells an op moves), read by
    /// [`RelationDerived::Table`]. `Err` if the relation has no such table.
    fn table(&self, index: usize, _setup: &RelationSetup<F>) -> Result<Vec<F>, ProofVerifyError> {
        Err(ProofVerifyError::Malformed(format!(
            "this relation has no table {index}"
        )))
    }
}

/// A relation that draws its own setup.
pub trait SetupRelation<F: JoltField>: Relation<F> {
    /// The incoming point and challenges, in protocol order: read from the accumulator, drawn
    /// from the transcript, or both. Prover and verifier run it at the same transcript state.
    /// `Err` if the accumulator lacks a claim the setup reads.
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<RelationSetup<F>, ProofVerifyError>;
}

/// The value of an opening the relation reads, from the accumulator; `Err` if it holds none.
pub fn resolve<F: JoltField>(
    accumulator: &dyn OpeningAccumulator<F>,
    opening: &RelationOpening,
) -> Result<F, ProofVerifyError> {
    Ok(match *opening {
        RelationOpening::Id(id) => match id.polynomial {
            PolynomialId::Virtual(_) => accumulator.get_virtual_polynomial_opening(id)?.1,
            PolynomialId::Committed(_) => accumulator.get_committed_polynomial_opening(id)?.1,
        },
        RelationOpening::ReducedNodeOutput(node) => accumulator.get_node_output_opening(node)?.1,
    })
}

/// `expr` over the accumulator's openings: every opening it names is resolved and every derived
/// scalar it names is computed first (`Err` if an opening is missing or `derive` fails), then the
/// expression is evaluated.
pub fn evaluate_expression<F: JoltField>(
    expr: RelationExpr<F>,
    accumulator: &dyn OpeningAccumulator<F>,
    challenge: impl Fn(&RelationChallenge) -> F,
    derive: impl Fn(&RelationDerived) -> Result<F, ProofVerifyError>,
) -> Result<F, ProofVerifyError> {
    let mut resolved: Vec<(RelationOpening, F)> = Vec::new();
    let mut derived: Vec<(RelationDerived, F)> = Vec::new();
    for term in &expr.terms {
        for factor in &term.factors {
            match factor {
                Source::Opening(opening) => {
                    if !resolved.iter().any(|(id, _)| id == opening) {
                        resolved.push((*opening, resolve(accumulator, opening)?));
                    }
                }
                Source::Derived(id) => {
                    if !derived.iter().any(|(computed, _)| computed == id) {
                        derived.push((*id, derive(id)?));
                    }
                }
                Source::Challenge(_) => {}
            }
        }
    }
    Ok(expr.evaluate(
        |opening| {
            resolved
                .iter()
                .find(|(id, _)| id == opening)
                .map_or(F::zero(), |(_, value)| *value)
        },
        challenge,
        |id| {
            derived
                .iter()
                .find(|(computed, _)| computed == id)
                .map_or(F::zero(), |(_, value)| *value)
        },
    ))
}

/// The relation's input claim: its input expression over the accumulator's openings.
pub fn relation_input_claim<F: JoltField, R: Relation<F>>(
    relation: &R,
    setup: &RelationSetup<F>,
    accumulator: &dyn OpeningAccumulator<F>,
) -> Result<F, ProofVerifyError> {
    evaluate_expression(
        relation.input_expression::<F>(),
        accumulator,
        |challenge| setup.challenge(challenge),
        |derived| {
            Err(ProofVerifyError::Malformed(format!(
                "the input claim is taken before binding: no {derived:?} term"
            )))
        },
    )
}

/// A relation with its setup done: what both sides start from.
pub struct RelationParams<F: JoltField, R: Relation<F>> {
    relation: R,
    setup: RelationSetup<F>,
}

impl<F: JoltField, R: Relation<F>> RelationParams<F, R> {
    /// The relation with an explicit setup.
    pub fn at(relation: R, setup: RelationSetup<F>) -> Self {
        Self { relation, setup }
    }

    /// The relation.
    pub fn relation(&self) -> &R {
        &self.relation
    }

    /// The setup.
    pub fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    /// The relation and its setup.
    pub fn into_parts(self) -> (R, RelationSetup<F>) {
        (self.relation, self.setup)
    }
}

/// The verifier of a relation: its output expression at the bound point.
pub struct RelationVerifier<F: JoltField, R: Relation<F>> {
    relation: R,
    setup: RelationSetup<F>,
}

impl<F: JoltField, R: SetupRelation<F>> RelationVerifier<F, R> {
    /// The verifier for `relation`, set up against the accumulator and transcript.
    pub fn new<T: Transcript>(
        relation: R,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<Self, ProofVerifyError> {
        let setup = relation.setup(accumulator, transcript)?;
        Ok(Self { relation, setup })
    }
}

impl<F: JoltField, R: Relation<F>> RelationVerifier<F, R> {
    /// The verifier for `relation` with the setup its caller supplies.
    pub fn at(relation: R, setup: RelationSetup<F>) -> Self {
        Self { relation, setup }
    }
}

/// The [`SymbolicSumcheck`] items every atlas relation shares: the atlas id types and no
/// upstream claim structs (openings resolve through the accumulator).
#[macro_export]
macro_rules! relation_types {
    () => {
        type RelationId = $crate::relation::RelationId;
        type OpeningId = $crate::relation::RelationOpening;
        type DerivedId = $crate::relation::RelationDerived;
        type ChallengeId = $crate::relation::RelationChallenge;
        type Challenges<C> = jolt_claims::NoChallenges<C>;
        type Inputs<C> = jolt_claims::NoInputs<C>;
        type Outputs<C> = jolt_claims::NoOutputs<C>;
    };
}
