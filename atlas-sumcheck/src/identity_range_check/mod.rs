use crate::{
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId, VerifierOpeningAccumulator},
    ps_shout::CycleWeight,
    relation::{
        Produces, Relation, RelationChallenge, RelationDerived, RelationExpr, RelationId,
        RelationOpening, RelationSetup, RelationVerifier,
    },
    sumcheck_verifier::SumcheckInstanceParams,
};
use atlas_algebra::{
    field::JoltField,
    poly::{
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::PolynomialEvaluation,
        opening_point::{OpeningPoint, BIG_ENDIAN},
    },
};
use common::VirtualPoly;
use jolt_claims::{challenge, derived, opening, SymbolicSumcheck};
use jolt_field::Ring;
use rayon::prelude::*;

const DEGREE_BOUND: usize = 2;

#[derive(Debug, Clone)]
/// Parameters for the Prefix suffix Read-raf checking sum-check protocol.
///
/// This protocol proves correct lookups from instruction tables by verifying that
/// read values match the table entries at computed addresses. The protocol uses
/// gamma batching to efficiently combine the lookup output with left/right operand
/// contributions and RAF values that depend on operand interleaving.
///
/// The sumcheck proceeds in two phases:
/// - Address phase (log K rounds): binds address variables using prefix-suffix decomposition
/// - Cycle phase (log T rounds): binds cycle variables and evaluates equality polynomials
pub struct IdentityRCParams<F>
where
    F: JoltField,
{
    /// log2(T): number of variables in the node output polynomial (cycle variables).
    pub log_T: usize,
    /// log2(K): number of variables in the address polynomial (address variables).
    pub log_K: usize,
    /// Per-cycle weight: `eq(r_node_output, ·)` for one node, or a packed
    /// multi-node weight (see [`CycleWeight`]).
    pub cycle: CycleWeight<F>,
    /// Polynomial types for opening accumulator (to cache ra claim)
    pub polynomial_type: VirtualPoly,
    /// Sumcheck ID for opening accumulator (to cache ra claim)
    pub sumcheck_id: SumcheckId,
    /// Input claim for the read-checking sum-check
    pub input_claim: F,
    /// Number of phases in the address rounds (first log K rounds).
    pub phases: usize,
}

impl<F> IdentityRCParams<F>
where
    F: JoltField,
{
    pub fn new(
        provider: &impl IdentityRCProvider<F>,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let r_node_output = provider.r_cycle(accumulator);
        let log_T = r_node_output.len();
        let (polynomial_type, sumcheck_id) = provider.ra_poly();
        Self {
            log_T,
            cycle: CycleWeight::Eq(r_node_output),
            polynomial_type,
            sumcheck_id,
            log_K: provider.log_K(),
            phases: provider.phases(),
            input_claim: provider.input_claim(accumulator),
        }
    }

    /// Params for an explicit [`CycleWeight`] and input claim (packed
    /// multi-node instances): `input_claim = Σ_n γ_n · claim_n`.
    pub fn from_parts_with_cycle(
        cycle: CycleWeight<F>,
        log_K: usize,
        ra_poly: (VirtualPoly, SumcheckId),
        input_claim: F,
    ) -> Self {
        Self {
            log_T: cycle.log_T(),
            cycle,
            polynomial_type: ra_poly.0,
            sumcheck_id: ra_poly.1,
            log_K,
            phases: default_phases(log_K),
            input_claim,
        }
    }
}

/// Address-phase count for a `log_K`-bit identity range check: `log_m = 4`
/// when possible, else `2` (the prefix/suffix MLE halving needs even chunks).
pub fn default_phases(log_K: usize) -> usize {
    if log_K <= 2 {
        1
    } else if log_K.is_multiple_of(4) {
        log_K / 4
    } else if log_K.is_multiple_of(2) {
        log_K / 2
    } else {
        panic!(
            "IdentityRCProvider: odd log_K={log_K} is not supported \
             (PrefixSuffix decomposition requires even chunk sizes)"
        )
    }
}

impl<F> SumcheckInstanceParams<F> for IdentityRCParams<F>
where
    F: JoltField,
{
    fn num_rounds(&self) -> usize {
        self.log_K + self.log_T
    }

    // output = eq_eval * ra_claim * identity_eval = Challenge(0) * Opening(ra)

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.input_claim
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let (r_address_prime, r_node_output_prime) = challenges.split_at(self.log_K);
        let r_node_output_prime = r_node_output_prime
            .iter()
            .copied()
            .rev()
            .collect::<Vec<_>>();
        OpeningPoint::new([r_address_prime.to_vec(), r_node_output_prime].concat())
    }
}

/// The identity range check as a relation: over the address-then-cycle hypercube,
///
/// ```text
/// input claim   the range-checked opening's claim
/// output claim  w(ρ_cycle) · ra(ρ) · id(ρ_address)
/// ```
///
/// with `w` the [`CycleWeight`], `id` the identity polynomial and `ρ = (ρ_address, ρ_cycle)` the
/// bound point (address in round order, cycle reversed). **S ⊆ R.** An honest `ra` is the one-hot
/// encoding of in-range values, so the sum over addresses recovers each value; an out-of-range
/// value has no address. The parameters carry the cycle weight; the setup carries the claim.
impl<F: JoltField> IdentityRCParams<F> {
    /// The bound point split into its address part (round order) and cycle part (reversed).
    fn split_point(&self, challenges: &[F]) -> (Vec<F>, Vec<F>) {
        let (address, cycle) = challenges.split_at(self.log_K);
        (address.to_vec(), cycle.iter().rev().copied().collect())
    }

    /// The relation these parameters instantiate, with its setup.
    pub fn into_verifier(self) -> IdentityRCVerifier<F> {
        let setup = RelationSetup {
            incoming_point: Vec::new(),
            challenges: vec![(RelationChallenge::Claim, self.input_claim)],
        };
        RelationVerifier::at(self, setup)
    }
}

impl<F: JoltField> SymbolicSumcheck for IdentityRCParams<F> {
    crate::relation_types!();
    type Shape = Self;

    fn new(shape: Self) -> Self {
        shape
    }

    fn id() -> RelationId {
        RelationId::IdentityRangeCheck
    }

    fn rounds(&self) -> usize {
        self.log_K + self.log_T
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        challenge(RelationChallenge::Claim)
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        derived::<G, _, _, _>(RelationDerived::CycleEq)
            * opening(RelationOpening::Id(self.produced()[0]))
            * derived(RelationDerived::Identity)
    }
}

impl<F: JoltField> Produces for IdentityRCParams<F> {
    fn produced(&self) -> Vec<OpeningId> {
        vec![OpeningId::new(self.polynomial_type, self.sumcheck_id)]
    }
}

impl<F: JoltField> Relation<F> for IdentityRCParams<F> {
    fn opening_points(&self, _setup: &RelationSetup<F>, challenges: &[F]) -> Vec<Vec<F>> {
        let (address, cycle) = self.split_point(challenges);
        vec![[address, cycle].concat()]
    }

    fn derive(&self, id: &RelationDerived, _setup: &RelationSetup<F>, challenges: &[F]) -> F {
        let (address, cycle) = self.split_point(challenges);
        match id {
            RelationDerived::CycleEq => self.cycle.mle(&cycle),
            RelationDerived::Identity => {
                IdentityPolynomial::<F>::new(self.log_K).evaluate(&address)
            }
            other => panic!("{other:?} is not derived by this relation"),
        }
    }
}

/// The verifier of an identity range check: its relation.
pub type IdentityRCVerifier<F> = RelationVerifier<F, IdentityRCParams<F>>;

#[derive(Debug, Clone, Copy)]
pub struct ReadRafClaims<F>
where
    F: JoltField,
{
    pub rv_claim: F,
    pub left_operand_claim: F,
    pub right_operand_claim: F,
}

/// Provider trait for identity-based range-check sumcheck instances.
pub trait IdentityRCProvider<F>
where
    F: JoltField,
{
    /// Returns the opening claim for the polynomial being range-checked.
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F;
    /// Returns the virtual polynomial and sumcheck id for the one-hot encoded read-address polynomial.
    fn ra_poly(&self) -> (VirtualPoly, SumcheckId);
    /// Returns challenge used in the prefix-suffix sum-check protocol
    fn r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F>;
    /// Returns log₂ of the range upper bound K.
    fn log_K(&self) -> usize;
    /// Number of phases for the range-check sumcheck.
    ///
    /// The prefix-suffix decomposition requires `log_m = log_K / phases` to be
    /// **even** (for `IdentityPolynomial` suffix/prefix MLE halving) and to
    /// divide `log_K` exactly.  The default picks `log_m = 4` when possible,
    /// falling back to `log_m = 2` for even `log_K` values that are not
    /// multiples of 4.
    fn phases(&self) -> usize {
        default_phases(self.log_K())
    }
}

/// Verifier counterpart of [`identity_rangecheck_prover_with_cycle`].
pub fn identity_rangecheck_verifier_with_cycle<F: JoltField>(
    cycle: CycleWeight<F>,
    log_K: usize,
    ra_poly: (VirtualPoly, SumcheckId),
    input_claim: F,
) -> IdentityRCVerifier<F> {
    IdentityRCParams::from_parts_with_cycle(cycle, log_K, ra_poly, input_claim).into_verifier()
}

pub fn identity_rangecheck_verifier<F: JoltField>(
    provider: &impl IdentityRCProvider<F>,
    accumulator: &mut VerifierOpeningAccumulator<F>,
) -> IdentityRCVerifier<F> {
    IdentityRCParams::new(provider, accumulator).into_verifier()
}

pub mod prover;
