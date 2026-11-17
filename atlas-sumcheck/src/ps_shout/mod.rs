use crate::relation::not_derived;
use crate::{
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId, VerifierOpeningAccumulator},
    relation::{
        Produces, Relation, RelationChallenge, RelationDerived, RelationExpr, RelationId,
        RelationOpening, RelationSetup, RelationVerifier,
    },
};
use atlas_algebra::field::Ring;
use atlas_algebra::poly::split_eq::GruenSplitEqPolynomial;
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_algebra::{
    field::JoltField,
    poly::{
        eq::EqPolynomial,
        multilinear::{MultilinearPolynomial, PolynomialEvaluation},
        point::{OpeningPoint, BIG_ENDIAN},
        signed_identity::SignedIdentityPoly,
    },
    utils::thread::unsafe_allocate_zero_vec,
};
use atlas_claims::{challenge, derived, opening, SymbolicSumcheck};
use atlas_lookup_tables::{JoltLookupTable, PrefixSuffixDecompositionTrait};
use atlas_transcript::Transcript;
use common::{parallel::par_enabled, VirtualPoly};
use rayon::prelude::*;

#[cfg(test)]
mod lookup_table_tests;
#[cfg(test)]
pub mod tests;

pub(crate) const DEGREE_BOUND: usize = 2;

/// Where a prefix-suffix lookup's read-address polynomial and cycle point live.
pub trait RafShoutProvider<F: JoltField> {
    fn ra_poly(&self) -> (VirtualPoly, SumcheckId);
    fn r_cycle(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<OpeningPoint<BIG_ENDIAN, F>, ProofVerifyError>;
}
pub(crate) const NUM_PHASES: usize = 8;

/// One node's slice of a packed (multi-node) cycle space: its opening point
/// `r` (log T_n variables) lives at prefix `prefix` (`prefix_len` high bits),
/// weighted by the batching coefficient `gamma`.
#[derive(Clone, Debug)]
pub struct PackedSegment<F: JoltField> {
    pub gamma: F,
    pub prefix: usize,
    pub prefix_len: usize,
    pub r: Vec<F>,
}

/// The per-cycle weight of a read-raf sumcheck: `Σ_t weight(t)·Σ_k ra(k,t)·(…)`.
///
/// A single node's lookup uses `eq(r_cycle, t)`. A *packed* instance covers
/// several nodes' lookups laid out in disjoint, power-of-two-aligned segments of
/// one cycle space, and weights segment `n` by `γ_n · eq(r_n, t − off_n)` — so
/// one sumcheck proves `Σ_n γ_n · (rv_n(r_n) + γ·raf_n(r_n))` for all of them.
#[derive(Clone, Debug)]
pub enum CycleWeight<F: JoltField> {
    /// `eq(r, t)`.
    Eq(OpeningPoint<BIG_ENDIAN, F>),
    /// `Σ_n γ_n · eq(prefix_n, t_high) · eq(r_n, t_low)`.
    Packed {
        log_T: usize,
        segments: Vec<PackedSegment<F>>,
    },
}

impl<F: JoltField> CycleWeight<F> {
    pub fn log_T(&self) -> usize {
        match self {
            Self::Eq(r) => r.len(),
            Self::Packed { log_T, .. } => *log_T,
        }
    }

    /// Dense weights over the cycle space.
    pub fn evals(&self) -> Vec<F> {
        match self {
            Self::Eq(r) => EqPolynomial::evals(&r.r),
            Self::Packed { log_T, segments } => {
                let mut w = unsafe_allocate_zero_vec(1 << log_T);
                for seg in segments {
                    let e: Vec<F> = EqPolynomial::evals(&seg.r);
                    let offset = seg.prefix << seg.r.len();
                    w[offset..offset + e.len()]
                        .par_iter_mut()
                        .zip(e.par_iter())
                        .with_min_len(par_enabled())
                        .for_each(|(x, y)| *x = seg.gamma * *y);
                }
                w
            }
        }
    }

    /// The weight's multilinear extension at `r_prime` (big-endian cycle point).
    pub fn mle(&self, r_prime: &[F]) -> F {
        match self {
            Self::Eq(r) => EqPolynomial::mle(&r.r, r_prime),
            Self::Packed { log_T, segments } => {
                debug_assert_eq!(r_prime.len(), *log_T);
                segments
                    .iter()
                    .map(|seg| {
                        let (hi, lo) = r_prime.split_at(seg.prefix_len);
                        // eq(prefix bits, hi): bit j (from the MSB) ↔ hi[j].
                        let eq_prefix: F = hi
                            .iter()
                            .enumerate()
                            .map(|(j, h)| {
                                let bit = (seg.prefix >> (seg.prefix_len - 1 - j)) & 1;
                                if bit == 1 {
                                    *h
                                } else {
                                    F::one() - *h
                                }
                            })
                            .product();
                        seg.gamma * eq_prefix * EqPolynomial::mle(&seg.r, lo)
                    })
                    .sum()
            }
        }
    }
}

/// Prover-side state for the cycle rounds: Gruen's split-eq trick for a plain
/// `eq(r, ·)` weight, a dense MLE for a packed one.
pub(crate) enum CyclePoly<F: JoltField> {
    Gruen(GruenSplitEqPolynomial<F>),
    Dense(MultilinearPolynomial<F>),
}

/// Address-phase schedule for a `log_k`-bit lookup: `(phases, log_m)` with
/// `phases * log_m == log_k`. Prefix checkpoints advance two rounds at a time,
/// so `log_m` must be even: the default 8 phases work when `log_k / 8` is
/// even (32, 48, 64, ...); otherwise (40, 56, ...) use 8-bit phases instead.
pub(crate) fn phase_schedule(log_k: usize) -> (usize, usize) {
    assert!(
        log_k.is_multiple_of(8),
        "lookup width must be a multiple of 8"
    );
    let log_m = log_k / NUM_PHASES;
    if log_m.is_multiple_of(2) {
        (NUM_PHASES, log_m)
    } else {
        (log_k / 8, 8)
    }
}

/// The read address's side of a read-raf: the operand whose signed value the address encodes.
/// Its contribution to the input claim is the operand's claim, and at an address point it is
/// the signed identity polynomial.
pub struct OperandRaf<F: JoltField> {
    pub operand_claim: F,
    pub log_k: usize,
}

impl<F: JoltField> OperandRaf<F> {
    /// The signed identity polynomial over the `log_k` address bits, at `r_address`.
    pub fn claim_at(&self, r_address: &[F]) -> F {
        SignedIdentityPoly::<F>::new(self.log_k).evaluate(r_address)
    }
}

/// The claims a prefix-suffix lookup's read-raf starts from.
#[derive(Debug, Clone, Copy)]
pub struct ReadRafClaims<F: JoltField> {
    pub rv_claim: F,
    pub operand_claim: F,
}

/// A prefix-suffix lookup: its read-address polynomial, cycle point and starting claims.
pub trait PrefixSuffixShoutProvider<F, LUT, const LOG_K: usize>: RafShoutProvider<F>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K> + Default,
{
    fn read_raf_claims(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<ReadRafClaims<F>, ProofVerifyError>;
}

/// Parameters shared by the prover and verifier.
///
/// `LOG_K`: total address bits (= number of address-phase rounds).
pub struct ReadRafSumcheckParams<F, LUT, const LOG_K: usize>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K>,
{
    pub gamma: F,
    pub log_T: usize,
    pub cycle: CycleWeight<F>,
    pub table: LUT,
    pub polynomial_type: VirtualPoly,
    pub sumcheck_id: SumcheckId,
    pub rv_claim: F,
    pub raf: OperandRaf<F>,
}

impl<F, LUT, const LOG_K: usize> ReadRafSumcheckParams<F, LUT, LOG_K>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        gamma: F,
        log_T: usize,
        r_node_output: OpeningPoint<BIG_ENDIAN, F>,
        table: LUT,
        polynomial_type: VirtualPoly,
        sumcheck_id: SumcheckId,
        rv_claim: F,
        raf: OperandRaf<F>,
    ) -> Self {
        Self {
            gamma,
            log_T,
            cycle: CycleWeight::Eq(r_node_output),
            table,
            polynomial_type,
            sumcheck_id,
            rv_claim,
            raf,
        }
    }
}

/// The prefix-suffix read-raf check as a relation: over the address-then-cycle hypercube,
///
/// ```text
/// input claim   rv + γ·raf_input
/// output claim  w(ρ_cycle) · ra(ρ) · (val(ρ_address) + γ·raf(ρ_address))
/// ```
///
/// with `w` the [`CycleWeight`] and `ρ = (ρ_address, ρ_cycle)` the bound point (address in round
/// order, cycle reversed). **S ⊆ R.** An honest `ra` is the one-hot encoding of the lookup
/// addresses, so the sum over addresses reads the table value and the address at each cycle.
/// The parameters carry their field data; the setup carries `γ` and both claims.
impl<F, LUT, const LOG_K: usize> ReadRafSumcheckParams<F, LUT, LOG_K>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K>,
{
    /// The bound point split into its address part (round order) and cycle part (reversed).
    fn split_point(challenges: &[F]) -> (Vec<F>, Vec<F>) {
        let (address, cycle) = challenges.split_at(LOG_K);
        (address.to_vec(), cycle.iter().rev().copied().collect())
    }

    /// The relation these parameters instantiate, with its setup.
    pub fn into_verifier(self) -> ReadRafSumcheckVerifier<F, LUT, LOG_K> {
        let setup = self.relation_setup();
        RelationVerifier::at(self, setup)
    }

    /// The relation's setup: `γ`, the read-value claim and the RAF claim.
    pub fn relation_setup(&self) -> RelationSetup<F> {
        RelationSetup {
            incoming_point: Vec::new(),
            challenges: vec![
                (RelationChallenge::Gamma, self.gamma),
                (RelationChallenge::Claim, self.rv_claim),
                (RelationChallenge::SecondClaim, self.raf.operand_claim),
            ],
        }
    }
}

impl<F, LUT, const LOG_K: usize> SymbolicSumcheck for ReadRafSumcheckParams<F, LUT, LOG_K>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K>,
{
    crate::relation_types!();

    fn id(&self) -> RelationId {
        RelationId::PrefixSuffixReadRaf
    }

    fn rounds(&self) -> usize {
        LOG_K + self.log_T
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        challenge::<G, _, _, _>(RelationChallenge::Claim)
            + challenge::<G, _, _, _>(RelationChallenge::Gamma)
                * challenge(RelationChallenge::SecondClaim)
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        derived::<G, _, _, _>(RelationDerived::CycleEq)
            * opening(RelationOpening::Id(self.produced()[0]))
            * (derived::<G, _, _, _>(RelationDerived::Table(0))
                + challenge::<G, _, _, _>(RelationChallenge::Gamma) * derived(RelationDerived::Raf))
    }
}

impl<F, LUT, const LOG_K: usize> Produces for ReadRafSumcheckParams<F, LUT, LOG_K>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K>,
{
    fn produced(&self) -> Vec<OpeningId> {
        vec![OpeningId::new(self.polynomial_type, self.sumcheck_id)]
    }
}

impl<F, LUT, const LOG_K: usize> Relation<F> for ReadRafSumcheckParams<F, LUT, LOG_K>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K> + Send + Sync,
{
    fn opening_points(
        &self,
        _setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Result<Vec<Vec<F>>, ProofVerifyError> {
        let (address, cycle) = Self::split_point(challenges);
        Ok(vec![[address, cycle].concat()])
    }

    fn derive(
        &self,
        id: &RelationDerived,
        _setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Result<F, ProofVerifyError> {
        let (address, cycle) = Self::split_point(challenges);
        Ok(match id {
            RelationDerived::Table(0) => self.table.evaluate_mle(&address),
            RelationDerived::CycleEq => self.cycle.mle(&cycle),
            RelationDerived::Raf => self.raf.claim_at(&address),
            other => return Err(not_derived(other)),
        })
    }
}

/// The verifier of a prefix-suffix read-raf check: its relation.
pub type ReadRafSumcheckVerifier<F, LUT, const LOG_K: usize> =
    RelationVerifier<F, ReadRafSumcheckParams<F, LUT, LOG_K>>;

/// The verifier of a prefix-suffix lookup's read-raf: `γ` drawn, the claims read from the
/// accumulator. `Err` if the accumulator lacks a claim it reads.
pub fn ps_read_raf_verifier<
    F: JoltField,
    T: Transcript,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K> + Default,
    const LOG_K: usize,
>(
    provider: &impl PrefixSuffixShoutProvider<F, LUT, LOG_K>,
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<ReadRafSumcheckVerifier<F, LUT, LOG_K>, ProofVerifyError> {
    let gamma = transcript.challenge_scalar();
    let claims = provider.read_raf_claims(accumulator)?;
    let raf = OperandRaf {
        operand_claim: claims.operand_claim,
        log_k: LOG_K,
    };
    let r_node_output = provider.r_cycle(accumulator)?;
    let log_T = r_node_output.len();
    let (polynomial_type, sumcheck_id) = provider.ra_poly();
    let params = ReadRafSumcheckParams::from_parts(
        gamma,
        log_T,
        r_node_output,
        LUT::default(),
        polynomial_type,
        sumcheck_id,
        claims.rv_claim,
        raf,
    );
    Ok(params.into_verifier())
}

pub mod prover;
