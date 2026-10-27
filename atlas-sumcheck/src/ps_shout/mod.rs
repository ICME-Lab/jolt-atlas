use crate::{
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    relation::{
        Produces, Relation, RelationChallenge, RelationDerived, RelationExpr, RelationId,
        RelationOpening, RelationSetup, RelationVerifier,
    },
    sumcheck_verifier::SumcheckInstanceParams,
};
use atlas_algebra::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::MultilinearPolynomial,
        opening_point::{OpeningPoint, BIG_ENDIAN},
        prefix_suffix::PrefixRegistry,
    },
    utils::{lookup_bits::LookupBits, thread::unsafe_allocate_zero_vec},
};
use atlas_lookup_tables::{JoltLookupTable, PrefixSuffixDecompositionTrait};
use common::{parallel::par_enabled, VirtualPoly};
use jolt_claims::{challenge, derived, opening, SymbolicSumcheck};
use jolt_field::Ring;
use jolt_poly::GruenSplitEqPolynomial;
use rayon::prelude::*;

pub mod binary;
#[cfg(test)]
mod lookup_table_tests;
pub mod unary;

pub(crate) const DEGREE_BOUND: usize = 2;

/// Common interface shared by both binary and unary ps_shout providers.
///
/// Contains the three methods that are structurally identical across
/// `binary::PrefixSuffixShoutProvider` and `unary::PrefixSuffixShoutProvider`,
/// so implementations only write them once regardless of which variant is used.
pub trait RafShoutProvider<F: JoltField> {
    fn ra_poly(&self) -> (VirtualPoly, SumcheckId);
    fn r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F>;
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

/// Verifier-side RAF: computing the RAF's contribution to the input claim
/// and the RAF claim at a given address evaluation point.
///
/// Binary: `raf_input_claim(γ) = left_claim + γ * right_claim`
/// Unary:  `raf_input_claim(γ) = operand_claim`
///
/// Then `input_claim = rv_claim + γ * raf.raf_input_claim(γ)`.
pub trait RafVerifierData<F: JoltField>: Send + Sync {
    fn raf_input_claim(&self, gamma: F) -> F;
    fn raf_claim_at(&self, r_address: &[F], gamma: F) -> F;
}

/// Prover-side RAF state: holds the PS decompositions and produces
/// per-round univariate contributions and final checkpoint values.
pub trait RafProverState<F: JoltField>: Send + Sync {
    fn init_Q(&mut self, u_evals: &[F], lookup_indices: &[LookupBits]);
    fn init_P(&mut self, registry: &mut PrefixRegistry<F>);
    fn prover_msg(&self, gamma: F) -> [F; 2];
    fn bind(&mut self, r_j: F::Challenge);
    fn raf_val(&self, registry: &PrefixRegistry<F>, gamma: F) -> F;
}

/// Parameters shared by the prover and verifier.
///
/// `LOG_K`: total address bits (= number of address-phase rounds).
/// `N`: bit-width used for table prefix/suffix MLE computations.
/// `VD`: verifier-side RAF data (claims + evaluation).
pub struct ReadRafSumcheckParams<F, LUT, const LOG_K: usize, const N: usize, VD>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
{
    pub gamma: F,
    pub log_T: usize,
    pub cycle: CycleWeight<F>,
    pub table: LUT,
    pub polynomial_type: VirtualPoly,
    pub sumcheck_id: SumcheckId,
    pub rv_claim: F,
    pub raf: VD,
}

impl<F, LUT, const LOG_K: usize, const N: usize, VD> ReadRafSumcheckParams<F, LUT, LOG_K, N, VD>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
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
        raf: VD,
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

    /// Like [`Self::from_parts`], for an arbitrary [`CycleWeight`] (e.g. a
    /// packed multi-node instance).
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts_with_cycle(
        gamma: F,
        cycle: CycleWeight<F>,
        table: LUT,
        polynomial_type: VirtualPoly,
        sumcheck_id: SumcheckId,
        rv_claim: F,
        raf: VD,
    ) -> Self {
        Self {
            gamma,
            log_T: cycle.log_T(),
            cycle,
            table,
            polynomial_type,
            sumcheck_id,
            rv_claim,
            raf,
        }
    }
}

impl<F, LUT, const LOG_K: usize, const N: usize, VD> SumcheckInstanceParams<F>
    for ReadRafSumcheckParams<F, LUT, LOG_K, N, VD>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
{
    fn num_rounds(&self) -> usize {
        LOG_K + self.log_T
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.rv_claim + self.gamma * self.raf.raf_input_claim(self.gamma)
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let (r_address_prime, r_node_output_prime) = challenges.split_at(LOG_K);
        let r_node_output_prime = r_node_output_prime
            .iter()
            .copied()
            .rev()
            .collect::<Vec<_>>();
        OpeningPoint::new([r_address_prime.to_vec(), r_node_output_prime].concat())
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
impl<F, LUT, const LOG_K: usize, const N: usize, VD> ReadRafSumcheckParams<F, LUT, LOG_K, N, VD>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
{
    /// The bound point split into its address part (round order) and cycle part (reversed).
    fn split_point(challenges: &[F]) -> (Vec<F>, Vec<F>) {
        let (address, cycle) = challenges.split_at(LOG_K);
        (address.to_vec(), cycle.iter().rev().copied().collect())
    }

    /// The relation these parameters instantiate, with its setup.
    pub fn into_verifier(self) -> ReadRafSumcheckVerifier<F, LUT, LOG_K, N, VD> {
        let setup = RelationSetup {
            incoming_point: Vec::new(),
            challenges: vec![
                (RelationChallenge::Gamma, self.gamma),
                (RelationChallenge::Claim, self.rv_claim),
                (
                    RelationChallenge::SecondClaim,
                    self.raf.raf_input_claim(self.gamma),
                ),
            ],
        };
        RelationVerifier::at(self, setup)
    }
}

impl<F, LUT, const LOG_K: usize, const N: usize, VD> SymbolicSumcheck
    for ReadRafSumcheckParams<F, LUT, LOG_K, N, VD>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
{
    crate::relation_types!();
    type Shape = Self;

    fn new(shape: Self) -> Self {
        shape
    }

    fn id() -> RelationId {
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

impl<F, LUT, const LOG_K: usize, const N: usize, VD> Produces
    for ReadRafSumcheckParams<F, LUT, LOG_K, N, VD>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N>,
    VD: RafVerifierData<F>,
{
    fn produced(&self) -> Vec<OpeningId> {
        vec![OpeningId::new(self.polynomial_type, self.sumcheck_id)]
    }
}

impl<F, LUT, const LOG_K: usize, const N: usize, VD> Relation<F>
    for ReadRafSumcheckParams<F, LUT, LOG_K, N, VD>
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<N> + Send + Sync,
    VD: RafVerifierData<F>,
{
    fn opening_points(&self, _setup: &RelationSetup<F>, challenges: &[F]) -> Vec<Vec<F>> {
        let (address, cycle) = Self::split_point(challenges);
        vec![[address, cycle].concat()]
    }

    fn derive(&self, id: &RelationDerived, _setup: &RelationSetup<F>, challenges: &[F]) -> F {
        let (address, cycle) = Self::split_point(challenges);
        match id {
            RelationDerived::Table(0) => self.table.evaluate_mle(&address),
            RelationDerived::CycleEq => self.cycle.mle(&cycle),
            RelationDerived::Raf => self.raf.raf_claim_at(&address, self.gamma),
            other => panic!("{other:?} is not derived by this relation"),
        }
    }
}

/// The verifier of a prefix-suffix read-raf check: its relation.
pub type ReadRafSumcheckVerifier<F, LUT, const LOG_K: usize, const N: usize, VD> =
    RelationVerifier<F, ReadRafSumcheckParams<F, LUT, LOG_K, N, VD>>;

pub mod prover;
