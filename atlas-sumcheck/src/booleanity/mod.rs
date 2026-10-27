use crate::relation::Produces;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::CommittedPoly;
use rayon::prelude::*;
use std::fmt::Debug;

use crate::{
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId, VerifierOpeningAccumulator},
    relation::{
        low_to_high_point, Relation, RelationChallenge, RelationDerived, RelationExpr, RelationId,
        RelationOpening, RelationSetup, RelationVerifier,
    },
    sumcheck_verifier::SumcheckInstanceParams,
};
use atlas_algebra::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        opening_point::{OpeningPoint, BIG_ENDIAN},
    },
};
use atlas_transcript::Transcript;
use jolt_claims::{challenge, derived, opening, SymbolicSumcheck};
use jolt_field::Ring;

/// Degree bound of the sumcheck round polynomials in [`BooleanityRelation`].
const DEGREE_BOUND: usize = 3;

/// The input claim of a [`LinearTerm`]: `constant + Σ coef·claim(id)` over
/// virtual openings already in the accumulator.
#[derive(Clone, Debug)]
pub struct LinearClaim<F: JoltField> {
    pub constant: F,
    pub terms: Vec<(OpeningId, F)>,
}

impl<F: JoltField> LinearClaim<F> {
    pub fn constant(c: F) -> Self {
        Self {
            constant: c,
            terms: Vec::new(),
        }
    }

    pub fn evaluate(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.terms.iter().fold(self.constant, |acc, (id, coef)| {
            acc + *coef * accumulator.get_virtual_polynomial_opening(*id).1
        })
    }
}

/// An optional linear check folded into the Booleanity sumcheck, sharing its
/// `G` / `H` tables and its single opening of every chunk:
///
/// ```text
/// Σ_{k,t} Σ_d (γ'_d + w_d·k) · eq(r_cycle, t) · ra_d(k, t) = claim
/// ```
///
/// With `w = 0` this is the Hamming-weight check (`Σ_k ra_d(k, r_cycle)`
/// batched by `γ'`); with `w_d = β·2^{shift_d}` it also recovers the chunked
/// value `Σ_d 2^{shift_d}·Σ_k k·ra_d(k, r_cycle)`.
#[derive(Clone, Debug)]
pub struct LinearTerm<F: JoltField> {
    /// Per-chunk Hamming batching coefficients `γ'_d`.
    pub gammas: Vec<F>,
    /// Per-chunk value weights `w_d`.
    pub weights: Vec<F>,
    /// The claimed sum.
    pub claim: LinearClaim<F>,
}

pub struct BooleanitySumcheckParams<F: JoltField> {
    /// Optional linear check sharing the chunks' opening (see [`LinearTerm`]).
    pub linear: Option<LinearTerm<F>>,
    /// Number of address chunks
    pub d: usize,
    /// Log of chunk size
    pub log_k_chunk: usize,
    /// Log of trace length
    pub log_t: usize,
    /// Batching challenges
    pub gammas: Vec<F::Challenge>,
    /// Address binding point
    pub r_address: Vec<F>,
    /// Cycle binding point
    pub r_cycle: Vec<F>,
    /// Polynomial types for opening accumulator
    pub polynomial_types: Vec<CommittedPoly>,
    /// Sumcheck ID for opening accumulator
    pub sumcheck_id: SumcheckId,
}

impl<F: JoltField> SumcheckInstanceParams<F> for BooleanitySumcheckParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.log_k_chunk + self.log_t
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.linear
            .as_ref()
            .map_or(F::zero(), |l| l.claim.evaluate(accumulator))
    }

    fn normalize_opening_point(&self, sumcheck_challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut opening_point = sumcheck_challenges.to_vec();
        opening_point[..self.log_k_chunk].reverse();
        opening_point[self.log_k_chunk..].reverse();
        opening_point.into()
    }

    // output = eq_eval * Σ_i γ_i * (ra_i² - ra_i)
    //        = Σ_i [ Challenge(2i) * Opening(ra_i) * Opening(ra_i)
    //              + Challenge(2i+1) * Opening(ra_i) ]
    // where Challenge(2i) = eq_eval * γ_i, Challenge(2i+1) = -eq_eval * γ_i
}

/// `Σ_{k,t} eq((r_address, r_cycle), (k, t)) · Σ_i γ_i·(ra_i(k,t)² − ra_i(k,t)) = 0`: every one-hot
/// chunk is boolean, optionally with a [`LinearTerm`] sharing the chunks' single opening.
///
/// ```text
/// input claim   linear claim (0 without a linear term)
/// output claim  eq(r_address‖r_cycle, ρ) · Σ_i γ_i·(ra_i(ρ)² − ra_i(ρ))
///             + eq(r_cycle, ρ_cycle) · Σ_i ra_i(ρ)·(γ′_i + w_i·k(ρ_address))
/// ```
///
/// with `ρ = (ρ_address, ρ_cycle)` the bound point (each part low-to-high) and
/// `k(ρ_address) = Σ_j 2^j·ρ_j` the address index.
///
/// **S ⊆ R.** An honest chunk is `0` or `1` everywhere, so the booleanity summands vanish; the
/// linear term sums `Σ_k (γ′_i + w_i·k)·ra_i(k, r_cycle)`, which the caller's claim states for an
/// honest one-hot chunk. The setup's incoming point is `r_address ‖ r_cycle`; the coefficients
/// are `γ_i` ([`RelationChallenge::Batching`]), `γ′_i` ([`RelationChallenge::SecondBatching`]),
/// `w_i` ([`RelationChallenge::Weight`]) and the linear claim's constant and coefficients.
pub struct BooleanityRelation {
    d: usize,
    log_k_chunk: usize,
    log_t: usize,
    /// The virtual openings of the linear claim, when there is a linear term.
    linear: Option<Vec<OpeningId>>,
    polynomial_types: Vec<CommittedPoly>,
    sumcheck_id: SumcheckId,
}

impl BooleanityRelation {
    fn ra<G: Ring>(&self, i: usize) -> RelationExpr<G> {
        opening(RelationOpening::Id(OpeningId::new(
            self.polynomial_types[i],
            self.sumcheck_id,
        )))
    }

    /// The bound point in opening order: each part's challenges, last first.
    fn bound_point<F: Copy>(&self, challenges: &[F]) -> Vec<F> {
        let (address, cycle) = challenges.split_at(self.log_k_chunk);
        [low_to_high_point(address), low_to_high_point(cycle)].concat()
    }
}

impl SymbolicSumcheck for BooleanityRelation {
    crate::relation_types!();
    type Shape = Self;

    fn new(shape: Self) -> Self {
        shape
    }

    fn id() -> RelationId {
        RelationId::Booleanity
    }

    fn rounds(&self) -> usize {
        self.log_k_chunk + self.log_t
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        let Some(terms) = &self.linear else {
            return RelationExpr::zero();
        };
        terms.iter().enumerate().fold(
            challenge(RelationChallenge::ClaimConstant),
            |acc, (j, &id)| {
                acc + challenge::<G, _, _, _>(RelationChallenge::ClaimCoefficient(j))
                    * opening(RelationOpening::Id(id))
            },
        )
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        let booleanity = (0..self.d).fold(RelationExpr::zero(), |acc, i| {
            acc + challenge::<G, _, _, _>(RelationChallenge::Batching(i))
                * (self.ra::<G>(i) * self.ra::<G>(i) - self.ra::<G>(i))
        });
        let output = derived::<G, _, _, _>(RelationDerived::OneHotEq) * booleanity;
        if self.linear.is_none() {
            return output;
        }
        let linear = (0..self.d).fold(RelationExpr::zero(), |acc, i| {
            acc + self.ra::<G>(i)
                * (challenge::<G, _, _, _>(RelationChallenge::SecondBatching(i))
                    + challenge::<G, _, _, _>(RelationChallenge::Weight(i))
                        * derived(RelationDerived::AddressIndex))
        });
        output + derived::<G, _, _, _>(RelationDerived::CycleEq) * linear
    }
}

impl Produces for BooleanityRelation {
    fn produced(&self) -> Vec<OpeningId> {
        self.polynomial_types
            .iter()
            .map(|&poly| OpeningId::new(poly, self.sumcheck_id))
            .collect()
    }
}

impl<F: JoltField> Relation<F> for BooleanityRelation {
    fn opening_points(&self, _setup: &RelationSetup<F>, challenges: &[F]) -> Vec<Vec<F>> {
        vec![self.bound_point(challenges); self.d]
    }

    fn append_produced_requests<T: Transcript>(
        &self,
        _setup: &RelationSetup<F>,
        challenges: &[F],
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) {
        accumulator.append_sparse(
            transcript,
            self.polynomial_types.clone(),
            self.sumcheck_id,
            self.bound_point(challenges),
        );
    }

    fn derive(&self, id: &RelationDerived, setup: &RelationSetup<F>, challenges: &[F]) -> F {
        let (r_address, r_cycle) = setup.incoming_point.split_at(self.log_k_chunk);
        match id {
            // The challenges in round order against each part of the incoming point reversed.
            RelationDerived::OneHotEq => {
                let combined: Vec<F> = r_address
                    .iter()
                    .rev()
                    .chain(r_cycle.iter().rev())
                    .copied()
                    .collect();
                EqPolynomial::mle(challenges, &combined)
            }
            RelationDerived::CycleEq => {
                let r_cycle_rev: Vec<F> = r_cycle.iter().rev().copied().collect();
                EqPolynomial::mle(&challenges[self.log_k_chunk..], &r_cycle_rev)
            }
            RelationDerived::AddressIndex => challenges[..self.log_k_chunk]
                .iter()
                .enumerate()
                .map(|(j, c)| F::from_u64(1u64 << j) * *c)
                .sum(),
            other => panic!("{other:?} is not derived by this relation"),
        }
    }
}

/// The verifier of a booleanity check: its relation.
pub type BooleanitySumcheckVerifier<F> = RelationVerifier<F, BooleanityRelation>;

impl<F: JoltField> BooleanitySumcheckParams<F> {
    /// The relation these parameters instantiate, with its setup.
    pub fn into_verifier(self) -> BooleanitySumcheckVerifier<F> {
        let mut challenges: Vec<(RelationChallenge, F)> = self
            .gammas
            .iter()
            .enumerate()
            .map(|(i, gamma)| (RelationChallenge::Batching(i), (*gamma).into()))
            .collect();
        let linear = self.linear.map(|lin| {
            challenges.push((RelationChallenge::ClaimConstant, lin.claim.constant));
            for (i, (gamma, weight)) in lin.gammas.iter().zip(&lin.weights).enumerate() {
                challenges.push((RelationChallenge::SecondBatching(i), *gamma));
                challenges.push((RelationChallenge::Weight(i), *weight));
            }
            lin.claim
                .terms
                .iter()
                .enumerate()
                .map(|(j, (id, coefficient))| {
                    challenges.push((RelationChallenge::ClaimCoefficient(j), *coefficient));
                    *id
                })
                .collect()
        });
        let setup = RelationSetup {
            incoming_point: [self.r_address, self.r_cycle].concat(),
            challenges,
        };
        RelationVerifier::at(
            BooleanityRelation {
                d: self.d,
                log_k_chunk: self.log_k_chunk,
                log_t: self.log_t,
                linear,
                polynomial_types: self.polynomial_types,
                sumcheck_id: self.sumcheck_id,
            },
            setup,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::prover::*;
    use crate::opening_proof::prover::ProverOpeningAccumulator;
    use atlas_algebra::field::fp128::Fp128 as Fr;
    use common::CommittedPoly;
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    use crate::sumcheck::Sumcheck;
    use atlas_algebra::{field::IntoOpening, utils::thread::unsafe_allocate_zero_vec};
    use atlas_transcript::{
        Blake2bProver, Blake2bVerifier, ProvingTranscript, Transcript, VerifyingTranscript,
    };

    use super::*;

    #[test]
    fn test_small_bool_sumcheck() {
        let mut rng = StdRng::seed_from_u64(0x456);
        let log_lookups = 4;
        let log_words = 3;
        let words = 1 << log_words;
        let lookups = 1 << log_lookups;

        let mut prover_transcript = Blake2bProver::new(b"");
        let mut prover_accumulator = ProverOpeningAccumulator::new();
        let r_lookups: Vec<Fr> = prover_transcript
            .challenge_vector_optimized::<Fr>(log_lookups)
            .into_opening();
        let r_words: Vec<Fr> = prover_transcript
            .challenge_vector_optimized::<Fr>(log_words)
            .into_opening();

        let params = BooleanitySumcheckParams {
            linear: None,
            d: 1,
            log_k_chunk: log_words,
            log_t: log_lookups,
            r_cycle: r_lookups,
            r_address: r_words,
            gammas: vec![<Fr as JoltField>::Challenge::from(1)],
            polynomial_types: vec![CommittedPoly::NodeOutputRaD(0, 0)],
            sumcheck_id: SumcheckId::Booleanity,
        };

        // Generate read addresses and G vector
        let read_addresses: Vec<usize> = (0..lookups)
            .map(|_| (rng.next_u32() as usize) % words)
            .collect();

        let E: Vec<Fr> = EqPolynomial::evals(&params.r_cycle);
        let mut G = unsafe_allocate_zero_vec(words);
        for (i, &addr) in read_addresses.iter().enumerate() {
            G[addr] += E[i];
        }

        let H_indices: Vec<Option<u16>> = read_addresses
            .iter()
            .map(|&addr| Some(addr as u16))
            .collect();

        let mut prover = SmallBooleanitySumcheckProver::gen(params, vec![G], vec![H_indices]);

        let (proof, _) =
            Sumcheck::prove(&mut prover, &mut prover_accumulator, &mut prover_transcript);

        let mut verifier_transcript = Blake2bVerifier::new(b"", prover_transcript.narg().to_vec());
        let mut verifier_accumulator = VerifierOpeningAccumulator::new();
        let _r_x: Vec<<Fr as JoltField>::Challenge> =
            verifier_transcript.challenge_vector_optimized::<Fr>(log_lookups);
        let _r_words: Vec<<Fr as JoltField>::Challenge> =
            verifier_transcript.challenge_vector_optimized::<Fr>(log_words);

        // Take claims
        for (key, (_, value)) in &prover_accumulator.openings {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::default();
            verifier_accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }

        let verifier = prover.params.into_verifier();
        Sumcheck::verify(
            &proof,
            &verifier,
            &mut verifier_accumulator,
            &mut verifier_transcript,
        )
        .unwrap();
    }
}

pub mod prover;
