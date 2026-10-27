use crate::relation::Produces;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::CommittedPoly;
use rayon::prelude::*;

use crate::{
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId, VerifierOpeningAccumulator},
    relation::{
        low_to_high_point, Relation, RelationChallenge, RelationExpr, RelationId, RelationOpening,
        RelationSetup, RelationVerifier,
    },
    sumcheck_verifier::SumcheckInstanceParams,
};
use atlas_algebra::{
    field::JoltField,
    poly::opening_point::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN},
};
use atlas_transcript::Transcript;
use jolt_claims::{challenge, opening, SymbolicSumcheck};
use jolt_field::Ring;

/// Degree bound of the sumcheck round polynomials in [`HammingWeightRelation`].
const DEGREE_BOUND: usize = 1;

pub struct HammingWeightSumcheckParams<F: JoltField> {
    pub d: usize,
    pub num_rounds: usize,
    pub gamma_powers: Vec<F>,
    pub polynomial_types: Vec<CommittedPoly>,
    pub sumcheck_id: SumcheckId,
    pub r_cycle: Vec<F>,
}

impl<F: JoltField> SumcheckInstanceParams<F> for HammingWeightSumcheckParams<F> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        // No consumer constructs a `RamHammingWeight` instance. A consumer
        // whose input claim is another sumcheck's output should carry that
        // claim's `OpeningId` in these params explicitly.
        self.gamma_powers.iter().sum()
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    // output = Σ_{i=0}^{d-1} γ^i * ra_i
}

/// `Σ_i γ^i · Σ_k ra_i(k, r_cycle) = Σ_i γ^i`: each one-hot chunk has Hamming weight one.
///
/// ```text
/// input claim   Σ_i γ^i
/// output claim  Σ_i γ^i · ra_i(r′, r_cycle)        (r′ the low-to-high bound address point)
/// ```
///
/// **S ⊆ R.** An honest chunk is one-hot in every cycle, so its sum over the address is one.
/// The setup's incoming point is `r_cycle`; `γ^i` are [`RelationChallenge::Batching`].
pub struct HammingWeightRelation {
    d: usize,
    num_rounds: usize,
    polynomial_types: Vec<CommittedPoly>,
    sumcheck_id: SumcheckId,
}

impl SymbolicSumcheck for HammingWeightRelation {
    crate::relation_types!();
    type Shape = Self;

    fn new(shape: Self) -> Self {
        shape
    }

    fn id() -> RelationId {
        RelationId::HammingWeight
    }

    fn rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        (0..self.d).fold(RelationExpr::zero(), |acc, i| {
            acc + challenge(RelationChallenge::Batching(i))
        })
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        (0..self.d).fold(RelationExpr::zero(), |acc, i| {
            acc + challenge::<G, _, _, _>(RelationChallenge::Batching(i))
                * opening(RelationOpening::Id(OpeningId::new(
                    self.polynomial_types[i],
                    self.sumcheck_id,
                )))
        })
    }
}

impl Produces for HammingWeightRelation {
    fn produced(&self) -> Vec<OpeningId> {
        self.polynomial_types
            .iter()
            .map(|&poly| OpeningId::new(poly, self.sumcheck_id))
            .collect()
    }
}

impl<F: JoltField> Relation<F> for HammingWeightRelation {
    fn opening_points(&self, setup: &RelationSetup<F>, challenges: &[F]) -> Vec<Vec<F>> {
        let point = [low_to_high_point(challenges), setup.incoming_point.clone()].concat();
        vec![point; self.d]
    }

    fn append_produced_requests<T: Transcript>(
        &self,
        setup: &RelationSetup<F>,
        challenges: &[F],
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) {
        let point = [low_to_high_point(challenges), setup.incoming_point.clone()].concat();
        accumulator.append_sparse(
            transcript,
            self.polynomial_types.clone(),
            self.sumcheck_id,
            point,
        );
    }
}

/// The verifier of a Hamming-weight check: its relation.
pub type HammingWeightSumcheckVerifier<F> = RelationVerifier<F, HammingWeightRelation>;

impl<F: JoltField> HammingWeightSumcheckParams<F> {
    /// The relation these parameters instantiate, with its setup.
    pub fn into_verifier(self) -> HammingWeightSumcheckVerifier<F> {
        let setup = RelationSetup {
            incoming_point: self.r_cycle,
            challenges: self
                .gamma_powers
                .into_iter()
                .enumerate()
                .map(|(i, gamma)| (RelationChallenge::Batching(i), gamma))
                .collect(),
        };
        RelationVerifier::at(
            HammingWeightRelation {
                d: self.d,
                num_rounds: self.num_rounds,
                polynomial_types: self.polynomial_types,
                sumcheck_id: self.sumcheck_id,
            },
            setup,
        )
    }
}

pub mod prover;
