use crate::relation::Produces;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::VirtualPoly;
use rayon::prelude::*;

use crate::{
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    relation::{
        Relation, RelationChallenge, RelationDerived, RelationExpr, RelationId, RelationOpening,
        RelationSetup, RelationVerifier,
    },
    sumcheck_verifier::SumcheckInstanceParams,
};
use atlas_algebra::{
    field::JoltField,
    poly::opening_point::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN},
};
use jolt_claims::{challenge, derived, opening, SymbolicSumcheck};
use jolt_field::Ring;

/// Degree bound of the sumcheck round polynomials in [`HammingBooleanityRelation`].
const DEGREE_BOUND: usize = 3;

pub struct HammingBooleanitySumcheckParams<F: JoltField> {
    pub d: usize,
    pub num_rounds: usize,
    pub gamma_powers: Vec<F>,
    pub polynomial_types: Vec<VirtualPoly>,
    pub sumcheck_id: SumcheckId,
    pub r_cycle: Vec<F>,
}

impl<F: JoltField> SumcheckInstanceParams<F> for HammingBooleanitySumcheckParams<F> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    // output = eq_eval * Σ_i gamma^i * hw_i * (hw_i - 1)
    //        = Σ_i [Challenge(2i) * Opening(hw_i)^2 + Challenge(2i+1) * Opening(hw_i)]
}

/// `Σ_t eq(r_cycle, t) · Σ_i γ^i · hw_i(t)·(hw_i(t) − 1) = 0`: every Hamming-weight indicator
/// is boolean.
///
/// ```text
/// input claim   0
/// output claim  eq(r_cycle, r′) · Σ_i γ^i · (hw_i(r′)² − hw_i(r′))   (r′ the low-to-high bound point)
/// ```
///
/// **S ⊆ R.** An honest indicator is `0` or `1` everywhere, so every summand vanishes. The
/// setup's incoming point is `r_cycle`; `γ^i` are [`RelationChallenge::Batching`].
pub struct HammingBooleanityRelation {
    d: usize,
    num_rounds: usize,
    polynomial_types: Vec<VirtualPoly>,
    sumcheck_id: SumcheckId,
}

impl SymbolicSumcheck for HammingBooleanityRelation {
    crate::relation_types!();
    type Shape = Self;

    fn new(shape: Self) -> Self {
        shape
    }

    fn id() -> RelationId {
        RelationId::HammingBooleanity
    }

    fn rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        RelationExpr::zero()
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        let booleanity = (0..self.d).fold(RelationExpr::zero(), |acc, i| {
            let hw = || {
                opening::<G, _, _, _>(RelationOpening::Id(OpeningId::new(
                    self.polynomial_types[i],
                    self.sumcheck_id,
                )))
            };
            acc + challenge::<G, _, _, _>(RelationChallenge::Batching(i)) * (hw() * hw() - hw())
        });
        derived::<G, _, _, _>(RelationDerived::Eq) * booleanity
    }
}

impl Produces for HammingBooleanityRelation {
    fn produced(&self) -> Vec<OpeningId> {
        self.polynomial_types
            .iter()
            .map(|&poly| OpeningId::new(poly, self.sumcheck_id))
            .collect()
    }
}

impl<F: JoltField> Relation<F> for HammingBooleanityRelation {}

/// The verifier of a Hamming-booleanity check: its relation.
pub type HammingBooleanitySumcheckVerifier<F> = RelationVerifier<F, HammingBooleanityRelation>;

impl<F: JoltField> HammingBooleanitySumcheckParams<F> {
    /// The relation these parameters instantiate, with its setup.
    pub fn into_verifier(self) -> HammingBooleanitySumcheckVerifier<F> {
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
            HammingBooleanityRelation {
                d: self.d,
                num_rounds: self.num_rounds,
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
    use crate::opening_proof::VerifierOpeningAccumulator;
    use atlas_algebra::field::fp128::Fp128 as Fr;
    use atlas_algebra::field::IntoOpening;
    use jolt_field::Ring;
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    use crate::sumcheck::Sumcheck;
    use atlas_transcript::{
        Blake2bProver, Blake2bVerifier, ProvingTranscript, Transcript, VerifyingTranscript,
    };

    use super::*;

    #[test]
    fn test_hamming_booleanity_sumcheck() {
        let mut rng = StdRng::seed_from_u64(0x123);
        let log_T = 5;
        let T = 1 << log_T;

        let mut prover_transcript = Blake2bProver::new(b"");
        let mut prover_accumulator = ProverOpeningAccumulator::new();
        let r_cycle = prover_transcript
            .challenge_vector_optimized::<Fr>(log_T)
            .into_opening();

        let params = HammingBooleanitySumcheckParams {
            d: 1,
            num_rounds: log_T,
            gamma_powers: vec![Fr::from_u8(1)],
            polynomial_types: vec![VirtualPoly::HammingWeight(0)],
            sumcheck_id: SumcheckId::RamHammingBooleanity,
            r_cycle,
        };

        let hw: Vec<Fr> = (0..T).map(|_| Fr::from_u32(rng.next_u32() % 2)).collect();

        let mut prover = HammingBooleanitySumcheckProver::gen(params, vec![hw]);

        let (proof, _) =
            Sumcheck::prove(&mut prover, &mut prover_accumulator, &mut prover_transcript);

        let mut verifier_transcript = Blake2bVerifier::new(b"", prover_transcript.narg().to_vec());
        let mut verifier_accumulator = VerifierOpeningAccumulator::new();
        let _r_cycle: Vec<Fr> = verifier_transcript
            .challenge_vector_optimized::<Fr>(log_T)
            .into_opening();

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
