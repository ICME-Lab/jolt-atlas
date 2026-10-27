use crate::relation::Produces;
use crate::{
    config::OneHotParams,
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
        opening_point::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN},
    },
};
use atlas_transcript::Transcript;
use common::CommittedPoly;
use jolt_claims::{challenge, derived, opening, SymbolicSumcheck};
use jolt_field::Ring;
use rayon::prelude::*;

// Instruction read-access (RA) virtualization sumcheck
//
// Proves the relation:
//   Σ_j eq(r_cycle, j) ⋅ Π_{i=0}^{D-1} ra_i(r_{address,i}, j) = ra_claim
// where:
// - eq is the MLE of equality on bitstrings; evaluated at field points (r_cycle, j).
// - ra_i are MLEs of chunk-wise access indicators (1 on matching {0,1}-points).
// - ra_claim is the claimed evaluation of the virtual read-access polynomial from the read-raf sumcheck.

pub struct RaSumcheckParams<F: JoltField> {
    pub r_address: OpeningPoint<BIG_ENDIAN, F>,
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    pub one_hot_params: OneHotParams,
    pub ra_claim: F,
    /// Polynomial types for opening accumulator
    pub polynomial_types: Vec<CommittedPoly>,
}

impl<F: JoltField> SumcheckInstanceParams<F> for RaSumcheckParams<F> {
    fn num_rounds(&self) -> usize {
        self.r_cycle.len()
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.ra_claim
    }

    fn degree(&self) -> usize {
        self.one_hot_params.instruction_d + 1
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    // output = eq_eval * ∏_{i=0}^{d-1} ra_i
    //        = Challenge(0) * Opening(ra_0) * Opening(ra_1) * ... * Opening(ra_{d-1})
}

/// `Σ_j eq(r_cycle, j) · Π_i ra_i(r_{address,i}, j) = ra(r_address, r_cycle)`: a virtual
/// read-address polynomial is the product of its one-hot chunks.
///
/// ```text
/// input claim   ra_claim                         (the virtual polynomial's opening)
/// output claim  eq(r_cycle, r′) · Π_i ra_i(r_{address,i}, r′)   (r′ the low-to-high bound point)
/// ```
///
/// **S ⊆ R.** An honest address decomposes into its chunks, and the one-hot encoding of the
/// address is the product of the chunks' encodings. The setup's incoming point is
/// `r_address ‖ r_cycle`; `ra_claim` is [`RelationChallenge::Claim`].
pub struct RaVirtualizationRelation {
    one_hot_params: OneHotParams,
    address_len: usize,
    cycle_len: usize,
    polynomial_types: Vec<CommittedPoly>,
}

impl RaVirtualizationRelation {
    fn cycle_point<F: Copy>(&self, setup: &RelationSetup<F>) -> Vec<F> {
        setup.incoming_point[self.address_len..].to_vec()
    }
}

impl SymbolicSumcheck for RaVirtualizationRelation {
    crate::relation_types!();
    type Shape = Self;

    fn new(shape: Self) -> Self {
        shape
    }

    fn id() -> RelationId {
        RelationId::RaVirtualization
    }

    fn rounds(&self) -> usize {
        self.cycle_len
    }

    fn degree(&self) -> usize {
        self.one_hot_params.instruction_d + 1
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        challenge(RelationChallenge::Claim)
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        (0..self.one_hot_params.instruction_d).fold(
            derived::<G, _, _, _>(RelationDerived::CycleEq),
            |acc, i| {
                acc * opening::<G, _, _, _>(RelationOpening::Id(OpeningId::new(
                    self.polynomial_types[i],
                    SumcheckId::RaVirtualization,
                )))
            },
        )
    }
}

impl Produces for RaVirtualizationRelation {
    fn produced(&self) -> Vec<OpeningId> {
        self.polynomial_types[..self.one_hot_params.instruction_d]
            .iter()
            .map(|&poly| OpeningId::new(poly, SumcheckId::RaVirtualization))
            .collect()
    }
}

impl<F: JoltField> Relation<F> for RaVirtualizationRelation {
    fn opening_points(&self, setup: &RelationSetup<F>, challenges: &[F]) -> Vec<Vec<F>> {
        let r_cycle = low_to_high_point(challenges);
        self.one_hot_params
            .compute_r_address_chunks(&setup.incoming_point[..self.address_len])
            .iter()
            .map(|r_address| [r_address.as_slice(), r_cycle.as_slice()].concat())
            .collect()
    }

    fn append_produced_requests<T: Transcript>(
        &self,
        setup: &RelationSetup<F>,
        challenges: &[F],
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) {
        for (i, point) in self
            .opening_points(setup, challenges)
            .into_iter()
            .enumerate()
        {
            accumulator.append_sparse(
                transcript,
                vec![self.polynomial_types[i]],
                SumcheckId::RaVirtualization,
                point,
            );
        }
    }

    fn derive(&self, id: &RelationDerived, setup: &RelationSetup<F>, challenges: &[F]) -> F {
        match id {
            RelationDerived::CycleEq => {
                EqPolynomial::mle(&self.cycle_point(setup), &low_to_high_point(challenges))
            }
            other => panic!("{other:?} is not derived by this relation"),
        }
    }
}

/// The verifier of a read-address virtualization: its relation.
pub type RaSumcheckVerifier<F> = RelationVerifier<F, RaVirtualizationRelation>;

impl<F: JoltField> RaSumcheckParams<F> {
    /// The relation these parameters instantiate, with its setup.
    pub fn into_verifier(self) -> RaSumcheckVerifier<F> {
        let address_len = self.r_address.len();
        let cycle_len = self.r_cycle.len();
        let setup = RelationSetup {
            incoming_point: [self.r_address.r, self.r_cycle.r].concat(),
            challenges: vec![(RelationChallenge::Claim, self.ra_claim)],
        };
        RelationVerifier::at(
            RaVirtualizationRelation {
                one_hot_params: self.one_hot_params,
                address_len,
                cycle_len,
                polynomial_types: self.polynomial_types,
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
    use common::parallel::par_enabled;
    use common::CommittedPoly;
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    use crate::{config::OneHotConfig, sumcheck::Sumcheck};
    use atlas_algebra::{
        field::{ChallengeFieldOps, FieldChallengeOps},
        poly::{
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            one_hot_polynomial::OneHotPolynomial,
        },
        utils::{index_to_field_bitvector, math::Math},
    };
    use atlas_transcript::{
        Blake2bProver, Blake2bVerifier, ProvingTranscript, Transcript, VerifyingTranscript,
    };

    use super::*;

    pub fn one_hot_poly_evaluate<C, F>(indices: &[usize], r_cycle: &[C], r_address: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: JoltField + std::ops::Mul<C, Output = F> + std::ops::SubAssign<F> + FieldChallengeOps<C>,
    {
        assert_eq!(r_cycle.len().pow2(), indices.len(),);
        let poly = MultilinearPolynomial::from(
            indices
                .par_iter()
                .with_min_len(par_enabled())
                .map(|index| {
                    let bit_vec: Vec<F> = index_to_field_bitvector(*index as u64, r_address.len());
                    EqPolynomial::mle(r_address, &bit_vec)
                })
                .collect::<Vec<F>>(),
        );
        poly.evaluate(r_cycle)
    }

    #[test]
    fn test_ra_d8() {
        test_ra_virtualization(32, 4);
    }

    #[test]
    fn test_ra_exponentiation() {
        test_ra_virtualization(11, 4);
    }

    #[test]
    fn test_ra_tanh() {
        test_ra_virtualization(10, 4);
    }

    fn test_ra_virtualization(log_K: usize, log_k_chunk: u8) {
        let mut rng = StdRng::seed_from_u64(0x4506);
        let log_T: usize = 16;
        let T: usize = 1 << log_T;
        let K: usize = 1 << log_K;

        let mut prover_transcript = Blake2bProver::new(b"");
        let mut prover_accumulator = ProverOpeningAccumulator::new();

        let r_cycle: Vec<<Fr as JoltField>::Challenge> =
            prover_transcript.challenge_vector_optimized::<Fr>(log_T);
        let r_address: Vec<<Fr as JoltField>::Challenge> =
            prover_transcript.challenge_vector_optimized::<Fr>(log_K);

        let config = OneHotConfig { log_k_chunk };
        let one_hot_params = OneHotParams::from_config_and_log_K(&config, log_K);
        println!("d={}", one_hot_params.instruction_d);

        let lookup_indices: Vec<usize> = (0..T).map(|_| (rng.next_u32() as usize) % K).collect();
        let H_indices: Vec<Vec<Option<u8>>> = (0..one_hot_params.instruction_d)
            .map(|i| {
                lookup_indices
                    .par_iter()
                    .with_min_len(par_enabled())
                    .map(|lookup_index| {
                        Some(one_hot_params.lookup_index_chunk(*lookup_index as u64, i))
                    })
                    .collect()
            })
            .collect();

        let ra_claim = one_hot_poly_evaluate(&lookup_indices, &r_cycle, &r_address);

        let params = RaSumcheckParams {
            r_cycle: OpeningPoint::<BIG_ENDIAN, Fr>::new(r_cycle),
            r_address: OpeningPoint::<BIG_ENDIAN, Fr>::new(r_address),
            one_hot_params: one_hot_params.clone(),
            ra_claim,
            polynomial_types: vec![
                CommittedPoly::NodeOutputRaD(0, 0),
                CommittedPoly::NodeOutputRaD(0, 1),
                CommittedPoly::NodeOutputRaD(0, 2),
                CommittedPoly::NodeOutputRaD(0, 3),
                CommittedPoly::NodeOutputRaD(0, 4),
                CommittedPoly::NodeOutputRaD(0, 5),
                CommittedPoly::NodeOutputRaD(0, 6),
                CommittedPoly::NodeOutputRaD(0, 7),
            ],
        };

        let mut prover = RaSumcheckProver::gen(params, H_indices.clone());

        let (proof, _) =
            Sumcheck::prove(&mut prover, &mut prover_accumulator, &mut prover_transcript);

        let mut verifier_transcript = Blake2bVerifier::new(b"", prover_transcript.narg().to_vec());
        let mut verifier_accumulator = VerifierOpeningAccumulator::new();
        let _r_x: Vec<<Fr as JoltField>::Challenge> =
            verifier_transcript.challenge_vector_optimized::<Fr>(log_T);
        let _r_words: Vec<<Fr as JoltField>::Challenge> =
            verifier_transcript.challenge_vector_optimized::<Fr>(log_K);

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

        for i in 0..one_hot_params.instruction_d {
            let id = OpeningId::new(
                CommittedPoly::NodeOutputRaD(0, i),
                SumcheckId::RaVirtualization,
            );
            let (opening_point, claim) = verifier_accumulator.get_committed_polynomial_opening(id);
            let H_index: Vec<Option<u16>> = H_indices[i]
                .iter()
                .map(|v| v.map(|new_v| new_v as u16))
                .collect();
            let one_hot_polynomial =
                OneHotPolynomial::<Fr>::from_indices(H_index, (log_k_chunk as usize).pow2());
            assert_eq!(
                one_hot_polynomial.evaluate(&opening_point.r),
                claim,
                "ra_{i} claim"
            );
        }
    }
}

pub mod prover;
