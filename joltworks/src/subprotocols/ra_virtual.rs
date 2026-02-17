use std::sync::Arc;

use crate::{
    config::OneHotParams,
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        ra_poly::RaPolynomial,
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        mles_product_sum::compute_mles_product_sum,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
};
use common::CommittedPolynomial;
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
    pub polynomial_types: Vec<CommittedPolynomial>,
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

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

pub struct RaSumcheckProver<F: JoltField> {
    ra_i_polys: Vec<RaPolynomial<u8, F>>,
    eq_poly: GruenSplitEqPolynomial<F>,
    params: RaSumcheckParams<F>,
}

impl<F: JoltField> RaSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "RaSumcheckProver::gen")]
    pub fn gen(params: RaSumcheckParams<F>, H_indices: Vec<Vec<Option<u8>>>) -> Self {
        // Compute r_address_chunks with proper padding
        let r_address_chunks = params
            .one_hot_params
            .compute_r_address_chunks::<F>(&params.r_address.r);

        let ra_i_polys = H_indices
            .into_par_iter()
            .enumerate()
            .map(|(i, lookup_indices)| {
                let eq_evals = EqPolynomial::evals(&r_address_chunks[i]);
                RaPolynomial::new(Arc::new(lookup_indices), eq_evals)
            })
            .collect();

        Self {
            ra_i_polys,
            eq_poly: GruenSplitEqPolynomial::new(&params.r_cycle.r, BindingOrder::LowToHigh),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RaSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "RaSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let ra_i_polys = &self.ra_i_polys;
        let eq_poly = &self.eq_poly;
        compute_mles_product_sum(ra_i_polys, previous_claim, eq_poly)
    }

    #[tracing::instrument(skip_all, name = "RaSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.ra_i_polys
            .iter_mut()
            .for_each(|p| p.bind_parallel(r_j, BindingOrder::LowToHigh));
        self.eq_poly.bind(r_j);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = self.params.normalize_opening_point(sumcheck_challenges);

        // Compute r_address_chunks with proper padding
        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(&self.params.r_address.r);

        for (i, r_address) in r_address_chunks.into_iter().enumerate() {
            let claim = self.ra_i_polys[i].final_sumcheck_claim();
            accumulator.append_sparse(
                transcript,
                vec![self.params.polynomial_types[i]],
                SumcheckId::RaVirtualization,
                r_address,
                r_cycle.r.clone(),
                vec![claim],
            );
        }
    }
}

pub struct RaSumcheckVerifier<F: JoltField> {
    params: RaSumcheckParams<F>,
}

impl<F: JoltField> RaSumcheckVerifier<F> {
    pub fn new(params: RaSumcheckParams<F>) -> Self {
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for RaSumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = self.params.normalize_opening_point(sumcheck_challenges);
        let eq_eval = EqPolynomial::mle_endian(&self.params.r_cycle, &r);
        let ra_claim_prod: F = (0..self.params.one_hot_params.instruction_d)
            .map(|i| {
                let (_, ra_i_claim) = accumulator.get_committed_polynomial_opening(
                    self.params.polynomial_types[i],
                    SumcheckId::RaVirtualization,
                );
                ra_i_claim
            })
            .product();

        eq_eval * ra_claim_prod
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = self.params.normalize_opening_point(sumcheck_challenges);

        // Compute r_address_chunks with proper padding
        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(&self.params.r_address.r);

        for (i, r_address) in r_address_chunks.iter().enumerate() {
            let opening_point = [r_address.as_slice(), r_cycle.r.as_slice()].concat();

            accumulator.append_sparse(
                transcript,
                vec![self.params.polynomial_types[i]],
                SumcheckId::RaVirtualization,
                opening_point,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use common::CommittedPolynomial;
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    use crate::{
        config::OneHotConfig,
        field::{ChallengeFieldOps, FieldChallengeOps},
        poly::{
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            one_hot_polynomial::OneHotPolynomial,
        },
        subprotocols::sumcheck::Sumcheck,
        transcripts::{Blake2bTranscript, Transcript},
        utils::{index_to_field_bitvector, math::Math},
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

        let mut prover_transcript = Blake2bTranscript::default();
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
                CommittedPolynomial::NodeOutputRaD(0, 0),
                CommittedPolynomial::NodeOutputRaD(0, 1),
                CommittedPolynomial::NodeOutputRaD(0, 2),
                CommittedPolynomial::NodeOutputRaD(0, 3),
                CommittedPolynomial::NodeOutputRaD(0, 4),
                CommittedPolynomial::NodeOutputRaD(0, 5),
                CommittedPolynomial::NodeOutputRaD(0, 6),
                CommittedPolynomial::NodeOutputRaD(0, 7),
            ],
        };

        let mut prover = RaSumcheckProver::gen(params, H_indices.clone());

        let (proof, _) =
            Sumcheck::prove(&mut prover, &mut prover_accumulator, &mut prover_transcript);

        let mut verifier_transcript = Blake2bTranscript::default();
        let mut verifier_accumulator = VerifierOpeningAccumulator::new();
        let _r_x: Vec<<Fr as JoltField>::Challenge> =
            verifier_transcript.challenge_vector_optimized::<Fr>(log_T);
        let _r_words: Vec<<Fr as JoltField>::Challenge> =
            verifier_transcript.challenge_vector_optimized::<Fr>(log_K);

        // Take claims
        for (key, (_, value)) in &prover_accumulator.openings {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier_accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }

        let verifier = RaSumcheckVerifier::new(prover.params);
        Sumcheck::verify(
            &proof,
            &verifier,
            &mut verifier_accumulator,
            &mut verifier_transcript,
        )
        .unwrap();

        for i in 0..one_hot_params.instruction_d {
            let (opening_point, claim) = verifier_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::NodeOutputRaD(0, i),
                SumcheckId::RaVirtualization,
            );
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
