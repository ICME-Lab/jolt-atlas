use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::VirtualPolynomial;
use rayon::prelude::*;
use std::iter::zip;

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
};

/// Degree bound of the sumcheck round polynomials in [`HammingBooleanitySumcheckVerifier`].
const DEGREE_BOUND: usize = 3;

pub struct HammingBooleanitySumcheckParams<F: JoltField> {
    pub d: usize,
    pub num_rounds: usize,
    pub gamma_powers: Vec<F>,
    pub polynomial_types: Vec<VirtualPolynomial>,
    pub sumcheck_id: SumcheckId,
    pub r_cycle: Vec<F::Challenge>,
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

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

#[derive(Allocative)]
pub struct HammingBooleanitySumcheckProver<F: JoltField> {
    hw: Vec<MultilinearPolynomial<F>>,
    eq_r: GruenSplitEqPolynomial<F>,
    #[allocative(skip)]
    params: HammingBooleanitySumcheckParams<F>,
}

impl<F: JoltField> HammingBooleanitySumcheckProver<F> {
    pub fn gen(params: HammingBooleanitySumcheckParams<F>, G: Vec<Vec<F>>) -> Self {
        let hw = G.into_iter().map(MultilinearPolynomial::from).collect();
        let eq_r = GruenSplitEqPolynomial::new(&params.r_cycle, BindingOrder::LowToHigh);

        Self { hw, eq_r, params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for HammingBooleanitySumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "HammingBooleanitySumcheckProver::compute_message", fields(variant = ?self.params.sumcheck_id))]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self { hw, eq_r, .. } = self;

        let [q_constant, q_quadratic] = hw
            .par_iter()
            .zip(self.params.gamma_powers.par_iter())
            .map(|(hw_d, &gamma)| {
                let [qd_c, qd_q] = eq_r.par_fold_out_in_unreduced::<9, 2>(&|g| {
                    let hw0 = hw_d.get_bound_coeff(2 * g);
                    let hw1 = hw_d.get_bound_coeff(2 * g + 1);
                    let a = hw1 - hw0;

                    let c0 = hw0 * (hw0 - F::one());
                    let e = a * a;
                    [c0, e]
                });
                [qd_c * gamma, qd_q * gamma]
            })
            .reduce(
                || [F::zero(), F::zero()],
                |acc, new| [acc[0] + new[0], acc[1] + new[1]],
            );

        eq_r.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
    }

    #[tracing::instrument(skip_all, name = "HammingBooleanitySumcheckProver::ingest_challenge", fields(variant = ?self.params.sumcheck_id))]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.hw
            .par_iter_mut()
            .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh));
        self.eq_r.bind(r_j);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let claims: Vec<F> = self.hw.iter().map(|hw| hw.final_sumcheck_claim()).collect();

        self.params
            .polynomial_types
            .iter()
            .zip(claims)
            .for_each(|(&poly_type, claim)| {
                accumulator.append_virtual(
                    transcript,
                    poly_type,
                    self.params.sumcheck_id,
                    opening_point.clone(),
                    claim,
                );
            });
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct HammingBooleanitySumcheckVerifier<F: JoltField> {
    params: HammingBooleanitySumcheckParams<F>,
}

impl<F: JoltField> HammingBooleanitySumcheckVerifier<F> {
    pub fn new(params: HammingBooleanitySumcheckParams<F>) -> Self {
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for HammingBooleanitySumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let hw_claim = (0..self.params.d).map(|i| {
            accumulator
                .get_virtual_polynomial_opening(
                    self.params.polynomial_types[i],
                    self.params.sumcheck_id,
                )
                .1
        });

        let r = self.params.normalize_opening_point(sumcheck_challenges);
        // Compute batched claim: eq(r_cycle, r) * sum_{i=0}^{d-1} gamma^i * ( hw_i * (hw_i - 1) )
        EqPolynomial::mle(&self.params.r_cycle, &r.r)
            * zip(hw_claim, &self.params.gamma_powers)
                .map(|(hw_claim, gamma)| (hw_claim * (hw_claim - F::one())) * gamma)
                .sum::<F>()
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r = self.params.normalize_opening_point(sumcheck_challenges);

        self.params.polynomial_types.iter().for_each(|&poly_type| {
            accumulator.append_virtual(transcript, poly_type, self.params.sumcheck_id, r.clone());
        });
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    use crate::{
        subprotocols::sumcheck::Sumcheck,
        transcripts::{Blake2bTranscript, Transcript},
    };

    use super::*;

    #[test]
    fn test_hamming_booleanity_sumcheck() {
        let mut rng = StdRng::seed_from_u64(0x123);
        let log_T = 5;
        let T = 1 << log_T;

        let mut prover_transcript = Blake2bTranscript::default();
        let mut prover_accumulator = ProverOpeningAccumulator::new();
        let r_cycle: Vec<<Fr as JoltField>::Challenge> =
            prover_transcript.challenge_vector_optimized::<Fr>(log_T);

        let params = HammingBooleanitySumcheckParams {
            d: 1,
            num_rounds: log_T,
            gamma_powers: vec![Fr::from_u8(1)],
            polynomial_types: vec![VirtualPolynomial::HammingWeight],
            sumcheck_id: SumcheckId::RamHammingBooleanity,
            r_cycle,
        };

        let hw: Vec<Fr> = (0..T).map(|_| Fr::from(rng.next_u32() % 2)).collect();

        let mut prover = HammingBooleanitySumcheckProver::gen(params, vec![hw]);

        let (proof, _) =
            Sumcheck::prove(&mut prover, &mut prover_accumulator, &mut prover_transcript);

        let mut verifier_transcript = Blake2bTranscript::default();
        let mut verifier_accumulator = VerifierOpeningAccumulator::new();
        let _r_cycle: Vec<<Fr as JoltField>::Challenge> =
            verifier_transcript.challenge_vector_optimized::<Fr>(log_T);

        // Take claims
        for (key, (_, value)) in &prover_accumulator.openings {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier_accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }

        let verifier = HammingBooleanitySumcheckVerifier::new(prover.params);
        Sumcheck::verify(
            &proof,
            &verifier,
            &mut verifier_accumulator,
            &mut verifier_transcript,
        )
        .unwrap();
    }
}
