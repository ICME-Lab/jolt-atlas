use crate::onnx_proof::ops::softmax_axes::softmax::SoftmaxIndex;
use atlas_onnx_tracer::tensor::{ops::nonlinearities::SoftmaxTrace, Tensor};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
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

// TODO: Commit to R and also prove R is well formed

const DEGREE_BOUND: usize = 2;
pub const S: usize = 128;

#[derive(Clone)]
pub struct DivParams<F: JoltField> {
    r_feature_output: Vec<F::Challenge>,
    softmax_index: SoftmaxIndex,
}

impl<F: JoltField> DivParams<F> {
    pub fn new(softmax_index: SoftmaxIndex, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let r_feature_output = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxFeatureOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
            )
            .0
            .r;
        Self {
            r_feature_output,
            softmax_index,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for DivParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let q_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxFeatureOutput(
                    self.softmax_index.node_idx,
                    self.softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
            )
            .1;
        let sum_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxSumOutput(
                    self.softmax_index.node_idx,
                    self.softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
            )
            .1;
        q_claim * sum_claim
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.r_feature_output.len()
    }
}

pub struct DivProver<F: JoltField> {
    params: DivParams<F>,
    eq_r_feature_output: GruenSplitEqPolynomial<F>,
    left_operand: MultilinearPolynomial<F>,
    R: MultilinearPolynomial<F>,
}

impl<F: JoltField> DivProver<F> {
    pub fn initialize(trace: &SoftmaxTrace, params: DivParams<F>) -> Self {
        let eq_r_feature_output =
            GruenSplitEqPolynomial::new(&params.r_feature_output, BindingOrder::LowToHigh);
        let left_operand = &trace.exp_q_values;
        let right_operand = trace.exp_sum_q;
        let R_tensor = {
            let data: Vec<i32> = left_operand
                .iter()
                .map(|&a| {
                    let mut R = (a * S as i32) % right_operand;
                    if (R < 0 && right_operand > 0) || R > 0 && right_operand < 0 {
                        R += right_operand
                    }
                    R
                })
                .collect();
            Tensor::<i32>::construct(data, left_operand.dims().to_vec())
        };

        let left_operand = MultilinearPolynomial::from(left_operand.clone());
        let R = MultilinearPolynomial::from(R_tensor);
        Self {
            params,
            eq_r_feature_output,
            left_operand,
            R,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for DivProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            eq_r_feature_output,
            left_operand,
            R,
            ..
        } = self;
        let S_f = F::from_u32(S as u32);
        let [q_constant] = eq_r_feature_output.par_fold_out_in_unreduced::<9, 1>(&|g| {
            let lo0 = left_operand.get_bound_coeff(2 * g);
            let R0 = R.get_bound_coeff(2 * g);
            let c0 = lo0 * S_f - R0;
            [c0]
        });
        eq_r_feature_output.gruen_poly_deg_2(q_constant, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_feature_output.bind(r_j);
        self.left_operand
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.R.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxExponentiationOutput(
                self.params.softmax_index.node_idx,
                self.params.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
            opening_point.clone(),
            self.left_operand.final_sumcheck_claim(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::SoftmaxRemainder(
                self.params.softmax_index.node_idx,
                self.params.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
            opening_point.r.clone(),
            self.R.final_sumcheck_claim(),
        );
    }
}

pub struct DivVerifier<F: JoltField> {
    params: DivParams<F>,
}

impl<F: JoltField> DivVerifier<F> {
    pub fn new(softmax_index: SoftmaxIndex, accumulator: &VerifierOpeningAccumulator<F>) -> Self {
        let params = DivParams::new(softmax_index, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for DivVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r_feature_output = self.params.r_feature_output.clone();
        let r_feature_output_prime = self.params.normalize_opening_point(sumcheck_challenges).r;
        let eq_eval = EqPolynomial::mle(&r_feature_output, &r_feature_output_prime);
        let left_operand_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExponentiationOutput(
                    self.params.softmax_index.node_idx,
                    self.params.softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
            )
            .1;
        let R_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::SoftmaxRemainder(
                    self.params.softmax_index.node_idx,
                    self.params.softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
            )
            .1;
        eq_eval * (left_operand_claim * F::from_u32(S as u32) - R_claim)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::SoftmaxExponentiationOutput(
                self.params.softmax_index.node_idx,
                self.params.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
            opening_point.clone(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::SoftmaxRemainder(
                self.params.softmax_index.node_idx,
                self.params.softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
            opening_point.r.clone(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use atlas_onnx_tracer::tensor::{ops::nonlinearities::softmax_fixed_128, Tensor};
    use common::VirtualPolynomial;
    use joltworks::{
        field::JoltField,
        poly::{
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            opening_proof::{
                OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
                BIG_ENDIAN,
            },
        },
        subprotocols::sumcheck::Sumcheck,
        transcripts::{Blake2bTranscript, Transcript},
    };
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_div() {
        let n: usize = 4;
        let N: usize = 1 << n;

        let mut rng = StdRng::seed_from_u64(0x100081);
        let input = Tensor::random_small(&mut rng, &[N]);

        let (_, trace) = softmax_fixed_128::<true>(&input);
        let trace = trace.unwrap();

        let prover_transcript = &mut Blake2bTranscript::new(&[]);
        let mut prover_opening_accumulator: ProverOpeningAccumulator<Fr> =
            ProverOpeningAccumulator::new();
        let verifier_transcript = &mut Blake2bTranscript::new(&[]);
        let mut verifier_opening_accumulator: VerifierOpeningAccumulator<Fr> =
            VerifierOpeningAccumulator::new();

        let _r_feature_output: Vec<<Fr as JoltField>::Challenge> =
            prover_transcript.challenge_vector_optimized::<Fr>(n);
        let r_feature_output: Vec<<Fr as JoltField>::Challenge> =
            verifier_transcript.challenge_vector_optimized::<Fr>(n);

        let softmax_index = SoftmaxIndex {
            node_idx: 0,
            feature_idx: 0,
        };

        let div_claim =
            MultilinearPolynomial::from(trace.softmax_q.clone()).evaluate(&r_feature_output);
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualPolynomial::SoftmaxFeatureOutput(
                softmax_index.node_idx,
                softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
            r_feature_output.clone().into(),
            div_claim,
        );
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualPolynomial::SoftmaxSumOutput(softmax_index.node_idx, softmax_index.feature_idx),
            SumcheckId::Execution,
            vec![].into(),
            Fr::from_i32(trace.exp_sum_q),
        );

        let params: DivParams<Fr> = DivParams::new(softmax_index, &prover_opening_accumulator);
        let mut prover_sumcheck = DivProver::initialize(&trace, params);

        let (proof, r_sumcheck) = Sumcheck::prove(
            &mut prover_sumcheck,
            &mut prover_opening_accumulator,
            prover_transcript,
        );

        // Take claims
        for (key, (_, value)) in &prover_opening_accumulator.openings {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier_opening_accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }

        verifier_opening_accumulator.append_virtual(
            verifier_transcript,
            VirtualPolynomial::SoftmaxFeatureOutput(
                softmax_index.node_idx,
                softmax_index.feature_idx,
            ),
            SumcheckId::Execution,
            r_feature_output.into(),
        );
        verifier_opening_accumulator.append_virtual(
            verifier_transcript,
            VirtualPolynomial::SoftmaxSumOutput(softmax_index.node_idx, softmax_index.feature_idx),
            SumcheckId::Execution,
            vec![].into(),
        );

        let verifier_sumcheck = DivVerifier::new(softmax_index, &verifier_opening_accumulator);
        let res = Sumcheck::verify(
            &proof,
            &verifier_sumcheck,
            &mut verifier_opening_accumulator,
            verifier_transcript,
        );

        let r_sumcheck_verif = res.unwrap();
        assert_eq!(r_sumcheck, r_sumcheck_verif);
    }
}
