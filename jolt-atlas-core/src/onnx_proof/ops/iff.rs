use crate::onnx_proof::{ops::OperatorHandler, ProofId, ProofType, Prover, Verifier};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Iff,
};
use common::VirtualPolynomial;
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
        sumcheck::SumcheckInstanceProof,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

use crate::impl_standard_sumcheck_handler;

impl_standard_sumcheck_handler!(Iff, IffParams, IffProver, IffVerifier);

const DEGREE_BOUND: usize = 3;

#[derive(Clone)]
pub struct IffParams<F: JoltField> {
    r_node_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
}

impl<F: JoltField> IffParams<F> {
    pub fn new(computation_node: ComputationNode, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let r_node_output = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(computation_node.idx),
                SumcheckId::Execution,
            )
            .0
            .r;
        Self {
            r_node_output,
            computation_node,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for IffParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, iff_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(self.computation_node.idx),
            SumcheckId::Execution,
        );
        iff_claim
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.computation_node.num_output_elements().log_2()
    }
}

pub struct IffProver<F: JoltField> {
    params: IffParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    mask_operand: MultilinearPolynomial<F>,
    a_operand: MultilinearPolynomial<F>,
    b_operand: MultilinearPolynomial<F>,
}

impl<F: JoltField> IffProver<F> {
    pub fn initialize(trace: &Trace, params: IffParams<F>) -> Self {
        let eq_r_node_output =
            GruenSplitEqPolynomial::new(&params.r_node_output, BindingOrder::LowToHigh);
        let LayerData {
            operands,
            output: _output,
        } = Trace::layer_data(trace, &params.computation_node);
        let [mask_operand, a_operand, b_operand] = operands[..] else {
            panic!("Expected three operands for Iff operation")
        };
        let mask_operand = MultilinearPolynomial::from(mask_operand.clone());
        let a_operand = MultilinearPolynomial::from(a_operand.clone());
        let b_operand = MultilinearPolynomial::from(b_operand.clone());

        #[cfg(test)]
        {
            use joltworks::poly::multilinear_polynomial::PolynomialEvaluation;
            let eq = EqPolynomial::evals(&params.r_node_output);
            let output = MultilinearPolynomial::from(_output.clone());
            let claim = (0..a_operand.len())
                .map(|i| {
                    let mask = mask_operand.get_bound_coeff(i);
                    let a = a_operand.get_bound_coeff(i);
                    let b = b_operand.get_bound_coeff(i);

                    eq[i] * (mask * a + (F::one() - mask) * b)
                })
                .sum();

            assert_eq!(output.evaluate(&params.r_node_output), claim)
        }
        Self {
            params,
            eq_r_node_output,
            mask_operand,
            a_operand,
            b_operand,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for IffProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            eq_r_node_output,
            mask_operand,
            a_operand,
            b_operand,
            ..
        } = self;
        let [q_constant, q_quadratic] = eq_r_node_output.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let mask0 = mask_operand.get_bound_coeff(2 * g);
            let mask1 = mask_operand.get_bound_coeff(2 * g + 1);
            let mask_inf = mask1 - mask0;

            let a0 = a_operand.get_bound_coeff(2 * g);
            let a1 = a_operand.get_bound_coeff(2 * g + 1);
            let a_inf = a1 - a0;

            let b0 = b_operand.get_bound_coeff(2 * g);
            let b1 = b_operand.get_bound_coeff(2 * g + 1);
            let b_inf = b1 - b0;

            let c0 = mask0 * a0 + (F::one() - mask0) * b0;
            let f = (F::one() - mask1) - (F::one() - mask0);
            let e = mask_inf * a_inf + f * b_inf;
            [c0, e]
        });
        eq_r_node_output.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_node_output.bind(r_j);
        self.mask_operand
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.a_operand.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.b_operand.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            opening_point.clone(),
            self.mask_operand.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
            SumcheckId::Execution,
            opening_point.clone(),
            self.a_operand.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[2]),
            SumcheckId::Execution,
            opening_point,
            self.b_operand.final_sumcheck_claim(),
        );
    }
}

pub struct IffVerifier<F: JoltField> {
    params: IffParams<F>,
}

impl<F: JoltField> IffVerifier<F> {
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = IffParams::new(computation_node, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for IffVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r_node_output = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .0
            .r;
        let r_node_output_prime = self.params.normalize_opening_point(sumcheck_challenges).r;
        let eq_eval = EqPolynomial::mle(&r_node_output, &r_node_output_prime);
        let mask_operand_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
                SumcheckId::Execution,
            )
            .1;
        let a_operand_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
                SumcheckId::Execution,
            )
            .1;
        let b_operand_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[2]),
                SumcheckId::Execution,
            )
            .1;

        eq_eval
            * (mask_operand_claim * a_operand_claim
                + (F::one() - mask_operand_claim) * b_operand_claim)
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
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
            SumcheckId::Execution,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[2]),
            SumcheckId::Execution,
            opening_point,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use atlas_onnx_tracer::{
        model::{
            self,
            trace::{LayerData, Trace},
        },
        tensor::Tensor,
    };
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
    fn test_iff() {
        let log_T = 16;
        let T = 1 << log_T;
        let mut rng = StdRng::seed_from_u64(0x899);
        let input = Tensor::<i32>::random(&mut rng, &[T]);
        let model = model::test::iff_model(&mut rng, T);
        let trace = model.trace(&[input]);

        let prover_transcript = &mut Blake2bTranscript::new(&[]);
        let mut prover_opening_accumulator: ProverOpeningAccumulator<Fr> =
            ProverOpeningAccumulator::new(log_T);
        let verifier_transcript = &mut Blake2bTranscript::new(&[]);
        let mut verifier_opening_accumulator: VerifierOpeningAccumulator<Fr> =
            VerifierOpeningAccumulator::new(log_T);

        let r_node_output: Vec<<Fr as JoltField>::Challenge> =
            prover_transcript.challenge_vector_optimized::<Fr>(log_T);
        let _r_node_output: Vec<<Fr as JoltField>::Challenge> =
            verifier_transcript.challenge_vector_optimized::<Fr>(log_T);

        let output_index = model.outputs()[0];
        let computation_node = &model[output_index];
        let LayerData {
            operands: _,
            output,
        } = Trace::layer_data(&trace, computation_node);

        let iff_claim = MultilinearPolynomial::from(output.clone()).evaluate(&r_node_output);
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.clone().into(),
            iff_claim,
        );

        let params: IffParams<Fr> =
            IffParams::new(computation_node.clone(), &prover_opening_accumulator);
        let mut prover_sumcheck = IffProver::initialize(&trace, params);

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
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.into(),
        );

        let verifier_sumcheck =
            IffVerifier::new(computation_node.clone(), &verifier_opening_accumulator);
        let res = Sumcheck::verify(
            &proof,
            &verifier_sumcheck,
            &mut verifier_opening_accumulator,
            verifier_transcript,
        );
        prover_transcript.compare_to(verifier_transcript.clone());
        let r_sumcheck_verif = res.unwrap();
        assert_eq!(r_sumcheck, r_sumcheck_verif);
    }
}
