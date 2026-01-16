use crate::onnx_proof::{ops::OperatorProofTrait, ProofId, ProofType, Prover, Verifier};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Div,
    tensor::Tensor,
};
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
        sumcheck::SumcheckInstanceProof,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

use crate::impl_standard_sumcheck_proof_api;

impl_standard_sumcheck_proof_api!(Div, DivParams, DivProver, DivVerifier);

// TODO: Reduce two claims to 1 via 4.5.2 PAZK for Quotient polynomial openings
// TODO: Commit to polynomials q and R
// TODO: Prove R is well formed

const DEGREE_BOUND: usize = 3;

#[derive(Clone)]
pub struct DivParams<F: JoltField> {
    r_node_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
}

impl<F: JoltField> DivParams<F> {
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

impl<F: JoltField> SumcheckInstanceParams<F> for DivParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.computation_node.num_output_elements().log_2()
    }
}

pub struct DivProver<F: JoltField> {
    params: DivParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    left_operand: MultilinearPolynomial<F>,
    right_operand: MultilinearPolynomial<F>,
    q: MultilinearPolynomial<F>,
    R: MultilinearPolynomial<F>,
}

impl<F: JoltField> DivProver<F> {
    pub fn initialize(trace: &Trace, params: DivParams<F>) -> Self {
        let eq_r_node_output =
            GruenSplitEqPolynomial::new(&params.r_node_output, BindingOrder::LowToHigh);
        let LayerData { operands, output } = Trace::layer_data(trace, &params.computation_node);
        let [left_operand, right_operand] = operands[..] else {
            panic!("Expected two operands for Div operation")
        };
        let R_tensor = {
            let data: Vec<i32> = left_operand
                .iter()
                .zip(right_operand.iter())
                .map(|(&a, &b)| {
                    let mut R = a % b;
                    if (R < 0 && b > 0) || R > 0 && b < 0 {
                        R += b
                    }
                    R
                })
                .collect();
            Tensor::<i32>::construct(data, left_operand.dims().to_vec())
        };
        let left_operand = MultilinearPolynomial::from(left_operand.clone());
        let right_operand = MultilinearPolynomial::from(right_operand.clone());
        let q = MultilinearPolynomial::from(output.clone());
        let R = MultilinearPolynomial::from(R_tensor);
        #[cfg(test)]
        {
            let claim = (0..left_operand.len())
                .map(|i| {
                    let a = left_operand.get_bound_coeff(i);
                    let b = right_operand.get_bound_coeff(i);
                    let q = q.get_bound_coeff(i);
                    let R = R.get_bound_coeff(i);
                    b * q + R - a
                })
                .sum();
            assert_eq!(F::zero(), claim)
        }
        Self {
            params,
            eq_r_node_output,
            left_operand,
            right_operand,
            q,
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
            eq_r_node_output,
            left_operand,
            right_operand,
            q,
            R,
            ..
        } = self;
        let [q_constant, q_quadratic] = eq_r_node_output.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let lo0 = left_operand.get_bound_coeff(2 * g);
            let lo1 = left_operand.get_bound_coeff(2 * g + 1);
            let ro0 = right_operand.get_bound_coeff(2 * g);
            let ro1 = right_operand.get_bound_coeff(2 * g + 1);
            let q0 = q.get_bound_coeff(2 * g);
            let q1 = q.get_bound_coeff(2 * g + 1);
            let R0 = R.get_bound_coeff(2 * g);
            let R1 = R.get_bound_coeff(2 * g + 1);
            let c0 = (ro0 * q0) + R0 - lo0;
            let m1 = ro1 * q1 + R1 - lo1;
            let e = m1 - c0;
            [c0, e]
        });
        eq_r_node_output.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_node_output.bind(r_j);
        self.left_operand
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.right_operand
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.q.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            opening_point.clone(),
            self.left_operand.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
            SumcheckId::Execution,
            opening_point.clone(),
            self.right_operand.final_sumcheck_claim(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::DivNodeQuotient(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.r.clone(),
            self.q.final_sumcheck_claim(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::DivNodeRemainder(self.params.computation_node.idx),
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
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = DivParams::new(computation_node, accumulator);
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
        let r_node_output = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .0
            .r;
        let r_node_output_prime = self.params.normalize_opening_point(sumcheck_challenges).r;
        let eq_eval = EqPolynomial::mle(&r_node_output, &r_node_output_prime);
        let left_operand_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
                SumcheckId::Execution,
            )
            .1;
        let right_operand_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
                SumcheckId::Execution,
            )
            .1;
        let q_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::DivNodeQuotient(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;
        let R_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::DivNodeRemainder(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;
        eq_eval * ((right_operand_claim * q_claim) + R_claim - left_operand_claim)
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
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::DivNodeQuotient(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.r.clone(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::DivNodeRemainder(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.r.clone(),
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
    fn test_div() {
        let log_T = 16;
        let T = 1 << log_T;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::<i32>::random(&mut rng, &[T]);
        let model = model::test::div_model(T);
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

        let div_claim = MultilinearPolynomial::from(output.clone()).evaluate(&r_node_output);
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.clone().into(),
            div_claim,
        );

        let params: DivParams<Fr> =
            DivParams::new(computation_node.clone(), &prover_opening_accumulator);
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
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.into(),
        );

        let verifier_sumcheck =
            DivVerifier::new(computation_node.clone(), &verifier_opening_accumulator);
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
