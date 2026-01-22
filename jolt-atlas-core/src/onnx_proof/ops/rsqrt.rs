use crate::onnx_proof::{ops::OperatorProofTrait, ProofId, ProofType, Prover, Verifier};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Rsqrt,
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
        sumcheck::{Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

// TODO: Handle global scale
const Q: i32 = 128;
const Q_SQUARE: i32 = Q * Q;

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Rsqrt {
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let params = RsqrtParams::new(node.clone(), &prover.accumulator);
        let mut prover_sumcheck =
            RsqrtProver::initialize(&prover.trace, &mut prover.transcript, params);
        let (proof, _) = Sumcheck::prove(
            &mut prover_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        vec![(ProofId(node.idx, ProofType::Execution), proof)]
    }

    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let verifier_sumcheck = RsqrtVerifier::new(
            node.clone(),
            &verifier.accumulator,
            &mut verifier.transcript,
        );
        Sumcheck::verify(
            proof,
            &verifier_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;
        Ok(())
    }
}

// Decomposes rsqrt into an inverse and a square root where the inverse `inv` of x is such that:
// 1 = x * inv + r_i  where 0 <= r_i < x
// and the square root `sqrt` of inv is such that:
// inv = sqrt * sqrt + r_s  where 0 <= r_s < 2 * sqrt + 1

// HANDLING SCALE:
// Any input to the model is quantized by a scale factor of Q
// Therefore, for an input x_real, the quantized representation is x = x_real * Q
// The inverse inv_real of x_real is given by:
// inv_real = 1 / x_real = Q / x
// Therefore, the quantized representation of inv_real is:
// inv = Q * inv_real = Q^2 / x
// Similarly, for the square root sqrt_real of inv_real:
// sqrt_real = sqrt(inv_real) = sqrt(Q / x)
// Therefore, the quantized representation of sqrt_real is:
// sqrt = Q * sqrt_real = Q * sqrt(Q / x) = sqrt(Q^3 / x) = sqrt(Q * inv)

// The two relations that we will batck together in a sumcheck instance are:
// - 0 = x * inv + r_i - Q^2
// - 0 = Q * inv - sqrt * sqrt - r_s
// TODO: Reduce two claims to 1 via 4.5.2 PAZK for Quotient polynomial openings
// TODO: Commit to polynomials i, s, r_i, r_s
// TODO: Prove r_i and r_s are well formed via range checks

// Possible optimization is to only commit to the result and a remainder,
// and find the associated range check for the unique remainder.

const DEGREE_BOUND: usize = 3;

#[derive(Clone)]
pub struct RsqrtParams<F: JoltField> {
    r_node_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
}

impl<F: JoltField> RsqrtParams<F> {
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

impl<F: JoltField> SumcheckInstanceParams<F> for RsqrtParams<F> {
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

pub struct RsqrtProver<F: JoltField> {
    params: RsqrtParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    left_operand: MultilinearPolynomial<F>,
    inv: MultilinearPolynomial<F>,
    rsqrt: MultilinearPolynomial<F>,
    r_i: MultilinearPolynomial<F>,
    r_s: MultilinearPolynomial<F>,
    // folding challenge
    gamma: F,
}

impl<F: JoltField> RsqrtProver<F> {
    pub fn initialize<T: Transcript>(
        trace: &Trace,
        transcript: &mut T,
        params: RsqrtParams<F>,
    ) -> Self {
        let eq_r_node_output =
            GruenSplitEqPolynomial::new(&params.r_node_output, BindingOrder::LowToHigh);
        let LayerData { operands, output } = Trace::layer_data(trace, &params.computation_node);
        let [left_operand] = operands[..] else {
            panic!("Expected one operand for Rsqrt operation")
        };
        let inv_data: Vec<i32> = left_operand.iter().map(|&x| Q_SQUARE / x).collect();
        let ri_data: Vec<i32> = left_operand.iter().map(|&x| Q_SQUARE % x).collect();
        let rs_data: Vec<i32> = inv_data
            .iter()
            .zip(output.iter())
            .map(|(&inv, &sqrt)| inv - sqrt * sqrt)
            .collect();

        let left_operand = MultilinearPolynomial::from(left_operand.clone());
        let inv = MultilinearPolynomial::from(inv_data.clone());
        let r_i = MultilinearPolynomial::from(ri_data.clone());

        let rsqrt = MultilinearPolynomial::from(output.clone());
        let r_s = MultilinearPolynomial::from(rs_data.clone());
        #[cfg(test)]
        {
            let claim_inv = (0..left_operand.len())
                .map(|i| {
                    let a: F = left_operand.get_bound_coeff(i);
                    let inv = inv.get_bound_coeff(i);
                    let r_i: F = r_i.get_bound_coeff(i);
                    // range checking
                    assert!(r_i.to_u64().unwrap() < a.to_u64().unwrap());

                    a * inv + r_i - F::from_i32(Q_SQUARE)
                })
                .sum();
            assert_eq!(F::zero(), claim_inv);

            let claim_sqrt = (0..left_operand.len())
                .map(|i| {
                    let inv = inv.get_bound_coeff(i);
                    let sqrt: F = rsqrt.get_bound_coeff(i);
                    let r_s: F = r_s.get_bound_coeff(i);
                    // range checking
                    assert!(r_s.to_u64().unwrap() <= 2 * sqrt.to_u64().unwrap());

                    sqrt * sqrt + r_s - F::from_i32(Q) * inv
                })
                .sum();
            assert_eq!(F::zero(), claim_sqrt)
        }

        let gamma = transcript.challenge_scalar();
        Self {
            params,
            eq_r_node_output,
            left_operand,
            inv,
            r_i,
            rsqrt,
            r_s,
            gamma,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RsqrtProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            eq_r_node_output,
            left_operand,
            inv,
            rsqrt,
            r_i,
            r_s,
            ..
        } = self;
        let [q_constant, q_quadratic] = eq_r_node_output.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let lo0 = left_operand.get_bound_coeff(2 * g);
            let lo1 = left_operand.get_bound_coeff(2 * g + 1);
            let inv0 = inv.get_bound_coeff(2 * g);
            let inv1 = inv.get_bound_coeff(2 * g + 1);
            let r_i0 = r_i.get_bound_coeff(2 * g);
            let r_i1 = r_i.get_bound_coeff(2 * g + 1);

            let rsqrt0 = rsqrt.get_bound_coeff(2 * g);
            let rsqrt1 = rsqrt.get_bound_coeff(2 * g + 1);
            let r_s0 = r_s.get_bound_coeff(2 * g);
            let r_s1 = r_s.get_bound_coeff(2 * g + 1);

            let c0 = lo0 * inv0 + r_i0 - F::from_i32(Q_SQUARE);
            let m1 = lo1 * inv1 + r_i1 - F::from_i32(Q_SQUARE);

            let c1 = rsqrt0 * rsqrt0 + r_s0 - F::from_i32(Q) * inv0;
            let m2 = rsqrt1 * rsqrt1 + r_s1 - F::from_i32(Q) * inv1;

            let e = m1 - c0 + self.gamma * (m2 - c1);
            [c0 + self.gamma * c1, e]
        });
        eq_r_node_output.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_node_output.bind(r_j);
        self.left_operand
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.inv.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.rsqrt.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.r_i.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.r_s.bind_parallel(r_j, BindingOrder::LowToHigh);
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
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RsqrtNodeInv(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.r.clone(),
            self.inv.final_sumcheck_claim(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RsqrtNodeRsqrt(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.r.clone(),
            self.rsqrt.final_sumcheck_claim(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RsqrtNodeRi(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.r.clone(),
            self.r_i.final_sumcheck_claim(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RsqrtNodeRs(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.r.clone(),
            self.r_s.final_sumcheck_claim(),
        );
    }
}

pub struct RsqrtVerifier<F: JoltField> {
    params: RsqrtParams<F>,
    gamma: F,
}

impl<F: JoltField> RsqrtVerifier<F> {
    pub fn new<T: Transcript>(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        let params = RsqrtParams::new(computation_node, accumulator);
        let gamma = transcript.challenge_scalar();
        Self { params, gamma }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for RsqrtVerifier<F> {
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
        let inv_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::RsqrtNodeInv(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;
        let r_i_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::RsqrtNodeRi(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;
        let rsqrt_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::RsqrtNodeRsqrt(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;
        let r_s_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::RsqrtNodeRs(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;

        eq_eval
            * (left_operand_claim * inv_claim + r_i_claim - F::from_i32(Q_SQUARE)
                + self.gamma * (rsqrt_claim * rsqrt_claim + r_s_claim - F::from_i32(Q) * inv_claim))
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
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RsqrtNodeInv(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.r.clone(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RsqrtNodeRsqrt(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.r.clone(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RsqrtNodeRi(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.r.clone(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RsqrtNodeRs(self.params.computation_node.idx),
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
    fn test_rsqrt() {
        let log_T = 16;
        let T = 1 << log_T;
        let mut rng = StdRng::seed_from_u64(0x888);
        let mut input = Tensor::<i32>::random(&mut rng, &[T]);
        // avoid inputs <= 0
        input.iter_mut().for_each(|x| {
            *x = (*x).abs() + 1; // avoid zero input
        });
        let model = model::test::rsqrt_model(T);
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

        let rsqrt_claim = MultilinearPolynomial::from(output.clone()).evaluate(&r_node_output);
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.clone().into(),
            rsqrt_claim,
        );

        let params: RsqrtParams<Fr> =
            RsqrtParams::new(computation_node.clone(), &prover_opening_accumulator);
        let mut prover_sumcheck = RsqrtProver::initialize(&trace, prover_transcript, params);

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

        let verifier_sumcheck = RsqrtVerifier::new(
            computation_node.clone(),
            &verifier_opening_accumulator,
            verifier_transcript,
        );
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
