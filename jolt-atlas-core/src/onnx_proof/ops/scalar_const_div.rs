use crate::onnx_proof::{ops::OperatorProofTrait, ProofId, ProofType, Prover, Verifier};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::{Operator, ScalarConstDiv},
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
        sumcheck::{Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for ScalarConstDiv {
    #[tracing::instrument(skip_all, name = "ScalarConstDiv::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let mut results = Vec::new();

        // Execution proof
        let params = ScalarConstDivParams::new(node.clone(), &prover.accumulator);
        let mut exec_sumcheck = ScalarConstDivProver::initialize(&prover.trace, params);
        let (proof, _) = Sumcheck::prove(
            &mut exec_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), proof));
        results
    }

    #[tracing::instrument(skip_all, name = "ScalarConstDiv::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // Execution verification
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let exec_sumcheck = ScalarConstDivVerifier::new(node.clone(), &verifier.accumulator);
        Sumcheck::verify(
            proof,
            &exec_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        Ok(())
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPolynomial> {
        let polys = vec![CommittedPolynomial::ScalarConstDivNodeRemainder(node.idx)];
        polys
    }
}

const DEGREE_BOUND: usize = 2;

/// Parameters for proving division by a scalar constant.
///
/// For division a/b where b is a constant, proves a = b*q + R where R is the remainder.
/// This is more efficient than general division since the divisor is known.
#[derive(Clone)]
pub struct ScalarConstDivParams<F: JoltField> {
    r_node_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
    scalar_const_divisor: i32,
}

impl<F: JoltField> ScalarConstDivParams<F> {
    /// Create new scalar constant division parameters from a computation node and opening accumulator.
    pub fn new(computation_node: ComputationNode, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let r_node_output = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(computation_node.idx),
                SumcheckId::Execution,
            )
            .0
            .r;
        let Operator::ScalarConstDiv(scalar_const_div) = &computation_node.operator else {
            panic!("Expected ScalarConstDiv operator")
        };
        Self {
            r_node_output,
            scalar_const_divisor: scalar_const_div.divisor,
            computation_node,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ScalarConstDivParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let q_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;
        q_claim * F::from_i32(self.scalar_const_divisor)
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.computation_node.num_output_elements().log_2()
    }
}

/// Prover state for scalar constant division sumcheck protocol.
///
/// Maintains the equality polynomial, operand polynomial, and remainder R
/// needed to prove the division relation: operand = divisor * q + R where divisor is constant.
pub struct ScalarConstDivProver<F: JoltField> {
    params: ScalarConstDivParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    left_operand: MultilinearPolynomial<F>,
    R: MultilinearPolynomial<F>,
}

impl<F: JoltField> ScalarConstDivProver<F> {
    /// Initialize the prover with trace data and parameters.
    #[tracing::instrument(skip_all)]
    pub fn initialize(trace: &Trace, params: ScalarConstDivParams<F>) -> Self {
        let eq_r_node_output =
            GruenSplitEqPolynomial::new(&params.r_node_output, BindingOrder::LowToHigh);
        let LayerData { operands, .. } = Trace::layer_data(trace, &params.computation_node);
        let [left_operand] = operands[..] else {
            panic!("Expected one operands for ScalarConstDiv operation")
        };
        let b = params.scalar_const_divisor;
        let R_tensor = {
            let data: Vec<i32> = left_operand
                .iter()
                .map(|&a| {
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
        let R = MultilinearPolynomial::from(R_tensor);
        Self {
            params,
            eq_r_node_output,
            left_operand,
            R,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ScalarConstDivProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            eq_r_node_output,
            left_operand,
            R,
            ..
        } = self;
        let [q_constant] = eq_r_node_output.par_fold_out_in_unreduced::<9, 1>(&|g| {
            let lo0 = left_operand.get_bound_coeff(2 * g);
            let R0 = R.get_bound_coeff(2 * g);
            let c0 = lo0 - R0;
            [c0]
        });
        eq_r_node_output.gruen_poly_deg_2(q_constant, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_node_output.bind(r_j);
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
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            opening_point.clone(),
            self.left_operand.final_sumcheck_claim(),
        );
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::ScalarConstDivNodeRemainder(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.r.clone(),
            self.R.final_sumcheck_claim(),
        );
        accumulator.cache_virtual_operand_claims(transcript, &self.params.computation_node);
    }
}

/// Verifier for scalar constant division sumcheck protocol.
///
/// Verifies that the prover's sumcheck messages are consistent with the claimed
/// division by scalar constant output and the division relation.
pub struct ScalarConstDivVerifier<F: JoltField> {
    params: ScalarConstDivParams<F>,
}

impl<F: JoltField> ScalarConstDivVerifier<F> {
    /// Create a new verifier for the scalar constant division operation.
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = ScalarConstDivParams::new(computation_node, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ScalarConstDivVerifier<F> {
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
        let R_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::ScalarConstDivNodeRemainder(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;
        let [left_operand_claim] = accumulator.get_operand_claims(self.params.computation_node.idx);
        eq_eval * (left_operand_claim - R_claim)
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
            CommittedPolynomial::ScalarConstDivNodeRemainder(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.r.clone(),
        );
        accumulator.append_operand_claims(transcript, self.params.computation_node.idx);
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{model::test::ModelBuilder, model::Model, tensor::Tensor};
    use rand::{rngs::StdRng, SeedableRng};

    fn scalar_const_div_model(T: usize, divisor: i32) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![T]);
        let res = b.scalar_const_div(i, divisor);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_scalar_const_div() {
        let T = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::<i32>::random_small(&mut rng, &[T]);
        let model = scalar_const_div_model(T, 128);
        unit_test_op(model, &[input]);
    }
}
