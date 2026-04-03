use crate::onnx_proof::{
    ops::{eval_reduction::NodeEvalReduction, OperatorProofTrait, ReductionFlow},
    range_checking::{
        range_check_operands::ScalarConstDivRangeCheckOperands, RangeCheckEncoding,
        RangeCheckProvider,
    },
    ProofId, ProofType, Prover, Verifier,
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::{Operator, ScalarConstDiv},
    tensor::Tensor,
};
use common::{consts::XLEN, CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    field::{IntoOpening, JoltField},
    lookup_tables::unsigned_less_than::UnsignedLessThanTable,
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
        shout::{self, RaOneHotEncoding},
        sumcheck::{Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for ScalarConstDiv {
    fn reduction_flow(&self) -> ReductionFlow {
        ReductionFlow::Custom
    }

    #[tracing::instrument(skip_all, name = "ScalarConstDiv::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let mut results = Vec::new();

        let params = ScalarConstDivParams::new(node.clone(), &mut prover.transcript);
        let mut exec_sumcheck = ScalarConstDivProver::initialize(&prover.trace, params);
        let (proof, _) = Sumcheck::prove(
            &mut exec_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), proof));
        results.extend(prove_range_and_onehot(node, prover));
        results
    }

    fn prove_with_reduction(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> (
        joltworks::subprotocols::evaluation_reduction::EvalReductionProof<F>,
        Vec<(ProofId, SumcheckInstanceProof<F, T>)>,
    ) {
        let proofs = self.prove(node, prover);
        let eval_reduction_proof = NodeEvalReduction::prove(prover, node);
        (eval_reduction_proof, proofs)
    }

    #[tracing::instrument(skip_all, name = "ScalarConstDiv::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let exec_sumcheck = ScalarConstDivVerifier::new(node.clone(), &mut verifier.transcript);
        Sumcheck::verify(
            proof,
            &exec_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;
        verify_range_and_onehot(node, verifier)?;

        Ok(())
    }

    fn verify_with_reduction(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
        eval_reduction_proof: &joltworks::subprotocols::evaluation_reduction::EvalReductionProof<F>,
    ) -> Result<(), ProofVerifyError> {
        self.verify(node, verifier)?;
        NodeEvalReduction::verify(verifier, node, eval_reduction_proof)
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPolynomial> {
        let encoding = RangeCheckEncoding::<ScalarConstDivRangeCheckOperands>::new(node);
        let d = encoding.one_hot_params().instruction_d;
        (0..d)
            .map(|i| CommittedPolynomial::ScalarConstDivRangeCheckRaD(node.idx, i))
            .collect()
    }
}

const DEGREE_BOUND: usize = 2;

/// Sumcheck parameters for `ScalarConstDiv`.
///
/// Stores the verifier challenge point for the node output together with the
/// computation node metadata and the positive constant divisor used by the
/// execution relation.
#[derive(Clone)]
pub struct ScalarConstDivParams<F: JoltField> {
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
    scalar_const_divisor: i32,
}

impl<F: JoltField> ScalarConstDivParams<F> {
    /// Samples the output challenge point and validates the constant divisor.
    pub fn new<T: Transcript>(computation_node: ComputationNode, transcript: &mut T) -> Self {
        let num_vars = computation_node.pow2_padded_num_output_elements().log_2();
        let r_node_output = transcript.challenge_vector_optimized::<F>(num_vars);
        let Operator::ScalarConstDiv(scalar_const_div) = &computation_node.operator else {
            panic!("Expected ScalarConstDiv operator")
        };
        assert!(
            scalar_const_div.divisor > 0,
            "ScalarConstDiv proof requires a positive divisor, got {0}",
            scalar_const_div.divisor
        );
        Self {
            r_node_output: r_node_output.into(),
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
        let _ = accumulator;
        F::zero()
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.computation_node
            .pow2_padded_num_output_elements()
            .log_2()
    }
}

/// Prover state for the `ScalarConstDiv` execution sumcheck.
///
/// The relation enforces `divisor * q + r - a = 0` at the sampled point, where
/// `q` is the node output and `r` is the virtual remainder reconstructed by the
/// range-check / RA pipeline.
pub struct ScalarConstDivProver<F: JoltField> {
    params: ScalarConstDivParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    left_operand: MultilinearPolynomial<F>,
    q: MultilinearPolynomial<F>,
    remainder: MultilinearPolynomial<F>,
}

impl<F: JoltField> ScalarConstDivProver<F> {
    /// Builds prover polynomials from the execution trace for one `ScalarConstDiv` node.
    #[tracing::instrument(skip_all)]
    pub fn initialize(trace: &Trace, params: ScalarConstDivParams<F>) -> Self {
        let eq_r_node_output =
            GruenSplitEqPolynomial::new(&params.r_node_output.r, BindingOrder::LowToHigh);
        let LayerData { operands, output } = Trace::layer_data(trace, &params.computation_node);
        let [left_operand] = operands[..] else {
            panic!("Expected one operands for ScalarConstDiv operation")
        };
        let b = params.scalar_const_divisor;
        let remainder_tensor = {
            let data: Vec<i32> = left_operand
                .iter()
                .map(|&a| {
                    let mut remainder = a % b;
                    if (remainder < 0 && b > 0) || (remainder > 0 && b < 0) {
                        remainder += b
                    }
                    remainder
                })
                .collect();
            Tensor::<i32>::construct(data, left_operand.dims().to_vec())
        };
        let q = MultilinearPolynomial::from(output.clone());
        Self {
            params,
            eq_r_node_output,
            left_operand: MultilinearPolynomial::from(left_operand.clone()),
            q,
            remainder: MultilinearPolynomial::from(remainder_tensor),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ScalarConstDivProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let [q_constant] = self
            .eq_r_node_output
            .par_fold_out_in_unreduced::<9, 1>(&|g| {
                let lo0 = self.left_operand.get_bound_coeff(2 * g);
                let q0 = self.q.get_bound_coeff(2 * g);
                let r0 = self.remainder.get_bound_coeff(2 * g);
                [F::from_i32(self.params.scalar_const_divisor) * q0 + r0 - lo0]
            });
        self.eq_r_node_output
            .gruen_poly_deg_2(q_constant, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_node_output.bind(r_j);
        self.left_operand
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.q.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.remainder.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            opening_point.clone(),
            self.left_operand.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            opening_point.clone(),
            self.q.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::ScalarConstDivDivisor(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            opening_point.clone(),
            F::from_i32(self.params.scalar_const_divisor),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::DivRemainder(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            opening_point,
            self.remainder.final_sumcheck_claim(),
        );
    }
}

/// Verifier state for the `ScalarConstDiv` execution sumcheck.
///
/// This verifier checks the virtual quotient, divisor, and remainder openings
/// that witness the element-wise relation against the sampled output point.
pub struct ScalarConstDivVerifier<F: JoltField> {
    params: ScalarConstDivParams<F>,
}

impl<F: JoltField> ScalarConstDivVerifier<F> {
    /// Samples the output challenge point for verifying one `ScalarConstDiv` node.
    pub fn new<T: Transcript>(computation_node: ComputationNode, transcript: &mut T) -> Self {
        let params = ScalarConstDivParams::new(computation_node, transcript);
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
        let r_node_output = self.params.r_node_output.r.clone();
        let r_node_output_prime = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let eq_eval = EqPolynomial::mle(&r_node_output, &r_node_output_prime);
        let q_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
                SumcheckId::NodeExecution(self.params.computation_node.idx),
            )
            .1;
        let remainder_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::DivRemainder(self.params.computation_node.idx),
                SumcheckId::NodeExecution(self.params.computation_node.idx),
            )
            .1;
        let left_operand_claim = accumulator.get_node_output_claim(
            self.params.computation_node.inputs[0],
            self.params.computation_node.idx,
        );
        let divisor_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::ScalarConstDivDivisor(self.params.computation_node.idx),
                SumcheckId::NodeExecution(self.params.computation_node.idx),
            )
            .1;
        eq_eval * ((divisor_claim * q_claim) + remainder_claim - left_operand_claim)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::ScalarConstDivDivisor(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::DivRemainder(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            opening_point,
        );
    }
}

fn prove_range_and_onehot<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
    let mut results = Vec::new();

    let rangecheck_provider = RangeCheckProvider::<ScalarConstDivRangeCheckOperands>::new(node);
    let (mut rangecheck_sumcheck, lookup_indices) = rangecheck_provider
        .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
    let (rangecheck_proof, _) = Sumcheck::prove(
        &mut rangecheck_sumcheck,
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    results.push((ProofId(node.idx, ProofType::RangeCheck), rangecheck_proof));

    let encoding = RangeCheckEncoding::<ScalarConstDivRangeCheckOperands>::new(node);
    let [ra_sumcheck, hw_sumcheck, bool_sumcheck] = shout::ra_onehot_provers(
        &encoding,
        &lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let mut instances: Vec<
        Box<dyn joltworks::subprotocols::sumcheck_prover::SumcheckInstanceProver<_, _>>,
    > = vec![ra_sumcheck, hw_sumcheck, bool_sumcheck];
    let (ra_one_hot_proof, _) = joltworks::subprotocols::sumcheck::BatchedSumcheck::prove(
        instances.iter_mut().map(|v| &mut **v as _).collect(),
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    results.push((
        ProofId(node.idx, ProofType::RaOneHotChecks),
        ra_one_hot_proof,
    ));

    results
}

fn verify_range_and_onehot<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError> {
    let rangecheck_proof = verifier
        .proofs
        .get(&ProofId(node.idx, ProofType::RangeCheck))
        .ok_or(ProofVerifyError::MissingProof(node.idx))?;

    let rangecheck_provider = RangeCheckProvider::<ScalarConstDivRangeCheckOperands>::new(node);
    let rangecheck_verifier = rangecheck_provider
        .read_raf_verify::<F, T, UnsignedLessThanTable<XLEN>>(
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
    Sumcheck::verify(
        rangecheck_proof,
        &rangecheck_verifier,
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )?;

    let ra_one_hot_proof = verifier
        .proofs
        .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
        .ok_or(ProofVerifyError::MissingProof(node.idx))?;
    let encoding = RangeCheckEncoding::<ScalarConstDivRangeCheckOperands>::new(node);
    let [ra_sumcheck, hw_sumcheck, bool_sumcheck] =
        shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript);
    let mut instances: Vec<
        Box<dyn joltworks::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<_, _>>,
    > = vec![ra_sumcheck, hw_sumcheck, bool_sumcheck];
    joltworks::subprotocols::sumcheck::BatchedSumcheck::verify(
        ra_one_hot_proof,
        instances.iter_mut().map(|v| &mut **v as _).collect(),
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{model::test::ModelBuilder, model::Model, tensor::Tensor};
    use rand::{rngs::StdRng, SeedableRng};

    fn scalar_const_div_model(t: usize, divisor: i32) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![t]);
        let res = b.scalar_const_div(i, divisor);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_scalar_const_div() {
        let t = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::<i32>::random_small(&mut rng, &[t]);
        let model = scalar_const_div_model(t, 128);
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "non-power-of-two path not fully supported yet"]
    fn test_scalar_const_div_non_power_of_two_input_len() {
        let t = 1000;
        let mut rng = StdRng::seed_from_u64(0x889);
        let input = Tensor::<i32>::random_small(&mut rng, &[t]);
        let model = scalar_const_div_model(t, 128);
        unit_test_op(model, &[input]);
    }
}
