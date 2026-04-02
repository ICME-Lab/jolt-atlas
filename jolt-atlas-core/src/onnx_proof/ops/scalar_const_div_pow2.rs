use crate::onnx_proof::neural_teleport::utils::compute_ra_evals_direct;
use crate::onnx_proof::{
    ops::{eval_reduction::NodeEvalReduction, OperatorProofTrait, ReductionFlow},
    ProofId, ProofType, Prover, Verifier,
};
use crate::utils::adjusted_remainder;
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::{Operator, ScalarConstDivPow2},
    tensor::Tensor,
};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::{IntoOpening, JoltField},
    poly::{
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

const DEGREE_BOUND: usize = 2;

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for ScalarConstDivPow2 {
    fn reduction_flow(&self) -> ReductionFlow {
        ReductionFlow::Custom
    }

    #[tracing::instrument(skip_all, name = "ScalarConstDivPow2::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let mut results = Vec::new();

        let params = ScalarConstDivPow2Params::new(node.clone(), &mut prover.transcript);
        let mut exec_sumcheck = ScalarConstDivPow2Prover::initialize(
            &prover.trace,
            params,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        let (exec_proof, _) = Sumcheck::prove(
            &mut exec_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), exec_proof));

        let encoding = ScalarConstDivPow2RaEncoding::new(node);
        let lookup_indices = scalar_const_div_pow2_lookup_indices(&prover.trace, node);
        let [ra_sumcheck, hw_sumcheck, bool_sumcheck] = shout::ra_onehot_provers(
            &encoding,
            &lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![ra_sumcheck, hw_sumcheck, bool_sumcheck];
        let (ra_one_hot_proof, _) = BatchedSumcheck::prove(
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

    #[tracing::instrument(skip_all, name = "ScalarConstDivPow2::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let exec_sumcheck = ScalarConstDivPow2Verifier::new(
            node.clone(),
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
        Sumcheck::verify(
            proof,
            &exec_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        let ra_one_hot_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let encoding = ScalarConstDivPow2RaEncoding::new(node);
        let [ra_sumcheck, hw_sumcheck, bool_sumcheck] =
            shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript);
        BatchedSumcheck::verify(
            ra_one_hot_proof,
            vec![&*ra_sumcheck, &*hw_sumcheck, &*bool_sumcheck],
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

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
        let encoding = ScalarConstDivPow2RaEncoding::new(node);
        let d = encoding.one_hot_params().instruction_d;
        (0..d)
            .map(|i| CommittedPolynomial::ScalarConstDivPow2RaD(node.idx, i))
            .collect()
    }
}

/// Sumcheck parameters for `ScalarConstDivPow2`.
///
/// Stores the sampled output point together with the constant power-of-two
/// divisor and its log table size used by the one-hot remainder encoding.
#[derive(Clone)]
pub struct ScalarConstDivPow2Params<F: JoltField> {
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
    scalar_const_divisor: i32,
    log_table_size: usize,
}

impl<F: JoltField> ScalarConstDivPow2Params<F> {
    /// Samples the output challenge point and validates the power-of-two divisor.
    pub fn new<T: Transcript>(computation_node: ComputationNode, transcript: &mut T) -> Self {
        let num_vars = computation_node.pow2_padded_num_output_elements().log_2();
        let r_node_output = transcript.challenge_vector_optimized::<F>(num_vars);
        let Operator::ScalarConstDivPow2(op) = &computation_node.operator else {
            panic!("Expected ScalarConstDivPow2 operator")
        };
        let divisor = op.divisor;
        assert!(
            divisor > 0 && (divisor as u32).is_power_of_two(),
            "ScalarConstDivPow2 proof requires a positive power-of-two divisor, got {divisor}"
        );
        Self {
            r_node_output: r_node_output.into(),
            computation_node,
            scalar_const_divisor: divisor,
            log_table_size: (divisor as u32).trailing_zeros() as usize,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ScalarConstDivPow2Params<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let left_operand_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.computation_node.inputs[0]),
                SumcheckId::NodeExecution(self.computation_node.idx),
            )
            .1;
        let quotient_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.computation_node.idx),
                SumcheckId::NodeExecution(self.computation_node.idx),
            )
            .1;
        left_operand_claim - F::from_i32(self.scalar_const_divisor) * quotient_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.log_table_size
    }
}

/// Prover state for the `ScalarConstDivPow2` execution sumcheck.
///
/// The remainder is represented by a one-hot polynomial, so the prover only
/// needs that encoding and the identity polynomial over the lookup domain.
pub struct ScalarConstDivPow2Prover<F: JoltField> {
    params: ScalarConstDivPow2Params<F>,
    remainder_onehot: MultilinearPolynomial<F>,
    identity: IdentityPolynomial<F>,
}

impl<F: JoltField> ScalarConstDivPow2Prover<F> {
    /// Initializes the prover and caches the virtual execution openings used by the sumcheck.
    #[tracing::instrument(skip_all)]
    pub fn initialize(
        trace: &Trace,
        params: ScalarConstDivPow2Params<F>,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let LayerData { operands, output } = Trace::layer_data(trace, &params.computation_node);
        let [left_operand] = operands[..] else {
            panic!("Expected one operand for ScalarConstDivPow2 operation")
        };

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::ScalarConstDivPow2Divisor(params.computation_node.idx),
            SumcheckId::NodeExecution(params.computation_node.idx),
            params.r_node_output.clone(),
            F::from_i32(params.scalar_const_divisor),
        );
        let left_claim =
            MultilinearPolynomial::from(left_operand.clone()).evaluate(&params.r_node_output.r);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(params.computation_node.inputs[0]),
            SumcheckId::NodeExecution(params.computation_node.idx),
            params.r_node_output.clone(),
            left_claim,
        );

        let output_claim =
            MultilinearPolynomial::from(output.clone()).evaluate(&params.r_node_output.r);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(params.computation_node.idx),
            SumcheckId::NodeExecution(params.computation_node.idx),
            params.r_node_output.clone(),
            output_claim,
        );

        let remainder_tensor =
            scalar_const_div_pow2_remainder_tensor(left_operand, params.scalar_const_divisor);
        let remainder_onehot = MultilinearPolynomial::from(compute_ra_evals_direct(
            &params.r_node_output.r,
            &remainder_tensor,
            params.scalar_const_divisor as usize,
        ));

        Self {
            identity: IdentityPolynomial::new(params.log_table_size),
            params,
            remainder_onehot,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ScalarConstDivPow2Prover<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let uni_poly_evals: [F; 2] = (0..self.remainder_onehot.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals =
                    self.remainder_onehot
                        .sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                let id_evals =
                    self.identity
                        .sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                [ra_evals[0] * id_evals[0], ra_evals[1] * id_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        UniPoly::from_evals_and_hint(previous_claim, &uni_poly_evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.remainder_onehot
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.identity.bind_parallel(r_j, BindingOrder::LowToHigh);
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
        let r = [
            opening_point.r.as_slice(),
            self.params.r_node_output.r.as_slice(),
        ]
        .concat();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::ScalarConstDivPow2Ra(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            r.into(),
            self.remainder_onehot.final_sumcheck_claim(),
        );
    }
}

/// Verifier state for the `ScalarConstDivPow2` execution sumcheck.
///
/// The verifier samples the output point and pre-registers the virtual openings
/// that the execution relation will reference.
pub struct ScalarConstDivPow2Verifier<F: JoltField> {
    params: ScalarConstDivPow2Params<F>,
}

impl<F: JoltField> ScalarConstDivPow2Verifier<F> {
    /// Initializes the verifier and records the virtual openings used by the execution relation.
    pub fn new<T: Transcript>(
        computation_node: ComputationNode,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        let params = ScalarConstDivPow2Params::new(computation_node, transcript);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::ScalarConstDivPow2Divisor(params.computation_node.idx),
            SumcheckId::NodeExecution(params.computation_node.idx),
            params.r_node_output.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(params.computation_node.inputs[0]),
            SumcheckId::NodeExecution(params.computation_node.idx),
            params.r_node_output.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(params.computation_node.idx),
            SumcheckId::NodeExecution(params.computation_node.idx),
            params.r_node_output.clone(),
        );
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ScalarConstDivPow2Verifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let ra_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::ScalarConstDivPow2Ra(self.params.computation_node.idx),
                SumcheckId::NodeExecution(self.params.computation_node.idx),
            )
            .1;
        let int_eval =
            IdentityPolynomial::new(self.params.log_table_size).evaluate(&opening_point.r);
        ra_claim * int_eval
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
        let r = [
            opening_point.r.as_slice(),
            self.params.r_node_output.r.as_slice(),
        ]
        .concat();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::ScalarConstDivPow2Ra(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            r.into(),
        );
    }
}

/// One-hot encoding configuration for `ScalarConstDivPow2` remainder lookups.
pub struct ScalarConstDivPow2RaEncoding {
    /// Node index whose remainder one-hot columns are being committed.
    pub node_idx: usize,
    /// Log-size of the power-of-two lookup domain.
    pub log_table_size: usize,
}

impl ScalarConstDivPow2RaEncoding {
    /// Builds the one-hot encoding metadata for a single `ScalarConstDivPow2` node.
    pub fn new(node: &ComputationNode) -> Self {
        let Operator::ScalarConstDivPow2(op) = &node.operator else {
            panic!("Expected ScalarConstDivPow2 operator")
        };
        assert!(
            op.divisor > 0 && (op.divisor as u32).is_power_of_two(),
            "ScalarConstDivPow2 one-hot encoding requires a positive power-of-two divisor, got {0}",
            op.divisor
        );
        Self {
            node_idx: node.idx,
            log_table_size: (op.divisor as u32).trailing_zeros() as usize,
        }
    }
}

impl RaOneHotEncoding for ScalarConstDivPow2RaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPolynomial {
        CommittedPolynomial::ScalarConstDivPow2RaD(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> (VirtualPolynomial, SumcheckId) {
        (
            VirtualPolynomial::ScalarConstDivPow2Divisor(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn ra_source(&self) -> (VirtualPolynomial, SumcheckId) {
        (
            VirtualPolynomial::ScalarConstDivPow2Ra(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn log_k(&self) -> usize {
        self.log_table_size
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), self.log_table_size)
    }
}

fn scalar_const_div_pow2_lookup_indices(trace: &Trace, node: &ComputationNode) -> Vec<usize> {
    let LayerData { operands, .. } = Trace::layer_data(trace, node);
    let [input] = operands[..] else {
        panic!("Expected one operand for ScalarConstDivPow2 operation")
    };
    let Operator::ScalarConstDivPow2(op) = &node.operator else {
        panic!("Expected ScalarConstDivPow2 operator")
    };
    input
        .par_iter()
        .map(|&x| adjusted_remainder(x, op.divisor) as usize)
        .collect()
}

fn scalar_const_div_pow2_remainder_tensor(input: &Tensor<i32>, divisor: i32) -> Tensor<i32> {
    let remainder_data: Vec<i32> = input
        .iter()
        .map(|&a| adjusted_remainder(a, divisor))
        .collect();
    Tensor::<i32>::construct(remainder_data, input.dims().to_vec())
}
