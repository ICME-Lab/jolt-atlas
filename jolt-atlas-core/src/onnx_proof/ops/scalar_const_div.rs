use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    onnx_proof::{ops::OperatorProofTrait, ProofId, ProofType, Prover, Verifier},
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_trace::{op::ScalarConstDiv, trace::LayerData, Op, Tensor, Trace};
use common::CommittedPoly;
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN, LITTLE_ENDIAN,
        },
        split_eq_poly::SplitEqUnreducedFold,
    },
    subprotocols::{
        sumcheck::{Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
        upstream_sumcheck::round_check_failed,
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
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
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

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        let polys = vec![CommittedPoly::ScalarConstDivNodeRemainder(node.idx)];
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
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
    scalar_const_divisor: i32,
}

impl<F: JoltField> ScalarConstDivParams<F> {
    /// Create new scalar constant division parameters from a computation node and opening accumulator.
    pub fn new(computation_node: ComputationNode, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);
        let r_node_output = accessor.get_reduced_opening().0;
        let Op::ScalarConstDiv(scalar_const_div) = &computation_node.operator else {
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
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        let q_claim = accessor.get_reduced_opening().1;
        q_claim * F::from_i32(self.scalar_const_divisor)
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.computation_node
            .pow2_padded_num_output_elements()
            .log_2()
    }

    // output = eq_eval * (left_operand - R_claim)
    //        = eq_eval * left + (-eq_eval) * R
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
        let eq_r_node_output = GruenSplitEqPolynomial::new(
            &params.r_node_output.r,
            jolt_poly::BindingOrder::LowToHigh,
        );
        let LayerData { operands, .. } = TraceView::layer_data(trace, &params.computation_node);
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

    fn rounds(&mut self) -> Option<&mut dyn ProveRounds<F>> {
        Some(self)
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
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .into_provider(transcript, opening_point);

        provider.append_nodeio(Target::Input(0), self.left_operand.final_claim());
        provider.append_advice(
            CommittedPoly::ScalarConstDivNodeRemainder,
            self.R.final_claim(),
        );
    }
}

impl<F: JoltField> ScalarConstDivProver<F> {
    fn bind(&mut self, r: F) {
        let r_j = F::challenge_from_field(r);
        self.eq_r_node_output.bind(r);
        self.left_operand
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.R.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

impl<F: JoltField> ProveRounds<F> for ScalarConstDivProver<F> {
    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        let Self {
            eq_r_node_output,
            left_operand,
            R,
            ..
        } = self;
        eq_r_node_output
            .gruen_round_deg_2(
                &|g| left_operand.get_bound_coeff(2 * g + 1) - R.get_bound_coeff(2 * g + 1),
                &|g| left_operand.get_bound_coeff(2 * g) - R.get_bound_coeff(2 * g),
                previous_claim,
            )
            .map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
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
        let accessor = AccOpeningAccessor::new(accumulator, &self.params.computation_node);

        let r_node_output = &self.params.r_node_output.r;
        let r_node_output_prime = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let eq_eval = EqPolynomial::mle(r_node_output, &r_node_output_prime);

        let R_claim = accessor
            .get_advice(CommittedPoly::ScalarConstDivNodeRemainder)
            .1;
        let left_operand_claim = accessor.get_nodeio(Target::Input(0)).1;

        eq_eval * (left_operand_claim - R_claim)
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
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .into_provider(transcript, opening_point);
        provider.append_nodeio(Target::Input(0));
        provider.append_advice(CommittedPoly::ScalarConstDivNodeRemainder);
    }
}
