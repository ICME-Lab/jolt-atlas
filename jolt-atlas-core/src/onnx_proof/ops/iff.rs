use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    onnx_proof::{ops::OperatorProofTrait, ProofId, ProofType, Prover, Verifier},
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_trace::{op::Iff, trace::LayerData, Trace};
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
        sumcheck::SumcheckInstanceProof,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
        upstream_sumcheck::round_check_failed,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

use crate::impl_standard_sumcheck_proof_api;

impl_standard_sumcheck_proof_api!(Iff, IffParams, IffProver, IffVerifier);

const DEGREE_BOUND: usize = 3;

/// Parameters for proving conditional selection (if-then-else) operations.
///
/// The Iff operation computes: output = mask ? a : b (i.e., if mask then a else b).
/// Stores the opening point and computation node information needed for the sumcheck protocol.
#[derive(Clone)]
pub struct IffParams<F: JoltField> {
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
}

impl<F: JoltField> IffParams<F> {
    /// Create new conditional selection parameters from a computation node and opening accumulator.
    pub fn new(computation_node: ComputationNode, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);
        let r_node_output = accessor.get_reduced_opening().0;
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
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        accessor.get_reduced_opening().1
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.computation_node
            .pow2_padded_num_output_elements()
            .log_2()
    }

    // output = eq_eval * (mask * a + (1 - mask) * b)
    //        = eq_eval * mask * a + eq_eval * b - eq_eval * mask * b
}

/// Prover state for conditional selection (if-then-else) sumcheck protocol.
///
/// Maintains the equality polynomial, mask, and two operand polynomials needed to generate
/// sumcheck messages proving that output[i] = mask[i] ? a[i] : b[i] for all i.
pub struct IffProver<F: JoltField> {
    params: IffParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    mask_operand: MultilinearPolynomial<F>,
    a_operand: MultilinearPolynomial<F>,
    b_operand: MultilinearPolynomial<F>,
}

impl<F: JoltField> IffProver<F> {
    /// Initialize the prover with trace data and parameters.
    #[tracing::instrument(skip_all, name = "IffProver::initialize")]
    pub fn initialize(trace: &Trace, params: IffParams<F>) -> Self {
        let eq_r_node_output = GruenSplitEqPolynomial::new(
            &params.r_node_output.r,
            jolt_poly::BindingOrder::LowToHigh,
        );
        let LayerData {
            operands,
            output: _output,
        } = TraceView::layer_data(trace, &params.computation_node);
        let [mask_operand, a_operand, b_operand] = operands[..] else {
            panic!("Expected three operands for Iff operation")
        };
        let mask_operand = MultilinearPolynomial::from(mask_operand.clone());
        let a_operand = MultilinearPolynomial::from(a_operand.clone());
        let b_operand = MultilinearPolynomial::from(b_operand.clone());

        #[cfg(test)]
        {
            use joltworks::poly::multilinear_polynomial::PolynomialEvaluation;
            let eq = EqPolynomial::evals(&params.r_node_output.r);
            let output = MultilinearPolynomial::from(_output.clone());
            let claim = (0..a_operand.len())
                .map(|i| {
                    let mask = mask_operand.get_bound_coeff(i);
                    let a = a_operand.get_bound_coeff(i);
                    let b = b_operand.get_bound_coeff(i);

                    eq[i] * (mask * a + (F::one() - mask) * b)
                })
                .sum();

            crate::onnx_proof::testing::probe::probe_assert_eq!(
                output.evaluate(&params.r_node_output.r),
                claim
            )
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

        provider.append_nodeio(Target::Input(0), self.mask_operand.final_claim());
        provider.append_nodeio(Target::Input(1), self.a_operand.final_claim());
        provider.append_nodeio(Target::Input(2), self.b_operand.final_claim());
    }
}

impl<F: JoltField> IffProver<F> {
    fn bind(&mut self, r: F) {
        let r_j = F::challenge_from_field(r);
        self.eq_r_node_output.bind(r);
        self.mask_operand
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.a_operand.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.b_operand.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

impl<F: JoltField> ProveRounds<F> for IffProver<F> {
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
            mask_operand,
            a_operand,
            b_operand,
            ..
        } = self;
        eq_r_node_output
            .gruen_round_deg_3(
                &|g| {
                    let mask0 = mask_operand.get_bound_coeff(2 * g);
                    let mask1 = mask_operand.get_bound_coeff(2 * g + 1);
                    let mask_inf = mask1 - mask0;

                    let a0 = a_operand.get_bound_coeff(2 * g);
                    let a_inf = a_operand.get_bound_coeff(2 * g + 1) - a0;

                    let b0 = b_operand.get_bound_coeff(2 * g);
                    let b_inf = b_operand.get_bound_coeff(2 * g + 1) - b0;

                    let c0 = mask0 * a0 + (F::one() - mask0) * b0;
                    let f = (F::one() - mask1) - (F::one() - mask0);
                    [c0, mask_inf * a_inf + f * b_inf]
                },
                &|g| {
                    let mask1 = mask_operand.get_bound_coeff(2 * g + 1);
                    mask1 * a_operand.get_bound_coeff(2 * g + 1)
                        + (F::one() - mask1) * b_operand.get_bound_coeff(2 * g + 1)
                },
                previous_claim,
            )
            .map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

/// Verifier for conditional selection (if-then-else) sumcheck protocol.
///
/// Verifies that the prover's sumcheck messages are consistent with the claimed
/// conditional selection operation output.
pub struct IffVerifier<F: JoltField> {
    params: IffParams<F>,
}

impl<F: JoltField> IffVerifier<F> {
    /// Create a new verifier for the conditional selection operation.
    #[tracing::instrument(skip_all, name = "IffVerifier::new")]
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
        let accessor = AccOpeningAccessor::new(accumulator, &self.params.computation_node);
        let r_node_output = &self.params.r_node_output.r;
        let r_node_output_prime = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let eq_eval = EqPolynomial::mle(r_node_output, &r_node_output_prime);

        let mask_operand_claim = accessor.get_nodeio(Target::Input(0)).1;
        let a_operand_claim = accessor.get_nodeio(Target::Input(1)).1;
        let b_operand_claim = accessor.get_nodeio(Target::Input(2)).1;

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
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .into_provider(transcript, opening_point);

        provider.append_nodeio(Target::Input(0));
        provider.append_nodeio(Target::Input(1));
        provider.append_nodeio(Target::Input(2));
    }
}
