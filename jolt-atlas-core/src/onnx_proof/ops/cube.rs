use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    impl_fused_rescale_proof_api,
    onnx_proof::{fused_rebase, ops::OperatorProofTrait, ProofId, ProofType, Prover, Verifier},
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_trace::{op::Cube, trace::LayerData, Trace};
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
    },
    subprotocols::{
        mles_product_sum::compute_mle_product_sum,
        sumcheck::SumcheckInstanceProof,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
        upstream_sumcheck::round_check_failed,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

impl_fused_rescale_proof_api!(Cube, CubeParams, CubeProver, CubeVerifier);

/// Shared parameter block for the element-wise cube sumcheck proof.
#[derive(Clone)]
pub struct CubeParams<F: JoltField> {
    pub(crate) r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    pub(crate) computation_node: ComputationNode,
}

impl<F: JoltField> CubeParams<F> {
    /// Creates new params by reading the current output opening from the accumulator.
    pub fn new(computation_node: ComputationNode, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);
        let r_node_output = accessor.get_reduced_opening().0;
        Self {
            r_node_output,
            computation_node,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for CubeParams<F> {
    fn degree(&self) -> usize {
        4
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        // Fused rescaling seam: `acc(r) = rescaled·2^(2S) + R` (or the plain
        // output opening when not fused). See [`fused_rebase`] .
        fused_rebase::fused_input_claim(accumulator, &self.computation_node)
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        use joltworks::utils::math::Math;
        self.computation_node
            .pow2_padded_num_output_elements()
            .log_2()
    }

    // output = eq_eval * operand * operand * operand
}

const DEGREE_BOUND: usize = 4;

/// Prover state for element-wise cube sumcheck protocol.
///
/// Maintains the equality polynomial and operand polynomial needed to generate
/// sumcheck messages proving that output[i] = operand[i]³ for all i.
pub struct CubeProver<F: JoltField> {
    params: CubeParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    operand: MultilinearPolynomial<F>,
}

impl<F: JoltField> CubeProver<F> {
    /// Initialize the prover with trace data and parameters.
    #[tracing::instrument(skip_all, name = "CubeProver::initialize")]
    pub fn initialize(trace: &Trace, params: CubeParams<F>) -> Self {
        let eq_r_node_output = GruenSplitEqPolynomial::new(
            &params.r_node_output.r,
            jolt_poly::BindingOrder::LowToHigh,
        );
        let LayerData {
            operands,
            output: _,
        } = TraceView::layer_data(trace, &params.computation_node);
        let [operand] = operands[..] else {
            panic!("Expected one operand for Cube operation")
        };
        let operand = MultilinearPolynomial::from(operand.padded_pow2());
        Self {
            params,
            eq_r_node_output,
            operand,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for CubeProver<F> {
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
        provider.append_nodeio(Target::Input(0), self.operand.final_claim());
    }
}

impl<F: JoltField> CubeProver<F> {
    fn bind(&mut self, r: F) {
        let r_j = F::challenge_from_field(r);
        self.eq_r_node_output.bind(r);
        self.operand.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

impl<F: JoltField> ProveRounds<F> for CubeProver<F> {
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
        compute_mle_product_sum(
            DEGREE_BOUND - 1,
            &self.operand,
            previous_claim,
            &self.eq_r_node_output,
        )
        .map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

/// Verifier for element-wise cube sumcheck protocol.
///
/// Verifies that the prover's sumcheck messages are consistent with the claimed
/// cube operation output.
pub struct CubeVerifier<F: JoltField> {
    params: CubeParams<F>,
}

impl<F: JoltField> CubeVerifier<F> {
    /// Create a new verifier for the cube operation.
    #[tracing::instrument(skip_all, name = "CubeVerifier::new")]
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = CubeParams::new(computation_node, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for CubeVerifier<F> {
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

        let operand_claim = accessor.get_nodeio(Target::Input(0)).1;

        eq_eval * operand_claim * operand_claim * operand_claim
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
    }
}
