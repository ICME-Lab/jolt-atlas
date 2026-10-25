use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    impl_fused_rescale_proof_api,
    onnx_proof::{fused_rebase, ops::OperatorProofTrait, ProofId, ProofType, Prover, Verifier},
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_trace::{op::Mul, trace::LayerData, Trace};
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN, LITTLE_ENDIAN,
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
    utils::errors::ProofVerifyError,
};

impl_fused_rescale_proof_api!(Mul, MulParams, MulProver, MulVerifier);

/// Shared parameter block for the element-wise multiplication sumcheck proof.
#[derive(Clone)]
pub struct MulParams<F: JoltField> {
    pub(crate) r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    pub(crate) computation_node: ComputationNode,
}

impl<F: JoltField> MulParams<F> {
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

impl<F: JoltField> SumcheckInstanceParams<F> for MulParams<F> {
    fn degree(&self) -> usize {
        3
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        // Fused rescaling seam: `acc(r) = rescaled·2^S + R` (or the plain output
        // opening when not fused). See [`fused_rebase`] .
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

    // output = eq_eval * left * right
}

/// Prover state for element-wise multiplication sumcheck protocol.
///
/// Maintains the equality polynomial and operand polynomials needed to generate
/// sumcheck messages proving that output[i] = left[i] * right[i] for all i.
pub struct MulProver<F: JoltField> {
    params: MulParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    left_operand: MultilinearPolynomial<F>,
    right_operand: MultilinearPolynomial<F>,
}

impl<F: JoltField> MulProver<F> {
    /// Initialize the prover with trace data and parameters.
    #[tracing::instrument(skip_all, name = "MulProver::initialize")]
    pub fn initialize(trace: &Trace, params: MulParams<F>) -> Self {
        let eq_r_node_output =
            GruenSplitEqPolynomial::new(&params.r_node_output.r, BindingOrder::LowToHigh);
        let LayerData {
            operands,
            output: _,
        } = TraceView::layer_data(trace, &params.computation_node);
        let [left_operand, right_operand] = operands[..] else {
            panic!("Expected two operands for Mul operation")
        };
        let left_operand = MultilinearPolynomial::from(left_operand.padded_pow2());
        let right_operand = MultilinearPolynomial::from(right_operand.padded_pow2());
        Self {
            params,
            eq_r_node_output,
            left_operand,
            right_operand,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for MulProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            eq_r_node_output,
            left_operand,
            right_operand,
            ..
        } = self;
        let [q_constant, q_quadratic] = eq_r_node_output.par_fold_out_in_unreduced::<2>(&|g| {
            let lo0 = left_operand.get_bound_coeff(2 * g);
            let lo_inf = left_operand.get_bound_coeff(2 * g + 1) - lo0;
            let ro0 = right_operand.get_bound_coeff(2 * g);
            let ro_inf = right_operand.get_bound_coeff(2 * g + 1) - ro0;
            let c0 = lo0 * ro0;
            let e = lo_inf * ro_inf;
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
        provider.append_nodeio(Target::Input(1), self.right_operand.final_claim());
    }
}

/// Verifier for element-wise multiplication sumcheck protocol.
///
/// Verifies that the prover's sumcheck messages are consistent with the claimed
/// multiplication operation output.
pub struct MulVerifier<F: JoltField> {
    params: MulParams<F>,
}

impl<F: JoltField> MulVerifier<F> {
    /// Create a new verifier for the multiplication operation.
    #[tracing::instrument(skip_all, name = "MulVerifier::new")]
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = MulParams::new(computation_node, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for MulVerifier<F> {
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

        let left_operand_claim = accessor.get_nodeio(Target::Input(0)).1;
        let right_operand_claim = accessor.get_nodeio(Target::Input(1)).1;

        eq_eval * left_operand_claim * right_operand_claim
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
    }
}
