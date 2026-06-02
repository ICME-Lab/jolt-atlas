//! Implements the eval-shift sumcheck used by tanh, erf, and sigmoid output
//! polynomials to map their opening point (tied to their claim) to the shared
//! opening point produced by the neural-teleport division sumcheck.
use crate::utils::opening_access::AccOpeningAccessor;
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
};
use common::VirtualPoly;
#[cfg(feature = "zk")]
use joltworks::subprotocols::blindfold::{
    InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource,
};
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
};

/// Shared parameter block for the eval-shift sumcheck proof.
#[derive(Clone)]
pub struct EvalShiftParams<F: JoltField> {
    pub(crate) r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    pub(crate) computation_node: ComputationNode,
}

impl<F: JoltField> EvalShiftParams<F> {
    /// Creates parameters from the current reduced-output opening in the accumulator.
    pub fn new(computation_node: ComputationNode, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);
        let r_node_output = accessor.get_reduced_opening().0;
        Self {
            r_node_output,
            computation_node,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for EvalShiftParams<F> {
    fn degree(&self) -> usize {
        2
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        accessor.get_reduced_opening().1
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

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::default()
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(
        &self,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        Vec::new()
    }

    // output = eq_eval * output
    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        let output_id = OpeningId::new(
            VirtualPoly::NTEvalShiftOutput(self.computation_node.idx),
            SumcheckId::NTEvalShift,
        );
        let term = ProductTerm::scaled(
            ValueSource::Challenge(0),
            vec![ValueSource::Opening(output_id)],
        );
        Some(OutputClaimConstraint::sum_of_products(vec![term]))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        let r_node_output_prime: Vec<F> = self
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let eq_eval = EqPolynomial::mle(&self.r_node_output.r, &r_node_output_prime);
        vec![eq_eval]
    }
}

/// Prover state for the eval-shift sumcheck protocol.
///
/// Maintains the equality polynomial and output polynomial needed to generate
/// sumcheck messages for the reduced-output claim.
pub struct EvalShiftProver<F: JoltField> {
    params: EvalShiftParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    output: MultilinearPolynomial<F>,
}

impl<F: JoltField> EvalShiftProver<F> {
    /// Initializes the prover with trace data and reduction parameters.
    #[tracing::instrument(skip_all, name = "EvalShiftProver::initialize")]
    pub fn initialize(trace: &Trace, params: EvalShiftParams<F>) -> Self {
        let eq_r_node_output =
            GruenSplitEqPolynomial::new(&params.r_node_output.r, BindingOrder::LowToHigh);
        let LayerData {
            operands: _,
            output,
        } = Trace::layer_data(trace, &params.computation_node);
        let output = MultilinearPolynomial::from(output.padded_next_power_of_two());
        Self {
            params,
            eq_r_node_output,
            output,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for EvalShiftProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            eq_r_node_output,
            output,
            ..
        } = self;
        let [q_constant] = eq_r_node_output.par_fold_out_in_unreduced::<9, 1>(&|g| {
            let o0 = output.get_bound_coeff(2 * g);
            [o0]
        });
        eq_r_node_output.gruen_poly_deg_2(q_constant, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_node_output.bind(r_j);
        self.output.bind_parallel(r_j, BindingOrder::LowToHigh);
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
        let opening_id = OpeningId::new(
            VirtualPoly::NTEvalShiftOutput(self.params.computation_node.idx),
            SumcheckId::NTEvalShift,
        );
        provider.append_custom(opening_id, self.output.final_claim());
    }
}

/// Verifier for the eval-shift sumcheck protocol.
///
/// Verifies that the prover's sumcheck messages are consistent with the
/// reduced-output claim.
pub struct EvalShiftVerifier<F: JoltField> {
    params: EvalShiftParams<F>,
}

impl<F: JoltField> EvalShiftVerifier<F> {
    /// Creates a new verifier for the reduction operation.
    #[tracing::instrument(skip_all, name = "EvalShiftVerifier::new")]
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = EvalShiftParams::new(computation_node, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for EvalShiftVerifier<F> {
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
        let opening_id = OpeningId::new(
            VirtualPoly::NTEvalShiftOutput(self.params.computation_node.idx),
            SumcheckId::NTEvalShift,
        );
        let output_claim = accessor.get_custom(opening_id).1;
        eq_eval * output_claim
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
        let opening_id = OpeningId::new(
            VirtualPoly::NTEvalShiftOutput(self.params.computation_node.idx),
            SumcheckId::NTEvalShift,
        );
        provider.append_custom(opening_id);
    }
}
