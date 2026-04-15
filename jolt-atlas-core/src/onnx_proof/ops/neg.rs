use crate::{
    impl_standard_params, impl_standard_sumcheck_proof_api,
    onnx_proof::{ops::OperatorProofTrait, ProofId, ProofType, Prover, Verifier},
    utils::opening_id_builder::{AccOpeningAccessor, Target},
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Neg,
};
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

impl_standard_sumcheck_proof_api!(Neg, NegParams, NegProver, NegVerifier);
impl_standard_params!(NegParams, 2);

/// Prover state for element-wise negation sumcheck protocol.
///
/// Maintains the equality polynomial and operand polynomial needed to generate
/// sumcheck messages proving that output[i] = -operand[i] for all i.
pub struct NegProver<F: JoltField> {
    params: NegParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    operand: MultilinearPolynomial<F>,
}

impl<F: JoltField> NegProver<F> {
    /// Initialize the prover with trace data and parameters.
    #[tracing::instrument(skip_all, name = "NegProver::initialize")]
    pub fn initialize(trace: &Trace, params: NegParams<F>) -> Self {
        let eq_r_node_output =
            GruenSplitEqPolynomial::new(&params.r_node_output.r, BindingOrder::LowToHigh);
        let LayerData {
            operands,
            output: _,
        } = Trace::layer_data(trace, &params.computation_node);
        let [operand] = operands[..] else {
            panic!("Expected one operand for Neg operation")
        };
        let operand = MultilinearPolynomial::from(operand.padded_next_power_of_two());
        Self {
            params,
            eq_r_node_output,
            operand,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for NegProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            eq_r_node_output,
            operand,
            ..
        } = self;
        let [q_constant] = eq_r_node_output.par_fold_out_in_unreduced::<9, 1>(&|g| {
            let o0 = operand.get_bound_coeff(2 * g);
            [-o0]
        });
        eq_r_node_output.gruen_poly_deg_2(q_constant, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_node_output.bind(r_j);
        self.operand.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            .to_provider(transcript, opening_point);
        provider.append_node_io(Target::Input(0), self.operand.final_claim());
    }
}

/// Verifier for element-wise negation sumcheck protocol.
///
/// Verifies that the prover's sumcheck messages are consistent with the claimed
/// negation operation output.
pub struct NegVerifier<F: JoltField> {
    params: NegParams<F>,
}

impl<F: JoltField> NegVerifier<F> {
    /// Create a new verifier for the negation operation.
    #[tracing::instrument(skip_all, name = "NegVerifier::new")]
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = NegParams::new(computation_node, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for NegVerifier<F> {
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

        let operand_claim = accessor.get_node_io(Target::Input(0)).1;
        eq_eval * (-operand_claim)
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
            .to_provider(transcript, opening_point);
        provider.append_node_io(Target::Input(0));
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, Model},
        tensor::Tensor,
    };
    use rand::{rngs::StdRng, SeedableRng};

    fn neg_model(T: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![T]);
        let res = b.neg(i);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_neg() {
        let T = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::<i32>::random_small(&mut rng, &[T]);
        let model = neg_model(T);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_neg_non_power_of_two_input_len() {
        let t = 1000;
        let mut rng = StdRng::seed_from_u64(0x889);
        let input = Tensor::<i32>::random_small(&mut rng, &[t]);
        let model = neg_model(t);
        unit_test_op(model, &[input]);
    }
}
