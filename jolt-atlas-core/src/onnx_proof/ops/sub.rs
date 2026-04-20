use crate::{
    impl_standard_sumcheck_proof_api,
    onnx_proof::{ops::OperatorProofTrait, ProofId, ProofType, Prover, Verifier},
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Sub,
};
use common::VirtualPolynomial;
use joltworks::{
    field::{IntoOpening, JoltField},
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
        sumcheck::SumcheckInstanceProof,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

impl_standard_sumcheck_proof_api!(Sub, SubParams, SubProver, SubVerifier);

/// Shared parameter block for the element-wise subtraction sumcheck proof.
#[derive(Clone)]
pub struct SubParams<F: JoltField> {
    pub(crate) r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    pub(crate) computation_node: ComputationNode,
}

impl<F: JoltField> SubParams<F> {
    /// Creates new params by reading the current output opening from the accumulator.
    pub fn new(computation_node: ComputationNode, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let r_node_output = accumulator
            .get_node_output_opening(computation_node.idx)
            .0
            .r;
        Self {
            r_node_output: r_node_output.into(),
            computation_node,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for SubParams<F> {
    fn degree(&self) -> usize {
        2
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        accumulator
            .get_node_output_opening(self.computation_node.idx)
            .1
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
    fn input_claim_constraint(&self) -> joltworks::subprotocols::blindfold::InputClaimConstraint {
        joltworks::subprotocols::blindfold::InputClaimConstraint::default()
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(
        &self,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        Vec::new()
    }

    // output = eq_eval * (left - right) = eq_eval * left + (-eq_eval) * right
    #[cfg(feature = "zk")]
    fn output_claim_constraint(
        &self,
    ) -> Option<joltworks::subprotocols::blindfold::OutputClaimConstraint> {
        use joltworks::subprotocols::blindfold::{OutputClaimConstraint, ProductTerm, ValueSource};
        let left_id = joltworks::poly::opening_proof::OpeningId::Virtual(
            VirtualPolynomial::NodeOutput(self.computation_node.inputs[0]),
            SumcheckId::NodeExecution(self.computation_node.idx),
        );
        let right_id = joltworks::poly::opening_proof::OpeningId::Virtual(
            VirtualPolynomial::NodeOutput(self.computation_node.inputs[1]),
            SumcheckId::NodeExecution(self.computation_node.idx),
        );
        Some(OutputClaimConstraint::sum_of_products(vec![
            ProductTerm::scaled(
                ValueSource::Challenge(0),
                vec![ValueSource::Opening(left_id)],
            ),
            ProductTerm::scaled(
                ValueSource::Challenge(1),
                vec![ValueSource::Opening(right_id)],
            ),
        ]))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        let r_node_output_prime: Vec<F> = self
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let eq_eval = EqPolynomial::mle(&self.r_node_output.r, &r_node_output_prime);
        vec![eq_eval, -eq_eval]
    }
}

/// Prover state for element-wise subtraction sumcheck protocol.
///
/// Maintains the equality polynomial and operand polynomials needed to generate
/// sumcheck messages proving that output[i] = left[i] - right[i] for all i.
pub struct SubProver<F: JoltField> {
    params: SubParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    left_operand: MultilinearPolynomial<F>,
    right_operand: MultilinearPolynomial<F>,
}

impl<F: JoltField> SubProver<F> {
    /// Initialize the prover with trace data and parameters.
    #[tracing::instrument(skip_all, name = "SubProver::initialize")]
    pub fn initialize(trace: &Trace, params: SubParams<F>) -> Self {
        let eq_r_node_output =
            GruenSplitEqPolynomial::new(&params.r_node_output.r, BindingOrder::LowToHigh);
        let LayerData {
            operands,
            output: _,
        } = Trace::layer_data(trace, &params.computation_node);
        let [left_operand, right_operand] = operands[..] else {
            panic!("Expected two operands for Sub operation")
        };
        let left_operand = MultilinearPolynomial::from(left_operand.padded_next_power_of_two());
        let right_operand = MultilinearPolynomial::from(right_operand.padded_next_power_of_two());
        Self {
            params,
            eq_r_node_output,
            left_operand,
            right_operand,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for SubProver<F> {
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
        let [q_constant] = eq_r_node_output.par_fold_out_in_unreduced::<9, 1>(&|g| {
            let lo0 = left_operand.get_bound_coeff(2 * g);
            let ro0 = right_operand.get_bound_coeff(2 * g);
            [lo0 - ro0]
        });
        eq_r_node_output.gruen_poly_deg_2(q_constant, previous_claim)
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
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            opening_point.clone(),
            self.left_operand.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            opening_point,
            self.right_operand.final_sumcheck_claim(),
        );
    }
}

/// Verifier for element-wise subtraction sumcheck protocol.
///
/// Verifies that the prover's sumcheck messages are consistent with the claimed
/// subtraction operation output.
pub struct SubVerifier<F: JoltField> {
    params: SubParams<F>,
}

impl<F: JoltField> SubVerifier<F> {
    /// Create a new verifier for the subtraction operation.
    #[tracing::instrument(skip_all, name = "SubVerifier::new")]
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = SubParams::new(computation_node, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for SubVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r_node_output = accumulator
            .get_node_output_opening(self.params.computation_node.idx)
            .0
            .r;
        let r_node_output_prime = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let eq_eval = EqPolynomial::mle(&r_node_output, &r_node_output_prime);
        let left_operand_claim = accumulator.get_node_output_claim(
            self.params.computation_node.inputs[0],
            self.params.computation_node.idx,
        );
        let right_operand_claim = accumulator.get_node_output_claim(
            self.params.computation_node.inputs[1],
            self.params.computation_node.idx,
        );
        eq_eval * (left_operand_claim - right_operand_claim)
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
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            opening_point,
        );
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

    fn sub_model(rng: &mut StdRng, T: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![T]);
        let c = b.constant(Tensor::random_small(rng, &[T]));
        let res = b.sub(i, c);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_sub() {
        let T = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::<i32>::random_small(&mut rng, &[T]);
        let model = sub_model(&mut rng, T);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_sub_non_power_of_two_input_len() {
        let t = 1000;
        let mut rng = StdRng::seed_from_u64(0x889);
        let input = Tensor::<i32>::random_small(&mut rng, &[t]);
        let model = sub_model(&mut rng, t);
        unit_test_op(model, &[input]);
    }
}
