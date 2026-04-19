use crate::{
    impl_standard_sumcheck_proof_api,
    onnx_proof::{ops::OperatorProofTrait, ProofId, ProofType, Prover, Verifier},
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Square,
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

impl_standard_sumcheck_proof_api!(Square, SquareParams, SquareProver, SquareVerifier);

/// Shared parameter block for the element-wise square sumcheck proof.
#[derive(Clone)]
pub struct SquareParams<F: JoltField> {
    pub(crate) r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    pub(crate) computation_node: ComputationNode,
}

impl<F: JoltField> SquareParams<F> {
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

    /// OpeningId for the operand (input[0]) at this node's execution sumcheck.
    #[cfg(feature = "zk")]
    fn operand_opening_id(&self) -> joltworks::poly::opening_proof::OpeningId {
        joltworks::poly::opening_proof::OpeningId::Virtual(
            VirtualPolynomial::NodeOutput(self.computation_node.inputs[0]),
            SumcheckId::NodeExecution(self.computation_node.idx),
        )
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for SquareParams<F> {
    fn degree(&self) -> usize {
        3
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

    // The input claim is the node's own output evaluation from eval reduction.
    // This value is baked as a constant in BlindFold's R1CS (not a variable),
    // so no constraint is needed.
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

    // output = eq_eval * operand^2
    // eq_eval is a deterministic function of the sumcheck challenges and r_node_output,
    // so it is passed as Challenge(0). The operand opening appears twice (squared).
    #[cfg(feature = "zk")]
    fn output_claim_constraint(
        &self,
    ) -> Option<joltworks::subprotocols::blindfold::OutputClaimConstraint> {
        use joltworks::subprotocols::blindfold::{OutputClaimConstraint, ProductTerm, ValueSource};

        let operand_id = self.operand_opening_id();
        let term = ProductTerm::scaled(
            ValueSource::Challenge(0),
            vec![
                ValueSource::Opening(operand_id),
                ValueSource::Opening(operand_id),
            ],
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

/// Prover state for element-wise square sumcheck protocol.
///
/// Maintains the equality polynomial and operand polynomial needed to generate
/// sumcheck messages proving that output[i] = operand[i]² for all i.
pub struct SquareProver<F: JoltField> {
    params: SquareParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    operand: MultilinearPolynomial<F>,
}

impl<F: JoltField> SquareProver<F> {
    /// Initialize the prover with trace data and parameters.
    #[tracing::instrument(skip_all, name = "SquareProver::initialize")]
    pub fn initialize(trace: &Trace, params: SquareParams<F>) -> Self {
        let eq_r_node_output =
            GruenSplitEqPolynomial::new(&params.r_node_output.r, BindingOrder::LowToHigh);
        let LayerData {
            operands,
            output: _,
        } = Trace::layer_data(trace, &params.computation_node);
        let [operand] = operands[..] else {
            panic!("Expected one operand for Square operation")
        };
        let operand = MultilinearPolynomial::from(operand.padded_next_power_of_two());
        Self {
            params,
            eq_r_node_output,
            operand,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for SquareProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            eq_r_node_output,
            operand,
            ..
        } = self;
        let [q_constant, q_quadratic] = eq_r_node_output.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let o0 = operand.get_bound_coeff(2 * g);
            let o_inf = operand.get_bound_coeff(2 * g + 1) - o0;

            let c0 = o0 * o0;
            let e = o_inf * o_inf;
            [c0, e]
        });
        eq_r_node_output.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
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
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            opening_point,
            self.operand.final_sumcheck_claim(),
        );
    }
}

/// Verifier for element-wise square sumcheck protocol.
///
/// Verifies that the prover's sumcheck messages are consistent with the claimed
/// square operation output.
pub struct SquareVerifier<F: JoltField> {
    params: SquareParams<F>,
}

impl<F: JoltField> SquareVerifier<F> {
    /// Create a new verifier for the square operation.
    #[tracing::instrument(skip_all, name = "SquareVerifier::new")]
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = SquareParams::new(computation_node, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for SquareVerifier<F> {
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
        let operand_claim = accumulator.get_node_output_claim(
            self.params.computation_node.inputs[0],
            self.params.computation_node.idx,
        );
        eq_eval * operand_claim * operand_claim
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
            opening_point,
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{model::test::ModelBuilder, model::Model, tensor::Tensor};
    use rand::{rngs::StdRng, SeedableRng};

    fn square_model(T: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![T]);
        let res = b.square(i);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_square() {
        let T = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::<i32>::random_small(&mut rng, &[T]);
        let model = square_model(T);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_square_non_power_of_two_input_len() {
        let t = 1000;
        let mut rng = StdRng::seed_from_u64(0x889);
        let input = Tensor::<i32>::random_small(&mut rng, &[t]);
        let model = square_model(t);
        unit_test_op(model, &[input]);
    }

    /// Validates that Square's BlindFold constraint formulas produce the same
    /// values as the actual `input_claim` / `expected_output_claim` methods.
    ///
    /// This is the critical synchronization invariant: if these diverge,
    /// BlindFold R1CS will be unsatisfied and ZK proofs will fail.
    #[cfg(feature = "zk")]
    #[test]
    fn test_square_blindfold_constraint_consistency() {
        use super::*;
        use ark_bn254::Fr;
        use atlas_onnx_tracer::ops::Operator;
        use joltworks::{
            field::challenge::mont_ark_u128::MontU128Challenge,
            poly::opening_proof::{OpeningId, VerifierOpeningAccumulator},
            subprotocols::{
                evaluation_reduction::ReducedInstance, sumcheck_verifier::SumcheckInstanceParams,
            },
        };

        type F = Fr;
        type Challenge = MontU128Challenge<F>;

        // Simulate a computation node: node 1 (Square) takes input from node 0
        let node = ComputationNode::new(1, Operator::Square(Square), vec![0], vec![4]);

        // Set up the verifier accumulator with known opening values
        let mut accumulator = VerifierOpeningAccumulator::<F>::new();

        // Register the node output (reduced evaluation for node 1)
        let r_output: Vec<F> = vec![F::from(7u64), F::from(11u64)];
        let node_output_claim = F::from(42u64);
        accumulator.reduced_evaluations.insert(
            1,
            ReducedInstance {
                r: r_output.clone(),
                claim: node_output_claim,
            },
        );

        // Register the operand opening (node 0's output at node 1's execution sumcheck)
        let operand_opening_id = OpeningId::Virtual(
            VirtualPolynomial::NodeOutput(0),
            SumcheckId::NodeExecution(1),
        );
        let operand_claim = F::from(13u64);
        accumulator
            .openings
            .insert(operand_opening_id, (r_output.clone().into(), operand_claim));

        // Create SquareParams
        let params = SquareParams::new(node, &accumulator);

        // --- Test input_claim_constraint ---
        let actual_input_claim = params.input_claim(&accumulator);
        let input_constraint = params.input_claim_constraint();
        assert!(
            input_constraint.terms.is_empty(),
            "Square's input_claim_constraint should be empty (baked constant)"
        );
        assert_eq!(actual_input_claim, node_output_claim);

        // --- Test output_claim_constraint ---
        let sumcheck_challenges: Vec<Challenge> =
            vec![Challenge::from(3u128), Challenge::from(5u128)];

        let output_constraint = params
            .output_claim_constraint()
            .expect("Square should have an output constraint");

        let challenge_values = params.output_constraint_challenge_values(&sumcheck_challenges);

        // Compute the actual expected output claim the same way the verifier does:
        // eq_eval * operand_claim * operand_claim
        let sumcheck_as_field: Vec<F> = sumcheck_challenges.iter().map(|c| (*c).into()).collect();
        let r_node_output_prime = params.normalize_opening_point(&sumcheck_as_field).r;
        let eq_eval = EqPolynomial::mle(&r_output, &r_node_output_prime);
        let actual_output_claim = eq_eval * operand_claim * operand_claim;

        // Evaluate the constraint formula at the same opening and challenge values
        let opening_values: Vec<F> = output_constraint
            .required_openings
            .iter()
            .map(|id| {
                assert_eq!(*id, operand_opening_id, "Unexpected opening ID");
                operand_claim
            })
            .collect();

        let constraint_output = output_constraint.evaluate(&opening_values, &challenge_values);

        assert_eq!(
            actual_output_claim, constraint_output,
            "output_claim_constraint must evaluate to the same value as expected_output_claim"
        );
    }
}
