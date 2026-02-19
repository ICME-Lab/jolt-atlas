use crate::onnx_proof::{ops::OperatorProofTrait, ProofId, ProofType, Prover, Verifier};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Square,
};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
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
    utils::{errors::ProofVerifyError, math::Math},
};

use crate::impl_standard_sumcheck_proof_api;

impl_standard_sumcheck_proof_api!(Square, SquareParams, SquareProver, SquareVerifier);

const DEGREE_BOUND: usize = 3;

/// Parameters for proving element-wise square (x²) operations.
///
/// Stores the opening point and computation node information needed for the sumcheck protocol.
#[derive(Clone)]
pub struct SquareParams<F: JoltField> {
    r_node_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
}

impl<F: JoltField> SquareParams<F> {
    /// Create new square parameters from a computation node and opening accumulator.
    pub fn new(computation_node: ComputationNode, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let r_node_output = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(computation_node.idx),
                SumcheckId::Execution,
            )
            .0
            .r;
        Self {
            r_node_output,
            computation_node,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for SquareParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, square_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(self.computation_node.idx),
            SumcheckId::Execution,
        );
        square_claim
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.computation_node.num_output_elements().log_2()
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
            GruenSplitEqPolynomial::new(&params.r_node_output, BindingOrder::LowToHigh);
        let LayerData {
            operands,
            output: _,
        } = Trace::layer_data(trace, &params.computation_node);
        let [operand] = operands[..] else {
            panic!("Expected one operand for Square operation")
        };
        let operand = MultilinearPolynomial::from(operand.clone());
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
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            opening_point.clone(),
            self.operand.final_sumcheck_claim(),
        );
        accumulator.cache_virtual_operand_claims(transcript, &self.params.computation_node);
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
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .0
            .r;
        let r_node_output_prime = self.params.normalize_opening_point(sumcheck_challenges).r;
        let eq_eval = EqPolynomial::mle(&r_node_output, &r_node_output_prime);
        let [operand_claim] = accumulator.get_operand_claims::<1>(self.params.computation_node.idx);
        eq_eval * operand_claim * operand_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            opening_point.clone(),
        );
        accumulator.append_operand_claims(transcript, self.params.computation_node.idx);
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
}
