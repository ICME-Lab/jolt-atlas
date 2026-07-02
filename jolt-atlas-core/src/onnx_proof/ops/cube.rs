use crate::{
    impl_fused_rescale_proof_api,
    onnx_proof::{fused_rebase, ops::OperatorProofTrait, ProofId, ProofType, Prover, Verifier},
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Cube,
};
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
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN, LITTLE_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        mles_product_sum::compute_mle_product_sum,
        sumcheck::SumcheckInstanceProof,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
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

    // output = eq_eval * operand * operand * operand
    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        let op_id = crate::utils::opening_access::OpeningIdBuilder::new(&self.computation_node)
            .nodeio(Target::Input(0));
        Some(OutputClaimConstraint::sum_of_products(vec![
            ProductTerm::scaled(
                ValueSource::Challenge(0),
                vec![
                    ValueSource::Opening(op_id),
                    ValueSource::Opening(op_id),
                    ValueSource::Opening(op_id),
                ],
            ),
        ]))
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
        let eq_r_node_output =
            GruenSplitEqPolynomial::new(&params.r_node_output.r, BindingOrder::LowToHigh);
        let LayerData {
            operands,
            output: _,
        } = Trace::layer_data(trace, &params.computation_node);
        let [operand] = operands[..] else {
            panic!("Expected one operand for Cube operation")
        };
        let operand = MultilinearPolynomial::from(operand.padded_next_power_of_two());
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

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        compute_mle_product_sum(
            DEGREE_BOUND - 1,
            &self.operand,
            previous_claim,
            &self.eq_r_node_output,
        )
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
            .into_provider(transcript, opening_point);
        provider.append_nodeio(Target::Input(0), self.operand.final_claim());
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

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, Model},
        tensor::Tensor,
    };
    use rand::{rngs::StdRng, SeedableRng};

    fn cube_model(T: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![T]);
        let res = b.cube(i, 12); // TODO: Pass in scale from runtime args instead of hardcoding here.
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_cube() {
        let T = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::<i32>::random_small(&mut rng, &[T]);
        let model = cube_model(T);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_cube_non_power_of_two_input_len() {
        let t = 1000;
        let mut rng = StdRng::seed_from_u64(0x889);
        let input = Tensor::<i32>::random_small(&mut rng, &[t]);
        let model = cube_model(t);
        unit_test_op(model, &[input]);
    }

    /// Build a fused-`Cube` (scale 12 → rebase by `2^24`) model with all-`input_value`
    /// input so `x³ >> 24` saturates the i32 clamp . `x = 2^19+1` keeps `x³`
    /// within i64 while overflowing i32, with a non-zero remainder.
    fn saturating_cube_model(input_value: i32, t: usize) -> (Model, Tensor<i32>) {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![t]);
        let res = b.cube(i, 12);
        b.mark_output(res);
        let input = Tensor::new(Some(&vec![input_value; t]), &[t]).unwrap();
        (b.build(), input)
    }

    #[test]
    fn test_cube_saturating_clamp_positive() {
        let big = (1 << 19) + 1;
        let (model, input) = saturating_cube_model(big, 1 << 4);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_cube_saturating_clamp_negative() {
        // x³ < 0 for x < 0 → floor-rebase then clamp to i32::MIN, non-zero remainder.
        let big = (1 << 19) + 1;
        let (model, input) = saturating_cube_model(-big, 1 << 4);
        unit_test_op(model, &[input]);
    }
}
