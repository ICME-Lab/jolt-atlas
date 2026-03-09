//! Reshape proof (selector-based equality over padded layouts).
//!
//! Proof target:
//! `sum_i (A(i) * Sa(i) - B(i) * Sb(i)) = 0`
//! which is equivalent to:
//! `sum_i A(i) * Sa(i) = sum_i B(i) * Sb(i)`.
//!
//! Where:
//! - `A(i)`: input padded MLE values
//! - `B(i)`: output padded MLE values
//! - `Sa(i), Sb(i)`: selector MLEs that place the same coefficient `rho(t)=gamma^t`
//!   on the padded position corresponding to raw element `t`, and `0` on padding cells.
//!
//! This aligns input/output raw elements under different padded layouts and checks
//! reshape consistency via one sumcheck instance.

use crate::onnx_proof::{
    ops::{OperatorProofTrait, Prover, Verifier},
    ProofId, ProofType,
};
use atlas_onnx_tracer::{
    model::{trace::{LayerData, Trace}, ComputationGraph},
    node::ComputationNode,
    ops::Reshape,
};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        unipoly::UniPoly,
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
    },
    subprotocols::{
        sumcheck::Sumcheck,
        sumcheck::SumcheckInstanceProof,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Reshape {
    #[tracing::instrument(skip_all, name = "Reshape::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        // TODO(soundness): Audit challenge derivation/order to ensure this gamma is
        // transcript-bound to all required prior messages before selector construction.
        let gamma = prover.transcript.challenge_scalar_optimized::<F>();
        let params = ReshapeSumcheckParams::<F>::new(
            node.clone(),
            &prover.accumulator,
            &prover.preprocessing.model.graph,
            gamma.into(),
        );
        let mut prover_sumcheck = ReshapeSumcheckProver::initialize(&prover.trace, params);
        let (proof, _) = Sumcheck::prove(
            &mut prover_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        vec![(ProofId(node.idx, ProofType::Execution), proof)]
    }

    #[tracing::instrument(skip_all, name = "Reshape::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // TODO(soundness): Keep gamma derivation order exactly aligned with prover
        // and verify transcript binding assumptions for selector randomization.
        let gamma = verifier.transcript.challenge_scalar_optimized::<F>();
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let verifier_sumcheck = ReshapeSumcheckVerifier::new(
            node.clone(),
            &verifier.accumulator,
            &verifier.preprocessing.model.graph,
            gamma.into(),
        );
        Sumcheck::verify(
            proof,
            &verifier_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;
        Ok(())
    }
}

/// Build a reshape selector vector over the padded tensor domain.
///
/// For each raw (unpadded) linear position `t`, this places `rho(t)` at the
/// corresponding padded linear index that keeps the same raw coordinate.
/// All padding cells remain zero.
///
/// This lets us compare two differently padded layouts (e.g. input/output of
/// reshape) under the same extraction order by using the same `rho(t)` sequence.
pub(crate) fn build_reshape_selectors<F: JoltField>(
    dims: &[usize],
    rho: impl Fn(usize) -> F,
) -> Vec<F> {
    fn linear_to_coord(mut index: usize, dims: &[usize]) -> Vec<usize> {
        let mut coord = vec![0; dims.len()];
        for axis in (0..dims.len()).rev() {
            coord[axis] = index % dims[axis];
            index /= dims[axis];
        }
        coord
    }

    fn coord_to_linear(coord: &[usize], dims: &[usize]) -> usize {
        let mut index = 0usize;
        let mut stride = 1usize;
        for axis in (0..dims.len()).rev() {
            index += coord[axis] * stride;
            stride *= dims[axis];
        }
        index
    }

    let raw_len: usize = dims.iter().product();
    let padded_dims: Vec<usize> = dims.iter().map(|d| d.next_power_of_two()).collect();
    let padded_len: usize = padded_dims.iter().product();

    // Selector values are zero on padding cells by default.
    let mut selector = vec![F::zero(); padded_len];

    // For each raw linear position t, compute its padded linear position by:
    // 1) converting t to a raw coordinate
    // 2) re-encoding that coordinate under padded strides
    // Then store rho(t) at that padded position.
    for t in 0..raw_len {
        let raw_coord = linear_to_coord(t, dims);
        let padded_index = coord_to_linear(&raw_coord, &padded_dims);
        selector[padded_index] = rho(t);
    }

    selector
}

/// Parameters for the upcoming reshape sumcheck.
///
/// Intended claim shape:
/// sum_i (A(i) * Sa(i) - B(i) * Sb(i)) = 0
#[allow(dead_code)]
#[derive(Clone)]
pub struct ReshapeSumcheckParams<F: JoltField> {
    /// Reshape computation node being proven.
    pub computation_node: ComputationNode,
    /// Output opening point challenge vector for this node.
    pub r_output: Vec<F::Challenge>,
    /// Original (unpadded) input shape used to build selector `Sa`.
    pub input_raw_dims: Vec<usize>,
    /// Original (unpadded) output shape used to build selector `Sb`.
    pub output_raw_dims: Vec<usize>,
    /// Transcript-derived randomizer used in selector coefficients (`gamma^t`).
    pub gamma: F,
}

#[allow(dead_code)]
impl<F: JoltField> ReshapeSumcheckParams<F> {
    /// Build reshape sumcheck parameters from node/opening context and graph metadata.
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
        graph: &ComputationGraph,
        gamma: F,
    ) -> Self {
        let r_output = accumulator.get_node_output_opening(computation_node.idx).0.r;
        let input_raw_dims = graph
            .nodes
            .get(&computation_node.inputs[0])
            .expect("Reshape node should have one input")
            .output_dims
            .clone();
        let output_raw_dims = computation_node.output_dims.clone();
        Self {
            computation_node,
            r_output,
            input_raw_dims,
            output_raw_dims,
            gamma,
        }
    }
}

#[allow(dead_code)]
impl<F: JoltField> SumcheckInstanceParams<F> for ReshapeSumcheckParams<F> {
    fn degree(&self) -> usize {
        2
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn normalize_opening_point(
        &self,
        challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.computation_node.num_output_elements().log_2()
    }
}

/// Prover skeleton for reshape sumcheck.
#[allow(dead_code)]
pub struct ReshapeSumcheckProver<F: JoltField> {
    /// Static sumcheck parameters.
    pub params: ReshapeSumcheckParams<F>,
    /// Input polynomial (padded MLE).
    pub input_mle: MultilinearPolynomial<F>,
    /// Output polynomial (padded MLE).
    pub output_mle: MultilinearPolynomial<F>,
    /// Selector polynomial for input layout.
    pub selector_a_mle: MultilinearPolynomial<F>,
    /// Selector polynomial for output layout.
    pub selector_b_mle: MultilinearPolynomial<F>,
}

#[allow(dead_code)]
impl<F: JoltField> ReshapeSumcheckProver<F> {
    /// Initialize reshape prover state from trace tensors and prepared parameters.
    pub fn initialize(trace: &Trace, params: ReshapeSumcheckParams<F>) -> Self {
        let LayerData { operands, output } = Trace::layer_data(trace, &params.computation_node);
        let [input] = operands[..] else {
            panic!("Expected one operand for Reshape operation")
        };

        let input_mle = MultilinearPolynomial::from(input.padded_next_power_of_two());
        let output_mle = MultilinearPolynomial::from(output.padded_next_power_of_two());

        let selector_a = build_reshape_selectors(input.dims(), |t| gamma_power(params.gamma, t));
        let selector_b = build_reshape_selectors(output.dims(), |t| gamma_power(params.gamma, t));

        let selector_a_mle = MultilinearPolynomial::from(selector_a);
        let selector_b_mle = MultilinearPolynomial::from(selector_b);
        assert_eq!(input_mle.len(), output_mle.len());
        assert_eq!(input_mle.len(), selector_a_mle.len());
        assert_eq!(output_mle.len(), selector_b_mle.len());

        Self {
            params,
            input_mle,
            output_mle,
            selector_a_mle,
            selector_b_mle,
        }
    }
}

#[allow(dead_code)]
impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ReshapeSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        const DEGREE_BOUND: usize = 2;
        let Self {
            input_mle,
            output_mle,
            selector_a_mle,
            selector_b_mle,
            ..
        } = self;
        let half_poly_len = input_mle.len() / 2;
        let mut uni_poly_evals = [F::zero(); 2];
        for i in 0..half_poly_len {
            let a_evals = input_mle.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
            let b_evals = output_mle.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
            let sa_evals = selector_a_mle.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
            let sb_evals = selector_b_mle.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);

            uni_poly_evals[0] += a_evals[0] * sa_evals[0] - b_evals[0] * sb_evals[0];
            uni_poly_evals[1] += a_evals[1] * sa_evals[1] - b_evals[1] * sb_evals[1];
        }
        UniPoly::from_evals_and_hint(previous_claim, &uni_poly_evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.input_mle.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.output_mle.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.selector_a_mle.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.selector_b_mle.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            opening_point.clone(),
            self.input_mle.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            opening_point,
            self.output_mle.final_sumcheck_claim(),
        );
    }
}

/// Verifier skeleton for reshape sumcheck.
#[allow(dead_code)]
pub struct ReshapeSumcheckVerifier<F: JoltField> {
    /// Static sumcheck parameters.
    pub params: ReshapeSumcheckParams<F>,
}

#[allow(dead_code)]
impl<F: JoltField> ReshapeSumcheckVerifier<F> {
    /// Build the reshape sumcheck verifier from node/opening context and graph metadata.
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
        graph: &ComputationGraph,
        gamma: F,
    ) -> Self {
        let params = ReshapeSumcheckParams::new(computation_node, accumulator, graph, gamma);
        Self { params }
    }
}

#[allow(dead_code)]
impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ReshapeSumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r_node_output_prime = self.params.normalize_opening_point(sumcheck_challenges).r;

        let input_claim = accumulator.get_node_output_claim(
            self.params.computation_node.inputs[0],
            self.params.computation_node.idx,
        );
        let output_claim = accumulator.get_node_output_claim(
            self.params.computation_node.idx,
            self.params.computation_node.idx,
        );

        let selector_a =
            build_reshape_selectors(&self.params.input_raw_dims, |t| gamma_power(self.params.gamma, t));
        let selector_b =
            build_reshape_selectors(&self.params.output_raw_dims, |t| gamma_power(self.params.gamma, t));
        let selector_a_claim =
            MultilinearPolynomial::from(selector_a).evaluate(&r_node_output_prime);
        let selector_b_claim =
            MultilinearPolynomial::from(selector_b).evaluate(&r_node_output_prime);

        input_claim * selector_a_claim - output_claim * selector_b_claim
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
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            opening_point,
        );
    }
}

fn gamma_power<F: JoltField>(gamma: F, exponent: usize) -> F {
    // TODO: This is a naive O(exponent) implementation. Replace with fast exponentiation
    // (square-and-multiply) or incremental power caching when selector construction is optimized.
    let mut acc = F::one();
    for _ in 0..exponent {
        acc *= gamma;
    }
    acc
}
 
#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, Model},
        tensor::Tensor,
    };
    use rand::{rngs::StdRng, SeedableRng};

    fn reshape_model(input_shape: &[usize], output_shape: &[usize]) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(input_shape.to_vec());
        let res = b.reshape(i, output_shape.to_vec());
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_reshape() {
        let mut rng = StdRng::seed_from_u64(0x999);
        let test_cases = vec![
            // (input_shape, output_shape)
            (vec![16], vec![4, 4]),
            (vec![4, 8], vec![8, 4]),
            (vec![2, 4, 8], vec![64]),
            (vec![2, 4, 8], vec![8, 8]),
        ];

        for (input_shape, output_shape) in test_cases {
            let input = Tensor::<i32>::random_small(&mut rng, &input_shape);
            let model = reshape_model(&input_shape, &output_shape);
            unit_test_op(model, &[input]);
        }
    }

    #[test]
    fn test_reshape_non_power_of_two_input_len() {
        let mut rng = StdRng::seed_from_u64(0x99A);
        let input_shape = vec![10, 10];
        let output_shape = vec![20, 5];
        let input = Tensor::<i32>::random_small(&mut rng, &input_shape);
        let model = reshape_model(&input_shape, &output_shape);
        unit_test_op(model, &[input]);
    }
}
