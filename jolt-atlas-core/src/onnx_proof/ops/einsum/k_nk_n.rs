use crate::utils::opening_id_builder::{OpeningIdBuilder, OpeningTarget};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
};
use common::parallel::par_enabled;
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::math::Math,
};
use rayon::prelude::*;
use std::array;

use crate::utils::dims::EinsumDims;

const DEGREE_BOUND: usize = 2;

/// Parameters for proving Einsum k,nk->n operations.
///
/// This implements matrix-vector multiplication variants.
#[derive(Clone)]
pub struct KNkNParams<F: JoltField> {
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
    einsum_dims: EinsumDims,
}

impl<F: JoltField> KNkNParams<F> {
    /// Create new parameters for k,nk->n einsum.
    pub fn new(
        computation_node: ComputationNode,
        einsum_dims: EinsumDims,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let r_node_output = accumulator.get_node_output_opening(computation_node.idx).0;
        Self {
            r_node_output,
            computation_node,
            einsum_dims,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for KNkNParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, einsum_claim) = accumulator.get_node_output_opening(self.computation_node.idx);
        einsum_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(challenges.to_vec())
    }

    fn num_rounds(&self) -> usize {
        self.einsum_dims.left_operand()[0].log_2()
    }
}

/// Prover state for k,nk->n einsum sumcheck protocol.
pub struct KNkNProver<F: JoltField> {
    params: KNkNParams<F>,
    left_operand: MultilinearPolynomial<F>,
    right_operand: MultilinearPolynomial<F>,
}

impl<F: JoltField> KNkNProver<F> {
    /// Initialize the prover with trace data and parameters for k,nk->n einsum.
    #[tracing::instrument(skip_all, name = "KNkNProver::initialize")]
    pub fn initialize(trace: &Trace, params: KNkNParams<F>) -> Self {
        let LayerData {
            operands,
            output: _,
        } = Trace::layer_data(trace, &params.computation_node);
        let [left_operand, right_operand] = operands[..] else {
            panic!("Expected two operands for KNkN operation")
        };
        let (n, k) = (
            params.einsum_dims.output()[0],
            params.einsum_dims.left_operand()[0],
        );
        let eq_r_node_output = EqPolynomial::evals(&params.r_node_output.r);
        let right_operand: Vec<F> = (0..k)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|j| {
                (0..n)
                    .map(|h| F::from_i32(right_operand[h * k + j]) * eq_r_node_output[h])
                    .sum()
            })
            .collect();
        let left_operand = MultilinearPolynomial::from(left_operand.clone());
        let right_operand = MultilinearPolynomial::from(right_operand);
        Self {
            params,
            left_operand,
            right_operand,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for KNkNProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            left_operand,
            right_operand,
            ..
        } = self;
        let half_poly_len = left_operand.len() / 2;
        let uni_poly_evals: [F; 2] = (0..half_poly_len)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|i| {
                let l_evals = left_operand.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::HighToLow);
                let r_evals =
                    right_operand.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::HighToLow);
                [l_evals[0] * r_evals[0], l_evals[1] * r_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| array::from_fn(|i| running[i] + new[i]),
            );
        UniPoly::from_evals_and_hint(previous_claim, &uni_poly_evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.left_operand
            .bind_parallel(r_j, BindingOrder::HighToLow);
        self.right_operand
            .bind_parallel(r_j, BindingOrder::HighToLow);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let node = &self.params.computation_node;
        let builder = OpeningIdBuilder::new(node);
        let left_opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());

        let left_opening_id = builder.node_io(OpeningTarget::Input(0));
        accumulator.append_virtual(
            transcript,
            left_opening_id,
            left_opening_point.clone(),
            self.left_operand.final_sumcheck_claim(),
        );

        let r_right_node_output = [
            self.params.r_node_output.r.as_slice(),
            &sumcheck_challenges.into_opening(),
        ]
        .concat();
        let right_opening_point = self.params.normalize_opening_point(&r_right_node_output);

        let right_opening_id = builder.node_io(OpeningTarget::Input(1));
        accumulator.append_virtual(
            transcript,
            right_opening_id,
            right_opening_point,
            self.right_operand.final_sumcheck_claim(),
        );
    }
}

/// Verifier for k,nk->n einsum sumcheck protocol.
pub struct KNkNVerifier<F: JoltField> {
    params: KNkNParams<F>,
}

impl<F: JoltField> KNkNVerifier<F> {
    /// Create new verifier for k,nk->n einsum.
    #[tracing::instrument(skip_all, name = "KNkNVerifier::new")]
    pub fn new(
        computation_node: ComputationNode,
        einsum_dims: EinsumDims,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = KNkNParams::new(computation_node, einsum_dims, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for KNkNVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let left_operand_claim = accumulator.get_node_output_claim(
            self.params.computation_node.inputs[0],
            self.params.computation_node.idx,
        );
        let right_operand_claim = accumulator.get_node_output_claim(
            self.params.computation_node.inputs[1],
            self.params.computation_node.idx,
        );
        left_operand_claim * right_operand_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let node = &self.params.computation_node;
        let builder = OpeningIdBuilder::new(node);
        let left_opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let left_opening_id = builder.node_io(OpeningTarget::Input(0));
        accumulator.append_virtual(transcript, left_opening_id, left_opening_point.clone());

        let r_right_node_output = [
            self.params.r_node_output.r.as_slice(),
            &sumcheck_challenges.into_opening(),
        ]
        .concat();
        let right_opening_point = self.params.normalize_opening_point(&r_right_node_output);
        let right_opening_id = builder.node_io(OpeningTarget::Input(1));
        accumulator.append_virtual(transcript, right_opening_id, right_opening_point);
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{model::test::ModelBuilder, model::Model, tensor::Tensor};
    use rand::{rngs::StdRng, SeedableRng};

    fn matvec_model(rng: &mut StdRng, k: usize, n: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![k]);
        let c = b.constant(Tensor::random_small(rng, &[n, k]));
        let res = b.einsum("k,nk->n", vec![i, c], vec![n]);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_k_nk_n() {
        let k = 1 << 7;
        let n = 1 << 8;
        let mut rng = StdRng::seed_from_u64(0x878);
        let input = Tensor::<i32>::random_small(&mut rng, &[k]);
        let model = matvec_model(&mut rng, k, n);
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "TODO: non-power-of-two einsum path not fully validated yet"]
    fn test_k_nk_n_non_power_of_two_input_len() {
        let k = 65;
        let n = 33;
        let mut rng = StdRng::seed_from_u64(0x879);
        let input = Tensor::<i32>::random_small(&mut rng, &[k]);
        let model = matvec_model(&mut rng, k, n);
        unit_test_op(model, &[input]);
    }
}
