use std::array;

use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
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

use crate::utils::dims::EinsumDims;

const DEGREE_BOUND: usize = 2;

/// Parameters for proving Einsum mk,kn->mn operations.
///
/// This implements standard matrix multiplication.
#[derive(Clone)]
pub struct MkKnMnParams<F: JoltField> {
    r_node_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
    einsum_dims: EinsumDims,
}

impl<F: JoltField> MkKnMnParams<F> {
    /// Create new parameters for mk,kn->mn einsum.
    pub fn new(
        computation_node: ComputationNode,
        einsum_dims: EinsumDims,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
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
            einsum_dims,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for MkKnMnParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, einsum_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(self.computation_node.idx),
            SumcheckId::Execution,
        );
        einsum_claim
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(challenges.to_vec())
    }

    fn num_rounds(&self) -> usize {
        self.einsum_dims.right_operand()[0].log_2()
    }
}

/// Prover state for mk,kn->mn einsum sumcheck protocol.
pub struct MkKnMnProver<F: JoltField> {
    params: MkKnMnParams<F>,
    left_operand: MultilinearPolynomial<F>,
    right_operand: MultilinearPolynomial<F>,
}

impl<F: JoltField> MkKnMnProver<F> {
    /// Initialize the prover with trace data and parameters for mk,kn->mn einsum.
    #[tracing::instrument(skip_all, name = "MkKnMnProver::initialize")]
    pub fn initialize(trace: &Trace, params: MkKnMnParams<F>) -> Self {
        let LayerData {
            operands,
            output: _,
        } = Trace::layer_data(trace, &params.computation_node);
        let [left_operand, right_operand] = operands[..] else {
            panic!("Expected two operands for MkKnMn operation")
        };
        let (m, n, k) = (
            params.einsum_dims.output()[0],
            params.einsum_dims.output()[1],
            params.einsum_dims.right_operand()[0],
        );
        let (r_m, r_n) = params.r_node_output.split_at(m.log_2());
        let (eq_r_m, eq_r_n) = (EqPolynomial::evals(r_m), EqPolynomial::evals(r_n));
        let left_operand: Vec<F> = (0..k)
            .into_par_iter()
            .map(|j| {
                (0..m)
                    .map(|i| F::from_i32(left_operand[i * k + j]) * eq_r_m[i])
                    .sum()
            })
            .collect();
        let right_operand: Vec<F> = (0..k)
            .into_par_iter()
            .map(|j| {
                (0..n)
                    .map(|h| F::from_i32(right_operand[j * n + h]) * eq_r_n[h])
                    .sum()
            })
            .collect();
        let left_operand = MultilinearPolynomial::from(left_operand);
        let right_operand = MultilinearPolynomial::from(right_operand);
        Self {
            params,
            left_operand,
            right_operand,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for MkKnMnProver<F> {
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
        let (r_m, r_n) = self
            .params
            .r_node_output
            .split_at(self.params.einsum_dims.output()[0].log_2());
        let r_left_node_output = [r_m, sumcheck_challenges].concat();
        let left_opening_point = self.params.normalize_opening_point(&r_left_node_output);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            left_opening_point.clone(),
            self.left_operand.final_sumcheck_claim(),
        );

        let r_right_node_output = [sumcheck_challenges, r_n].concat();
        let right_opening_point = self.params.normalize_opening_point(&r_right_node_output);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
            SumcheckId::Execution,
            right_opening_point,
            self.right_operand.final_sumcheck_claim(),
        );
        accumulator.cache_virtual_operand_claims(transcript, &self.params.computation_node);
    }
}

/// Verifier for mk,kn->mn einsum sumcheck protocol.
pub struct MkKnMnVerifier<F: JoltField> {
    params: MkKnMnParams<F>,
}

impl<F: JoltField> MkKnMnVerifier<F> {
    /// Create new verifier for mk,kn->mn einsum.
    #[tracing::instrument(skip_all, name = "MkKnMnVerifier::new")]
    pub fn new(
        computation_node: ComputationNode,
        einsum_dims: EinsumDims,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = MkKnMnParams::new(computation_node, einsum_dims, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for MkKnMnVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let [left_operand_claim, right_operand_claim] =
            accumulator.get_operand_claims::<2>(self.params.computation_node.idx);
        left_operand_claim * right_operand_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let (r_m, r_n) = self
            .params
            .r_node_output
            .split_at(self.params.einsum_dims.output()[0].log_2());
        let r_left_node_output = [r_m, sumcheck_challenges].concat();
        let left_opening_point = self.params.normalize_opening_point(&r_left_node_output);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            left_opening_point.clone(),
        );
        let r_right_node_output = [sumcheck_challenges, r_n].concat();
        let right_opening_point = self.params.normalize_opening_point(&r_right_node_output);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
            SumcheckId::Execution,
            right_opening_point,
        );
        accumulator.append_operand_claims(transcript, self.params.computation_node.idx);
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{model::test::ModelBuilder, model::Model, tensor::Tensor};
    use rand::{rngs::StdRng, SeedableRng};

    fn matmul_model(rng: &mut StdRng, m: usize, k: usize, n: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![m, k]);
        let c = b.constant(Tensor::random_small(rng, &[k, n]));
        let res = b.einsum("mk,kn->mn", vec![i, c], vec![m, n]);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_mk_kn_mn() {
        let m = 1 << 6;
        let k = 1 << 7;
        let n = 1 << 8;
        let mut rng = StdRng::seed_from_u64(0x878);
        let input = Tensor::<i32>::random_small(&mut rng, &[m, k]);
        let model = matmul_model(&mut rng, m, k, n);
        unit_test_op(model, &[input]);
    }
}
