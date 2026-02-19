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
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
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
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
};
use rayon::prelude::*;

use crate::utils::dims::EinsumDims;

// TODO: Add [DT24] opts

const DEGREE_BOUND: usize = 3;

/// Parameters for proving Einsum mbk,bnk->bmn operations.
///
/// This implements batch matrix multiplication with specific dimension ordering.
#[derive(Clone)]
pub struct MbkBnkBmnParams<F: JoltField> {
    r_node_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
    einsum_dims: EinsumDims,
    log_b: usize,
    log_k: usize,
}

impl<F: JoltField> MbkBnkBmnParams<F> {
    /// Create new parameters for mbk,bnk->bmn einsum.
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
        let log_b = einsum_dims.left_operand()[1].log_2();
        let log_k = einsum_dims.left_operand()[2].log_2();
        Self {
            r_node_output,
            computation_node,
            einsum_dims,
            log_b,
            log_k,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for MbkBnkBmnParams<F> {
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
        self.log_b + self.log_k
    }
}

/// Prover state for mbk,bnk->bmn einsum sumcheck protocol.
pub struct MbkBnkBmnProver<F: JoltField> {
    params: MbkBnkBmnParams<F>,
    left_operand: MultilinearPolynomial<F>,
    right_operand: MultilinearPolynomial<F>,
    eq_r_b: MultilinearPolynomial<F>,
    eq_rb_rh: Option<F>,
}

impl<F: JoltField> MbkBnkBmnProver<F> {
    /// Initialize the prover with trace data and parameters for mbk,bnk->bmn einsum.
    #[tracing::instrument(skip_all, name = "MbkBnkBmnProver::initialize")]
    pub fn initialize(trace: &Trace, params: MbkBnkBmnParams<F>) -> Self {
        let LayerData {
            operands,
            output: _,
        } = Trace::layer_data(trace, &params.computation_node);
        let [left_operand, right_operand] = operands[..] else {
            panic!("Expected two operands for MbkBnkBmn operation")
        };
        let (m, b, k, n) = (
            params.einsum_dims.left_operand()[0],
            params.einsum_dims.left_operand()[1],
            params.einsum_dims.left_operand()[2],
            params.einsum_dims.right_operand()[1],
        );
        let (r_b, r_mn) = params.r_node_output.split_at(b.log_2());
        let (r_m, r_n) = r_mn.split_at(m.log_2());
        let eq_r_m = EqPolynomial::evals(r_m);
        let eq_r_n = EqPolynomial::evals(r_n);
        let mut lo_r_m: Vec<F> = unsafe_allocate_zero_vec(k * b);
        let mut ro_r_n: Vec<F> = unsafe_allocate_zero_vec(k * b);
        lo_r_m.par_chunks_mut(k).enumerate().for_each(|(h, row)| {
            for j in 0..k {
                row[j] = (0..m)
                    .map(|i| F::from_i32(left_operand[i * (k * b) + h * (k) + j]) * eq_r_m[i])
                    .sum();
            }
        });
        ro_r_n.par_chunks_mut(k).enumerate().for_each(|(h, row)| {
            for j in 0..k {
                row[j] = (0..n)
                    .map(|l| F::from_i32(right_operand[h * (n * k) + l * (k) + j]) * eq_r_n[l])
                    .sum();
            }
        });
        let eq_r_b = MultilinearPolynomial::from(EqPolynomial::evals(r_b));
        let left_operand = MultilinearPolynomial::from(lo_r_m);
        let right_operand = MultilinearPolynomial::from(ro_r_n);
        Self {
            params,
            left_operand,
            right_operand,
            eq_r_b,
            eq_rb_rh: None,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for MbkBnkBmnProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            left_operand,
            right_operand,
            ..
        } = self;
        let half_poly_len = right_operand.len() / 2;
        let uni_poly_evals: [F; DEGREE_BOUND] = (0..half_poly_len)
            .into_par_iter()
            .map(|hj| {
                let l_evals =
                    left_operand.sumcheck_evals_array::<DEGREE_BOUND>(hj, BindingOrder::HighToLow);
                let r_evals =
                    right_operand.sumcheck_evals_array::<DEGREE_BOUND>(hj, BindingOrder::HighToLow);
                let eq_evals = if round < self.params.log_b {
                    let h = hj >> self.params.log_k;
                    self.eq_r_b
                        .sumcheck_evals_array::<DEGREE_BOUND>(h, BindingOrder::HighToLow)
                } else {
                    let eq_rb_rh = self.eq_rb_rh.expect("eq_rb_rh should be set");
                    [eq_rb_rh; 3]
                };
                [
                    l_evals[0] * r_evals[0] * eq_evals[0], // eval at 0
                    l_evals[1] * r_evals[1] * eq_evals[1], // eval at 2
                    l_evals[2] * r_evals[2] * eq_evals[2], // eval at 3
                ]
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |running, new| array::from_fn(|i| running[i] + new[i]),
            );
        UniPoly::from_evals_and_hint(previous_claim, &uni_poly_evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.left_operand
            .bind_parallel(r_j, BindingOrder::HighToLow);
        self.right_operand
            .bind_parallel(r_j, BindingOrder::HighToLow);
        if round < self.params.log_b {
            self.eq_r_b.bind_parallel(r_j, BindingOrder::HighToLow);
        };
        // cache eq eval
        if round == self.params.log_b - 1 {
            self.eq_rb_rh = Some(self.eq_r_b.final_sumcheck_claim());
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let (m, b) = (
            self.params.einsum_dims.left_operand()[0],
            self.params.einsum_dims.left_operand()[1],
        );
        let (_, r_mn) = self.params.r_node_output.split_at(b.log_2());
        let (r_m, r_n) = r_mn.split_at(m.log_2());

        let r_left_node_output = [r_m, sumcheck_challenges].concat();
        let left_opening_point = self.params.normalize_opening_point(&r_left_node_output);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            left_opening_point.clone(),
            self.left_operand.final_sumcheck_claim(),
        );

        let (r_h, r_j) = sumcheck_challenges.split_at(b.log_2());
        let r_right_node_output = [r_h, r_n, r_j].concat();
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

/// Verifier for mbk,bnk->bmn einsum sumcheck protocol.
pub struct MbkBnkBmnVerifier<F: JoltField> {
    params: MbkBnkBmnParams<F>,
}

impl<F: JoltField> MbkBnkBmnVerifier<F> {
    /// Create new verifier for mbk,bnk->bmn einsum.
    #[tracing::instrument(skip_all, name = "MbkBnkBmnVerifier::new")]
    pub fn new(
        computation_node: ComputationNode,
        einsum_dims: EinsumDims,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = MbkBnkBmnParams::new(computation_node, einsum_dims, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for MbkBnkBmnVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let b = self.params.einsum_dims.left_operand()[1];
        let (r_b, _) = self.params.r_node_output.split_at(b.log_2());
        let (r_h, _r_j) = sumcheck_challenges.split_at(self.params.log_b);
        let [left_operand_claim, right_operand_claim] =
            accumulator.get_operand_claims::<2>(self.params.computation_node.idx);
        left_operand_claim * right_operand_claim * EqPolynomial::mle(r_b, r_h)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let (m, b) = (
            self.params.einsum_dims.left_operand()[0],
            self.params.einsum_dims.left_operand()[1],
        );
        let (_, r_other) = self.params.r_node_output.split_at(b.log_2());
        let (r_m, r_n) = r_other.split_at(m.log_2());

        let r_left_node_output = [r_m, sumcheck_challenges].concat();
        let left_opening_point = self.params.normalize_opening_point(&r_left_node_output);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            left_opening_point.clone(),
        );

        let (r_h, r_j) = sumcheck_challenges.split_at(b.log_2());
        let r_right_node_output = [r_h, r_n, r_j].concat();
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

    fn mbk_bnk_bmn_model(rng: &mut StdRng, m: usize, b: usize, k: usize, n: usize) -> Model {
        let mut builder = ModelBuilder::new();
        let i = builder.input(vec![m, b, k]);
        let c = builder.constant(Tensor::random_small(rng, &[b, n, k]));
        let res = builder.einsum("mbk,bnk->bmn", vec![i, c], vec![b, m, n]);
        builder.mark_output(res);
        builder.build()
    }

    #[test]
    fn test_mbk_bnk_bmn() {
        let (m, b, k, n) = (1 << 2, 1 << 3, 1 << 4, 1 << 5);
        let mut rng = StdRng::seed_from_u64(0x86078);
        let input = Tensor::<i32>::random_small(&mut rng, &[m, b, k]);
        let model = mbk_bnk_bmn_model(&mut rng, m, b, k, n);
        unit_test_op(model, &[input]);
    }
}
