use common::parallel::par_enabled;
use std::array;

use crate::utils::{
    dims::{transpose_flat_matrix, EinsumDims},
    opening_id_builder::{AccOpeningAccessor, Target},
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
};
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
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
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
};
use rayon::prelude::*;

// TODO: Add [DT24] opts

const DEGREE_BOUND: usize = 3;

/// Parameters for proving Einsum bmk,kbn->mbn operations.
///
/// This implements batch matrix multiplication with transposed middle tensor.
#[derive(Clone)]
pub struct BmkKbnMbnParams<F: JoltField> {
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
    einsum_dims: EinsumDims,
    log_b: usize,
    log_k: usize,
}

impl<F: JoltField> BmkKbnMbnParams<F> {
    /// Create new parameters for bmk,kbn->mbn einsum.
    pub fn new(
        computation_node: ComputationNode,
        einsum_dims: EinsumDims,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);
        let r_node_output = accessor.get_reduced_opening().0;
        let log_b = einsum_dims.left_operand()[0].log_2();
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

impl<F: JoltField> SumcheckInstanceParams<F> for BmkKbnMbnParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        accessor.get_reduced_opening().1
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(challenges.to_vec())
    }

    fn num_rounds(&self) -> usize {
        self.log_b + self.log_k
    }
}

/// Prover state for bmk,kbn->mbn einsum sumcheck protocol.
pub struct BmkKbnMbnProver<F: JoltField> {
    params: BmkKbnMbnParams<F>,
    left_operand: MultilinearPolynomial<F>,
    right_operand: MultilinearPolynomial<F>,
    eq_r_b: MultilinearPolynomial<F>,
}

impl<F: JoltField> BmkKbnMbnProver<F> {
    /// Initialize the prover with trace data and parameters for bmk,kbn->mbn einsum.
    #[tracing::instrument(skip_all, name = "BmkKbnMbnProver::initialize")]
    pub fn initialize(trace: &Trace, params: BmkKbnMbnParams<F>) -> Self {
        let LayerData {
            operands,
            output: _,
        } = Trace::layer_data(trace, &params.computation_node);
        let [left_operand, right_operand] = operands[..] else {
            panic!("Expected two operands for BmkKbnMbn operation")
        };
        let (b, m, k, n) = (
            params.einsum_dims.left_operand()[0],
            params.einsum_dims.left_operand()[1],
            params.einsum_dims.left_operand()[2],
            params.einsum_dims.right_operand()[2],
        );
        let (r_m, r_bn) = params.r_node_output.split_at(m.log_2());
        let (r_b, r_n) = r_bn.split_at(b.log_2());
        let eq_r_m = EqPolynomial::evals(&r_m.r);
        let eq_r_n = EqPolynomial::evals(&r_n.r);
        let mut lo_r_m: Vec<F> = unsafe_allocate_zero_vec(k * b);
        let mut ro_r_n: Vec<F> = unsafe_allocate_zero_vec(k * b);
        lo_r_m.par_chunks_mut(k).enumerate().for_each(|(h, row)| {
            for j in 0..k {
                row[j] = (0..m)
                    .map(|i| F::from_i32(left_operand[h * (k * m) + i * (k) + j]) * eq_r_m[i])
                    .sum();
            }
        });
        ro_r_n.par_chunks_mut(b).enumerate().for_each(|(j, col)| {
            for h in 0..b {
                col[h] = (0..n)
                    .map(|l| F::from_i32(right_operand[j * (b * n) + h * (n) + l]) * eq_r_n[l])
                    .sum();
            }
        });
        let lo_r_m = transpose_flat_matrix(lo_r_m, b, k);
        let eq_r_b = MultilinearPolynomial::from(EqPolynomial::evals(&r_b.r));
        let left_operand = MultilinearPolynomial::from(lo_r_m);
        let right_operand = MultilinearPolynomial::from(ro_r_n);
        Self {
            params,
            left_operand,
            right_operand,
            eq_r_b,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for BmkKbnMbnProver<F> {
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
            .with_min_len(par_enabled())
            .map(|jh| {
                let left_evals =
                    left_operand.sumcheck_evals_array::<DEGREE_BOUND>(jh, BindingOrder::HighToLow);
                let right_evals =
                    right_operand.sumcheck_evals_array::<DEGREE_BOUND>(jh, BindingOrder::HighToLow);
                let eq_evals = if round < self.params.log_k {
                    let h = jh % (1 << self.params.log_b);
                    [self.eq_r_b.get_bound_coeff(h); 3]
                } else {
                    self.eq_r_b
                        .sumcheck_evals_array::<DEGREE_BOUND>(jh, BindingOrder::HighToLow)
                };
                [
                    left_evals[0] * right_evals[0] * eq_evals[0], // eval at 0
                    left_evals[1] * right_evals[1] * eq_evals[1], // eval at 2
                    left_evals[2] * right_evals[2] * eq_evals[2], // eval at 3
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
        if round >= self.params.log_k {
            self.eq_r_b.bind_parallel(r_j, BindingOrder::HighToLow)
        };
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let sumcheck_challenges = sumcheck_challenges.into_opening();
        let (b, m) = (
            self.params.einsum_dims.left_operand()[0],
            self.params.einsum_dims.left_operand()[1],
        );
        let (r_m, r_bn) = self.params.r_node_output.split_at(m.log_2());
        let (_, r_n) = r_bn.split_at(b.log_2());
        let (r_j, r_h) = sumcheck_challenges.split_at(self.params.log_k);

        let r_left_node_output = [r_h, &r_m.r, r_j].concat();
        let left_opening_point = self.params.normalize_opening_point(&r_left_node_output);
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .to_provider(transcript, left_opening_point.clone());
        provider.append_node_io(Target::Input(0), self.left_operand.final_claim());

        let r_right_node_output = [sumcheck_challenges.as_slice(), &r_n.r].concat();
        let right_opening_point = self.params.normalize_opening_point(&r_right_node_output);
        provider.update_point(right_opening_point);
        provider.append_node_io(Target::Input(1), self.right_operand.final_claim());
    }
}

/// Verifier for bmk,kbn->mbn einsum sumcheck protocol.
pub struct BmkKbnMbnVerifier<F: JoltField> {
    params: BmkKbnMbnParams<F>,
}

impl<F: JoltField> BmkKbnMbnVerifier<F> {
    /// Create new verifier for bmk,kbn->mbn einsum.
    #[tracing::instrument(skip_all, name = "BmkKbnMbnVerifier::new")]
    pub fn new(
        computation_node: ComputationNode,
        einsum_dims: EinsumDims,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = BmkKbnMbnParams::new(computation_node, einsum_dims, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for BmkKbnMbnVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.params.computation_node);

        let (b, m) = (
            self.params.einsum_dims.left_operand()[0],
            self.params.einsum_dims.left_operand()[1],
        );
        let (_, r_bn) = self.params.r_node_output.split_at(m.log_2());
        let (r_b, _) = r_bn.split_at(b.log_2());
        let (_, r_h) = sumcheck_challenges.split_at(self.params.log_k);

        let left_operand_claim = accessor.get_node_io(Target::Input(0)).1;
        let right_operand_claim = accessor.get_node_io(Target::Input(1)).1;
        left_operand_claim * right_operand_claim * EqPolynomial::mle(&r_b.r, r_h)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let sumcheck_challenges = sumcheck_challenges.into_opening();
        let (b, m) = (
            self.params.einsum_dims.left_operand()[0],
            self.params.einsum_dims.left_operand()[1],
        );
        let (r_m, r_bn) = self.params.r_node_output.split_at(m.log_2());
        let (_, r_n) = r_bn.split_at(b.log_2());
        let (r_j, r_h) = sumcheck_challenges.split_at(self.params.log_k);

        let r_left_node_output = [r_h, &r_m.r, r_j].concat();
        let left_opening_point = self.params.normalize_opening_point(&r_left_node_output);
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .to_provider(transcript, left_opening_point.clone());
        provider.append_node_io(Target::Input(0));

        let r_right_node_output = [sumcheck_challenges.as_slice(), &r_n.r].concat();
        let right_opening_point = self.params.normalize_opening_point(&r_right_node_output);
        provider.update_point(right_opening_point);
        provider.append_node_io(Target::Input(1));
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{model::test::ModelBuilder, model::Model, tensor::Tensor};
    use rand::{rngs::StdRng, SeedableRng};

    fn bmk_kbn_mbn_model(rng: &mut StdRng, b: usize, m: usize, k: usize, n: usize) -> Model {
        let mut builder = ModelBuilder::new();
        let i = builder.input(vec![b, m, k]);
        let c = builder.constant(Tensor::random_small(rng, &[k, b, n]));
        let res = builder.einsum("bmk,kbn->mbn", vec![i, c], vec![m, b, n]);
        builder.mark_output(res);
        builder.build()
    }

    #[test]
    fn test_bmk_kbn_mbn() {
        let (b, m, k, n) = (1 << 3, 1 << 2, 1 << 4, 1 << 5);
        let mut rng = StdRng::seed_from_u64(0x878);
        let input = Tensor::<i32>::random_small(&mut rng, &[b, m, k]);
        let model = bmk_kbn_mbn_model(&mut rng, b, m, k, n);
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "TODO: non-power-of-two einsum path not fully validated yet"]
    fn test_bmk_kbn_mbn_non_power_of_two_input_len() {
        let (b, m, k, n) = (3, 5, 9, 7);
        let mut rng = StdRng::seed_from_u64(0x879);
        let input = Tensor::<i32>::random_small(&mut rng, &[b, m, k]);
        let model = bmk_kbn_mbn_model(&mut rng, b, m, k, n);
        unit_test_op(model, &[input]);
    }
}
