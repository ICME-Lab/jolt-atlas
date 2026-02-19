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

use crate::utils::dims::{transpose_flat_matrix, EinsumDims};

// TODO: Add [DT24] opts

const DEGREE_BOUND: usize = 3;

/// Parameters for proving Einsum bmk,bkn->mbn operations.
///
/// This implements batch matrix multiplication with specific dimension ordering.
#[derive(Clone)]
pub struct BmkBknMbnParams<F: JoltField> {
    r_node_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
    einsum_dims: EinsumDims,
    log_b: usize,
    log_k: usize,
}

impl<F: JoltField> BmkBknMbnParams<F> {
    /// Create new parameters for bmk,bkn->mbn einsum.
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

impl<F: JoltField> SumcheckInstanceParams<F> for BmkBknMbnParams<F> {
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

/// Prover state for bmk,bkn->mbn einsum sumcheck protocol.
pub struct BmkBknMbnProver<F: JoltField> {
    params: BmkBknMbnParams<F>,
    left_operand: MultilinearPolynomial<F>,
    right_operand: MultilinearPolynomial<F>,
    eq_r_b: MultilinearPolynomial<F>,
}

impl<F: JoltField> BmkBknMbnProver<F> {
    /// Initialize the prover with trace data and parameters for bmk,bkn->mbn einsum.
    #[tracing::instrument(skip_all, name = "BmkBknMbnProver::initialize")]
    pub fn initialize(trace: &Trace, params: BmkBknMbnParams<F>) -> Self {
        let LayerData {
            operands,
            output: _,
        } = Trace::layer_data(trace, &params.computation_node);
        let [left_operand, right_operand] = operands[..] else {
            panic!("Expected two operands for BmkBknMbn operation")
        };
        let (b, m, k, n) = (
            params.einsum_dims.left_operand()[0],
            params.einsum_dims.left_operand()[1],
            params.einsum_dims.left_operand()[2],
            params.einsum_dims.right_operand()[2],
        );
        let (r_m, r_bn) = params.r_node_output.split_at(m.log_2());
        let (r_b, r_n) = r_bn.split_at(b.log_2());
        let eq_r_m = EqPolynomial::evals(r_m);
        let eq_r_n = EqPolynomial::evals(r_n);
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
        let ro_r_n = transpose_flat_matrix(ro_r_n, b, k);
        let eq_r_b = MultilinearPolynomial::from(EqPolynomial::evals(r_b));
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

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for BmkBknMbnProver<F> {
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
        let (b, m) = (
            self.params.einsum_dims.left_operand()[0],
            self.params.einsum_dims.left_operand()[1],
        );
        let (r_m, r_bn) = self.params.r_node_output.split_at(m.log_2());
        let (_, r_n) = r_bn.split_at(b.log_2());
        let (r_j, r_h) = sumcheck_challenges.split_at(self.params.log_k);

        let r_left_node_output = [r_h, r_m, r_j].concat();
        let left_opening_point = self.params.normalize_opening_point(&r_left_node_output);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            left_opening_point.clone(),
            self.left_operand.final_sumcheck_claim(),
        );

        let r_right_node_output = [r_h, r_j, r_n].concat();
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

/// Verifier for bmk,bkn->mbn einsum sumcheck protocol.
pub struct BmkBknMbnVerifier<F: JoltField> {
    params: BmkBknMbnParams<F>,
}

impl<F: JoltField> BmkBknMbnVerifier<F> {
    /// Create new verifier for bmk,bkn->mbn einsum.
    #[tracing::instrument(skip_all, name = "BmkBknMbnVerifier::new")]
    pub fn new(
        computation_node: ComputationNode,
        einsum_dims: EinsumDims,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = BmkBknMbnParams::new(computation_node, einsum_dims, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for BmkBknMbnVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let (b, m) = (
            self.params.einsum_dims.left_operand()[0],
            self.params.einsum_dims.left_operand()[1],
        );
        let (_, r_bn) = self.params.r_node_output.split_at(m.log_2());
        let (r_b, _) = r_bn.split_at(b.log_2());
        let (_, r_h) = sumcheck_challenges.split_at(self.params.log_k);
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
        let (b, m) = (
            self.params.einsum_dims.left_operand()[0],
            self.params.einsum_dims.left_operand()[1],
        );
        let (r_m, r_bn) = self.params.r_node_output.split_at(m.log_2());
        let (_, r_n) = r_bn.split_at(b.log_2());
        let (r_j, r_h) = sumcheck_challenges.split_at(self.params.log_k);

        let r_left_node_output = [r_h, r_m, r_j].concat();
        let left_opening_point = self.params.normalize_opening_point(&r_left_node_output);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            left_opening_point.clone(),
        );

        let r_right_node_output = [r_h, r_j, r_n].concat();
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
    use super::*;
    use crate::utils::dims::EINSUM_REGISTRY;
    use ark_bn254::Fr;
    use atlas_onnx_tracer::{
        model::{
            self,
            trace::{LayerData, Trace},
        },
        ops::{Einsum, Operator},
        tensor::Tensor,
    };
    use common::VirtualPolynomial;
    use joltworks::{
        field::JoltField,
        poly::{
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            opening_proof::{
                OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
                BIG_ENDIAN,
            },
        },
        subprotocols::sumcheck::Sumcheck,
        transcripts::{Blake2bTranscript, Transcript},
    };
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_bmk_bkn_mbn() {
        let log_m = 2;
        let log_b = 3;
        let log_k = 4;
        let log_n = 5;
        let m = 1 << log_m;
        let b = 1 << log_b;
        let k = 1 << log_k;
        let n = 1 << log_n;
        let mut rng = StdRng::seed_from_u64(0x878);
        let input = Tensor::<i32>::random_small(&mut rng, &[b, m, k]);
        let model = model::test::bmk_bkn_mbn_model(&mut rng, b, m, k, n);
        let trace = model.trace(&[input]);

        let prover_transcript = &mut Blake2bTranscript::new(&[]);
        let mut prover_opening_accumulator: ProverOpeningAccumulator<Fr> =
            ProverOpeningAccumulator::new();

        let r_node_output: Vec<<Fr as JoltField>::Challenge> =
            prover_transcript.challenge_vector_optimized::<Fr>(log_m + log_b + log_n);

        let output_index = model.outputs()[0];
        let computation_node = &model[output_index];
        let LayerData {
            operands: _,
            output,
        } = Trace::layer_data(&trace, computation_node);

        let bmk_bkn_mbn_claim =
            MultilinearPolynomial::from(output.clone()).evaluate(&r_node_output);
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.clone().into(),
            bmk_bkn_mbn_claim,
        );

        let config = match &computation_node.operator {
            Operator::Einsum(Einsum { equation }) => EINSUM_REGISTRY
                .iter()
                .find(|(pattern, _)| pattern == &equation.as_str())
                .map(|(_, config)| config)
                .unwrap_or_else(|| {
                    panic!("Einsum equation ({equation}) not supported by precompile system")
                }),
            _ => panic!("Unexpected operator"),
        };
        let einsum_dims = (config.dims_extractor)(computation_node, &model);

        let params: BmkBknMbnParams<Fr> = BmkBknMbnParams::new(
            computation_node.clone(),
            einsum_dims.clone(),
            &prover_opening_accumulator,
        );
        let mut prover_sumcheck = BmkBknMbnProver::initialize(&trace, params);
        let (proof, _) = Sumcheck::prove(
            &mut prover_sumcheck,
            &mut prover_opening_accumulator,
            prover_transcript,
        );

        let verifier_transcript = &mut Blake2bTranscript::new(&[]);
        verifier_transcript.compare_to(prover_transcript.clone());
        let mut verifier_opening_accumulator: VerifierOpeningAccumulator<Fr> =
            VerifierOpeningAccumulator::new();
        let _r_node_output: Vec<<Fr as JoltField>::Challenge> =
            verifier_transcript.challenge_vector_optimized::<Fr>(log_m + log_b + log_n);

        // Take claims
        for (key, (_, value)) in &prover_opening_accumulator.openings {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier_opening_accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }
        verifier_opening_accumulator.virtual_operand_claims =
            prover_opening_accumulator.virtual_operand_claims.clone();

        verifier_opening_accumulator.append_virtual(
            verifier_transcript,
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.into(),
        );

        let verifier_sumcheck = BmkBknMbnVerifier::new(
            computation_node.clone(),
            einsum_dims,
            &verifier_opening_accumulator,
        );
        let _ = Sumcheck::verify(
            &proof,
            &verifier_sumcheck,
            &mut verifier_opening_accumulator,
            verifier_transcript,
        );

        // Evaluate input at operand point and check it equals the expected output claim
        let [left_operand_idx, right_operand_idx] = computation_node.inputs[..] else {
            panic!("Expected two operands for BmkBknMbn operation")
        };

        // check left operand
        let left_operand_layer = &model[left_operand_idx];
        let left_operand_data = Trace::layer_data(&trace, left_operand_layer).output.clone();
        let left_operand_poly = MultilinearPolynomial::from(left_operand_data);
        let (opening_point, expected_left_operand_output_claim) = verifier_opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(left_operand_idx),
                SumcheckId::Execution,
            );
        let left_operand_eval = left_operand_poly.evaluate(&opening_point.r);
        assert_eq!(left_operand_eval, expected_left_operand_output_claim);

        // check right operand
        let right_operand_layer = &model[right_operand_idx];
        let right_operand_data = Trace::layer_data(&trace, right_operand_layer)
            .output
            .clone();
        let right_operand_poly = MultilinearPolynomial::from(right_operand_data);
        let (opening_point, right_operand_output_claim) = verifier_opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(right_operand_idx),
                SumcheckId::Execution,
            );
        let right_operand_eval = right_operand_poly.evaluate(&opening_point.r);
        assert_eq!(right_operand_eval, right_operand_output_claim);
    }
}
