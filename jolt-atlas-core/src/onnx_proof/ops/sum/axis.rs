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
    utils::math::Math,
};
use rayon::prelude::*;

use crate::utils::dims::{SumAxis, SumConfig};

const DEGREE_BOUND: usize = 1;

/// Parameters for proving sum operations along an axis.
#[derive(Clone)]
pub struct SumAxisParams<F: JoltField> {
    r_node_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
    sum_config: SumConfig,
}

impl<F: JoltField> SumAxisParams<F> {
    /// Create new parameters for sum operation along an axis.
    pub fn new(
        computation_node: ComputationNode,
        sum_config: SumConfig,
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
            sum_config,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for SumAxisParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, sum_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(self.computation_node.idx),
            SumcheckId::Execution,
        );
        sum_claim
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(challenges.to_vec())
    }

    fn num_rounds(&self) -> usize {
        self.sum_config.operand_dims()[self.sum_config.axis().axis_index()].log_2()
    }
}

/// Prover state for sum along axis sumcheck protocol.
pub struct SumAxisProver<F: JoltField> {
    params: SumAxisParams<F>,
    operand: MultilinearPolynomial<F>,
}

impl<F: JoltField> SumAxisProver<F> {
    /// Initialize the prover with trace data and parameters for sum along axis.
    #[tracing::instrument(skip_all, name = "SumAxisProver::initialize")]
    pub fn initialize(
        trace: &Trace,
        params: SumAxisParams<F>,
        _accumulator: &ProverOpeningAccumulator<F>,
    ) -> Self {
        let LayerData {
            operands,
            output: _,
        } = Trace::layer_data(trace, &params.computation_node);
        let [operand] = operands[..] else {
            panic!("Expected one operand for SumAxis operation")
        };

        let (m, n) = (
            params.sum_config.operand_dims()[0],
            params.sum_config.operand_dims()[1],
        );

        let operand: Vec<F> = match params.sum_config.axis() {
            SumAxis::Axis0 => {
                debug_assert_eq!(n.log_2(), params.r_node_output.len());
                let eq_r_node_output = EqPolynomial::evals(&params.r_node_output);
                (0..m)
                    .into_par_iter()
                    .map(|h| {
                        (0..n)
                            .map(|j| F::from_i32(operand[h * n + j]) * eq_r_node_output[j])
                            .sum::<F>()
                    })
                    .collect()
            }
            SumAxis::Axis1 => {
                debug_assert_eq!(m.log_2(), params.r_node_output.len());
                let eq_r_node_output = EqPolynomial::evals(&params.r_node_output);
                (0..n)
                    .into_par_iter()
                    .map(|j| {
                        (0..m)
                            .map(|h| F::from_i32(operand[h * n + j]) * eq_r_node_output[h])
                            .sum::<F>()
                    })
                    .collect()
            }
        };

        let operand = MultilinearPolynomial::from(operand);
        #[cfg(test)]
        {
            let claim = (0..operand.len())
                .into_par_iter()
                .map(|i| operand.get_bound_coeff(i))
                .sum::<F>();
            assert_eq!(claim, params.input_claim(_accumulator));
        }
        Self { params, operand }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for SumAxisProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let half_poly_len = self.operand.len() / 2;
        let eval_0 = (0..half_poly_len)
            .into_par_iter()
            .map(|i| self.operand.get_bound_coeff(i))
            .sum();
        UniPoly::from_evals_and_hint(previous_claim, &[eval_0])
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.operand.bind_parallel(r_j, BindingOrder::HighToLow);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = match self.params.sum_config.axis() {
            SumAxis::Axis0 => [sumcheck_challenges, &self.params.r_node_output].concat(),
            SumAxis::Axis1 => [&self.params.r_node_output, sumcheck_challenges].concat(),
        };
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            opening_point.into(),
            self.operand.final_sumcheck_claim(),
        );
        accumulator.cache_virtual_operand_claims(transcript, &self.params.computation_node);
    }
}

/// Verifier for sum along axis sumcheck protocol.
pub struct SumAxisVerifier<F: JoltField> {
    params: SumAxisParams<F>,
}

impl<F: JoltField> SumAxisVerifier<F> {
    /// Create new verifier for sum along axis.
    pub fn new(
        computation_node: ComputationNode,
        sum_config: SumConfig,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = SumAxisParams::new(computation_node, sum_config, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for SumAxisVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        accumulator.get_operand_claims::<1>(self.params.computation_node.idx)[0]
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = match self.params.sum_config.axis() {
            SumAxis::Axis0 => [sumcheck_challenges, &self.params.r_node_output].concat(),
            SumAxis::Axis1 => [&self.params.r_node_output, sumcheck_challenges].concat(),
        };
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            opening_point.into(),
        );
        accumulator.append_operand_claims(transcript, self.params.computation_node.idx);
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

    fn sum_model<const AXIS: usize>(m: usize, n: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![m, n]);
        let output_dims = if AXIS == 0 { vec![n] } else { vec![m] };
        let res = b.sum(i, vec![AXIS], output_dims);
        b.mark_output(res);
        b.build()
    }

    fn test_sum_axis_generic(log_m: usize, log_n: usize, seed: u64, axis: usize) {
        let m = 1 << log_m;
        let n = 1 << log_n;
        let mut rng = StdRng::seed_from_u64(seed);
        let input = Tensor::<i32>::random_small(&mut rng, &[m, n]);
        let model = match axis {
            0 => sum_model::<0>(m, n),
            1 => sum_model::<1>(m, n),
            _ => panic!("Invalid axis"),
        };
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_sum_axis_0_1d() {
        test_sum_axis_generic(5, 0, 0x123, 0);
    }

    #[test]
    fn test_sum_axis_0_2d() {
        test_sum_axis_generic(4, 5, 0x398, 0);
    }

    #[test]
    fn test_sum_axis_1_2d() {
        test_sum_axis_generic(4, 5, 0x878, 1);
    }

    #[test]
    fn test_sum_axis_1_1d() {
        test_sum_axis_generic(4, 0, 0x844, 1);
    }
}
