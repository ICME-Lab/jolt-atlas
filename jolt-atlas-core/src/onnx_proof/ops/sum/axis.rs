use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
};
use common::parallel::par_enabled;
use common::VirtualPolynomial;
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, VirtualOpeningId, BIG_ENDIAN,
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
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
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
            .get_node_output_opening(computation_node.idx)
            .0
            .r;
        Self {
            r_node_output: r_node_output.into(),
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
        let (_, sum_claim) = accumulator.get_node_output_opening(self.computation_node.idx);
        sum_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
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
                debug_assert_eq!(n.log_2(), params.r_node_output.r.len());
                let eq_r_node_output = EqPolynomial::evals(&params.r_node_output.r);
                (0..m)
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|h| {
                        (0..n)
                            .map(|j| F::from_i32(operand[h * n + j]) * eq_r_node_output[j])
                            .sum::<F>()
                    })
                    .collect()
            }
            SumAxis::Axis1 => {
                debug_assert_eq!(m.log_2(), params.r_node_output.r.len());
                let eq_r_node_output = EqPolynomial::evals(&params.r_node_output.r);
                (0..n)
                    .into_par_iter()
                    .with_min_len(par_enabled())
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
                .with_min_len(par_enabled())
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
            .with_min_len(par_enabled())
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
        let sumcheck_challenges = sumcheck_challenges.into_opening();
        let opening_point = match self.params.sum_config.axis() {
            SumAxis::Axis0 => [
                sumcheck_challenges.as_slice(),
                self.params.r_node_output.r.as_slice(),
            ]
            .concat(),
            SumAxis::Axis1 => [
                self.params.r_node_output.r.as_slice(),
                sumcheck_challenges.as_slice(),
            ]
            .concat(),
        };
        let operand_id = VirtualOpeningId::new(
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        accumulator.append_virtual(
            transcript,
            operand_id,
            OpeningPoint::new(opening_point),
            self.operand.final_sumcheck_claim(),
        );
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
        accumulator.get_node_output_claim(
            self.params.computation_node.inputs[0],
            self.params.computation_node.idx,
        )
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let sumcheck_challenges = sumcheck_challenges.into_opening();
        let opening_point = match self.params.sum_config.axis() {
            SumAxis::Axis0 => [
                sumcheck_challenges.as_slice(),
                self.params.r_node_output.r.as_slice(),
            ]
            .concat(),
            SumAxis::Axis1 => [
                self.params.r_node_output.r.as_slice(),
                sumcheck_challenges.as_slice(),
            ]
            .concat(),
        };
        let operand_id = VirtualOpeningId::new(
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        accumulator.append_virtual(
            transcript,
            operand_id,
            OpeningPoint::new(opening_point),
        );
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

    #[test]
    #[ignore = "TODO: non-power-of-two sum(axis) path not fully validated yet"]
    fn test_sum_axis_non_power_of_two_input_len() {
        let mut rng = StdRng::seed_from_u64(0x845);
        let m = 33;
        let n = 65;
        let input = Tensor::<i32>::random_small(&mut rng, &[m, n]);
        let model = sum_model::<1>(m, n);
        unit_test_op(model, &[input]);
    }
}
