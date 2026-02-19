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
    use ark_bn254::Fr;
    use atlas_onnx_tracer::{
        model::{
            self,
            trace::{LayerData, Trace},
        },
        ops::{Operator, Sum},
        tensor::Tensor,
    };
    use common::VirtualPolynomial;
    use joltworks::{
        field::JoltField,
        poly::{
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            opening_proof::{
                OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
                VerifierOpeningAccumulator, BIG_ENDIAN,
            },
        },
        subprotocols::sumcheck::Sumcheck,
        transcripts::{Blake2bTranscript, Transcript},
    };
    use rand::{rngs::StdRng, SeedableRng};

    use crate::{
        onnx_proof::ops::sum::axis::{SumAxis, SumAxisParams, SumAxisProver, SumAxisVerifier},
        utils::dims::{SumConfig, SumDims},
    };

    pub fn test_sum_axis_generic(log_m: usize, log_n: usize, seed: u64, axis: usize) {
        let m = 1 << log_m;
        let n = 1 << log_n;
        let mut rng = StdRng::seed_from_u64(seed);
        let input = Tensor::<i32>::random_small(&mut rng, &[m, n]);

        let model = match axis {
            0 => model::test::sum_model::<0>(m, n),
            1 => model::test::sum_model::<1>(m, n),
            _ => panic!("Invalid axis"),
        };
        let trace = model.trace(&[input]);

        let prover_transcript = &mut Blake2bTranscript::new(&[]);
        let mut prover_opening_accumulator: ProverOpeningAccumulator<Fr> =
            ProverOpeningAccumulator::new();

        let output_log_size = match axis {
            0 => log_n,
            1 => log_m,
            _ => panic!("Invalid axis"),
        };
        let r_node_output: Vec<<Fr as JoltField>::Challenge> =
            prover_transcript.challenge_vector_optimized::<Fr>(output_log_size);

        let output_index = model.outputs()[0];
        let computation_node = &model[output_index];
        let LayerData {
            operands: _,
            output,
        } = Trace::layer_data(&trace, computation_node);

        let sum_claim = MultilinearPolynomial::from(output.clone()).evaluate(&r_node_output);
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.clone().into(),
            sum_claim,
        );

        let sum_config = match &computation_node.operator {
            Operator::Sum(Sum { axes }) => {
                assert_eq!(axes, &[axis]);
                let operand_dims = model[computation_node.inputs[0]].output_dims.clone();
                let output_dims = computation_node.output_dims.clone();
                let sum_dims = SumDims::new(operand_dims, output_dims);
                let axis_enum = match axis {
                    0 => SumAxis::Axis0,
                    1 => SumAxis::Axis1,
                    _ => panic!("Invalid axis"),
                };
                SumConfig::new(sum_dims, axis_enum)
            }
            _ => panic!("Unexpected operator"),
        };

        let params: SumAxisParams<Fr> = SumAxisParams::new(
            computation_node.clone(),
            sum_config.clone(),
            &prover_opening_accumulator,
        );
        let mut prover_sumcheck =
            SumAxisProver::initialize(&trace, params, &prover_opening_accumulator);

        let (proof, r_sumcheck) = Sumcheck::prove(
            &mut prover_sumcheck,
            &mut prover_opening_accumulator,
            prover_transcript,
        );

        let verifier_transcript = &mut Blake2bTranscript::new(&[]);
        verifier_transcript.compare_to(prover_transcript.clone());

        let mut verifier_opening_accumulator: VerifierOpeningAccumulator<Fr> =
            VerifierOpeningAccumulator::new();
        let _r_node_output: Vec<<Fr as JoltField>::Challenge> =
            verifier_transcript.challenge_vector_optimized::<Fr>(output_log_size);
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

        let verifier_sumcheck = SumAxisVerifier::new(
            computation_node.clone(),
            sum_config.clone(),
            &verifier_opening_accumulator,
        );
        let res = Sumcheck::verify(
            &proof,
            &verifier_sumcheck,
            &mut verifier_opening_accumulator,
            verifier_transcript,
        );
        let r_sumcheck_verif = res.unwrap();
        assert_eq!(r_sumcheck, r_sumcheck_verif);

        // Evaluate input at operand point and check it equals the expected output claim
        let input_index = computation_node.inputs[0];
        let input_layer = &model[input_index];
        let input_data = Trace::layer_data(&trace, input_layer).output.clone();
        let input_poly = MultilinearPolynomial::from(input_data);
        let (opening_point, expected_output_claim) = verifier_opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(input_index),
                SumcheckId::Execution,
            );
        let input_eval = input_poly.evaluate(&opening_point.r);
        assert_eq!(input_eval, expected_output_claim);
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
