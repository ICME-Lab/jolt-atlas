use crate::onnx_proof::trace_view::TraceView;

use atlas_trace::{trace::LayerData, Trace};

use crate::node_view::ComputationNode;
use crate::utils::{
    dims::{SumAxis, SumConfig},
    opening_access::{AccOpeningAccessor, Target},
};
use common::{parallel::par_enabled, VirtualPoly};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::math::Math,
};
use rayon::prelude::*;

const DEGREE_BOUND: usize = 1;

/// Parameters for proving sum operations along an axis.
#[derive(Clone)]
pub struct SumAxisParams<F: JoltField> {
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
    sum_config: SumConfig,
}

impl<F: JoltField> SumAxisParams<F> {
    /// Create parameters whose input claim is the pre-clamp accumulation
    /// ([`VirtualPoly::ClampAcc`]); the clamp `output = SatClamp(acc)` is proven
    /// separately by the clamp lookup.
    pub fn new_clamped(
        computation_node: ComputationNode,
        sum_config: SumConfig,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);
        let r_node_output = accessor.get_reduced_opening().0;
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
        // The sumcheck reduces the pre-clamp accumulation.
        let acc_id = OpeningId::new(
            VirtualPoly::ClampAcc(self.computation_node.idx),
            SumcheckId::NodeExecution(self.computation_node.idx),
        );
        accumulator.get_virtual_polynomial_opening(acc_id).1
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
        } = TraceView::layer_data(trace, &params.computation_node);
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
            crate::onnx_proof::testing::probe::probe_assert_eq!(
                claim,
                params.input_claim(_accumulator)
            );
        }
        Self { params, operand }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for SumAxisProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
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
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .into_provider(transcript, OpeningPoint::new(opening_point));
        provider.append_nodeio(Target::Input(0), self.operand.final_claim());
    }
}

impl<F: JoltField> SumAxisProver<F> {
    fn round_message(&mut self, _round: usize, previous_claim: F) -> UnivariatePoly<F> {
        let half_poly_len = self.operand.len() / 2;
        let eval_0 = (0..half_poly_len)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|i| self.operand.get_bound_coeff(i))
            .sum();
        UnivariatePoly::from_evals_and_hint(previous_claim, &[eval_0])
    }

    fn bind(&mut self, r: F, _round: usize) {
        let r_j = F::challenge_from_field(r);
        self.operand.bind_parallel(r_j, BindingOrder::HighToLow);
    }
}

impl<F: JoltField> ProveRounds<F> for SumAxisProver<F> {
    fn num_rounds(&self) -> usize {
        SumcheckInstanceParams::num_rounds(&self.params)
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r, round - 1);
        }
        Ok(self.round_message(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind, SumcheckInstanceParams::num_rounds(&self.params) - 1);
        Ok(())
    }
}

/// Verifier for sum along axis sumcheck protocol.
pub struct SumAxisVerifier<F: JoltField> {
    params: SumAxisParams<F>,
}

impl<F: JoltField> SumAxisVerifier<F> {
    /// Create a verifier whose input claim is the pre-clamp accumulation
    /// ([`VirtualPoly::ClampAcc`]); pairs with the clamp lookup.
    pub fn new_clamped(
        computation_node: ComputationNode,
        sum_config: SumConfig,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = SumAxisParams::new_clamped(computation_node, sum_config, accumulator);
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
        let accessor = AccOpeningAccessor::new(accumulator, &self.params.computation_node);
        accessor.get_nodeio(Target::Input(0)).1
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
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .into_provider(transcript, OpeningPoint::new(opening_point));
        provider.append_nodeio(Target::Input(0));
    }
}
