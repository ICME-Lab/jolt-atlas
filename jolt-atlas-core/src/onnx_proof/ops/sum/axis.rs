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

use crate::utils::dims::SumAxis;

const DEGREE_BOUND: usize = 1;

// ---------------------------------------------------------------------------
// SumAxisProvider trait
// ---------------------------------------------------------------------------

/// Describes the polynomial configuration for a sum-axis sumcheck instance.
///
/// Implementors specify which [`VirtualPolynomial`] variants represent the
/// sum output and operand, enabling reuse of the same sumcheck logic across
/// standalone sum ops and composite ops like softmax.
///
/// This follows the same pattern as [`RaOneHotEncoding`] in `shout.rs`.
pub trait SumAxisProvider {
    /// The axis along which the sum is computed (0 or 1 after 2D normalization).
    fn axis(&self) -> SumAxis;

    /// Operand dimensions `[m, n]` of the 2D-normalized input.
    fn operand_dims(&self) -> [usize; 2];

    /// The `(VirtualPolynomial, SumcheckId)` whose opening provides the
    /// sum-output reduction point (`r_node_output`) and the input claim.
    ///
    /// For the standalone sum op this is `(NodeOutput(node_idx), NodeExecution(node_idx))`.
    /// For softmax stage 3 this is `(SoftmaxExpSum(node_idx), Execution)`.
    fn sum_output_source(&self) -> (VirtualPolynomial, SumcheckId);

    /// The `(VirtualPolynomial, SumcheckId)` for the operand polynomial,
    /// used in `cache_openings` and `expected_output_claim`.
    ///
    /// For the standalone sum op this is `(NodeOutput(input_idx), NodeExecution(node_idx))`.
    /// For softmax stage 3 this is `(SoftmaxExpQ(node_idx), Execution)`.
    fn operand_source(&self) -> (VirtualPolynomial, SumcheckId);
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Resolve a `(VirtualPolynomial, SumcheckId)` opening, routing `NodeOutput`
/// through `get_node_output_opening` (which scans consumer entries) instead of
/// requiring an exact key match.
fn resolve_vp_opening<F: JoltField>(
    accumulator: &dyn OpeningAccumulator<F>,
    vp: VirtualPolynomial,
    sid: SumcheckId,
) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
    if let VirtualPolynomial::NodeOutput(producer_idx) = vp {
        accumulator.get_node_output_opening(producer_idx)
    } else {
        accumulator.get_virtual_polynomial_opening(vp, sid)
    }
}

// ---------------------------------------------------------------------------
// SumAxisParams
// ---------------------------------------------------------------------------

/// Shared prover/verifier parameters for the sum-axis sumcheck, resolved from
/// a [`SumAxisProvider`].
#[derive(Clone)]
pub struct SumAxisParams<F: JoltField> {
    r_node_output: Vec<F::Challenge>,
    axis: SumAxis,
    operand_dims: [usize; 2],
    sum_output: (VirtualPolynomial, SumcheckId),
    operand: (VirtualPolynomial, SumcheckId),
}

impl<F: JoltField> SumAxisParams<F> {
    /// Build parameters from a [`SumAxisProvider`] and the current accumulator state.
    pub fn new(provider: &impl SumAxisProvider, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let axis = provider.axis();
        let operand_dims = provider.operand_dims();
        let sum_output = provider.sum_output_source();
        let operand = provider.operand_source();

        // Length of the non-summed dimension = length of r_node_output.
        let r_len = match axis {
            SumAxis::Axis0 => operand_dims[1].log_2(),
            SumAxis::Axis1 => operand_dims[0].log_2(),
        };

        let (point, _claim) = resolve_vp_opening(accumulator, sum_output.0, sum_output.1);
        let r_node_output = point.r[..r_len].to_vec();

        Self {
            r_node_output,
            axis,
            operand_dims,
            sum_output,
            operand,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for SumAxisParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        resolve_vp_opening(accumulator, self.sum_output.0, self.sum_output.1).1
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(challenges.to_vec())
    }

    fn num_rounds(&self) -> usize {
        self.operand_dims[self.axis.axis_index()].log_2()
    }
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Prover state for sum along axis sumcheck protocol.
pub struct SumAxisProver<F: JoltField> {
    params: SumAxisParams<F>,
    operand: MultilinearPolynomial<F>,
}

impl<F: JoltField> SumAxisProver<F> {
    /// Build the prover from raw operand values and resolved parameters.
    ///
    /// The operand is a flat `[m * n]` array in row-major order.
    /// The eq-polynomial reduction (absorbing `r_node_output`) is performed
    /// here, producing the univariate sumcheck polynomial.
    #[tracing::instrument(skip_all, name = "SumAxisProver::new")]
    pub fn new(
        operand_i32: &[i32],
        params: SumAxisParams<F>,
        #[allow(unused_variables)] accumulator: &ProverOpeningAccumulator<F>,
    ) -> Self {
        let (m, n) = (params.operand_dims[0], params.operand_dims[1]);

        let operand: Vec<F> = match params.axis {
            SumAxis::Axis0 => {
                debug_assert_eq!(n.log_2(), params.r_node_output.len());
                let eq_r_node_output = EqPolynomial::evals(&params.r_node_output);
                (0..m)
                    .into_par_iter()
                    .map(|h| {
                        (0..n)
                            .map(|j| F::from_i32(operand_i32[h * n + j]) * eq_r_node_output[j])
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
                            .map(|h| F::from_i32(operand_i32[h * n + j]) * eq_r_node_output[h])
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
            assert_eq!(claim, params.input_claim(accumulator));
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
        let opening_point = match self.params.axis {
            SumAxis::Axis0 => [sumcheck_challenges, &self.params.r_node_output].concat(),
            SumAxis::Axis1 => [&self.params.r_node_output, sumcheck_challenges].concat(),
        };
        let (vp, sid) = self.params.operand;
        accumulator.append_virtual(
            transcript,
            vp,
            sid,
            opening_point.into(),
            self.operand.final_sumcheck_claim(),
        );
    }
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Verifier for sum along axis sumcheck protocol.
pub struct SumAxisVerifier<F: JoltField> {
    params: SumAxisParams<F>,
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
        let (vp, sid) = self.params.operand;
        accumulator.get_virtual_polynomial_opening(vp, sid).1
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = match self.params.axis {
            SumAxis::Axis0 => [sumcheck_challenges, &self.params.r_node_output].concat(),
            SumAxis::Axis1 => [&self.params.r_node_output, sumcheck_challenges].concat(),
        };
        let (vp, sid) = self.params.operand;
        accumulator.append_virtual(transcript, vp, sid, opening_point.into());
    }
}

// ---------------------------------------------------------------------------
// Free functions (like ra_onehot_provers / ra_onehot_verifiers in shout.rs)
// ---------------------------------------------------------------------------

/// Build a sum-axis sumcheck **prover** from a [`SumAxisProvider`] and raw
/// operand values (flat `[m * n]` row-major `i32` slice).
pub fn sum_axis_prover<F: JoltField>(
    provider: &impl SumAxisProvider,
    operand: &[i32],
    accumulator: &ProverOpeningAccumulator<F>,
) -> SumAxisProver<F> {
    let params = SumAxisParams::new(provider, accumulator);
    SumAxisProver::new(operand, params, accumulator)
}

/// Build a sum-axis sumcheck **verifier** from a [`SumAxisProvider`].
pub fn sum_axis_verifier<F: JoltField>(
    provider: &impl SumAxisProvider,
    accumulator: &VerifierOpeningAccumulator<F>,
) -> SumAxisVerifier<F> {
    let params = SumAxisParams::new(provider, accumulator);
    SumAxisVerifier { params }
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
