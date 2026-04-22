//! Complementary slackness constraint for softmax saturation.
//!
//! Proves `∀ (k,j): sat_diff[k,j] * (z_bound − 1 − z_c[k,j]) = 0`
//! via a degree-3 sumcheck:
//!   0 = Σ_{x ∈ {0,1}^n} eq(r2, x) * sat_diff(x) * (z_bound−1 − z_hi(x)*B − z_lo(x))
//!
//! This ensures a unique decomposition `z = z_c + sat_diff` where
//! `z_c = min(z, z_bound − 1)`: either sat_diff = 0 (unsaturated) or
//! z_c = z_bound − 1 (saturated).

use crate::utils::opening_access::AccOpeningAccessor;
use atlas_onnx_tracer::node::ComputationNode;
use common::VirtualPoly;
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN, LITTLE_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
};

const DEGREE_BOUND: usize = 3;

// ---------------------------------------------------------------------------
// Shared parameters
// ---------------------------------------------------------------------------

/// Shared prover / verifier parameters for the complementary-slackness sumcheck.
#[derive(Clone)]
pub struct SatDiffSlacknessParams<F: JoltField> {
    /// Computation node reference.
    node: ComputationNode,
    /// The eq-binding point r₁ (from the Stage 1 exp-sum challenge).
    r1: Vec<F>,
    /// `z_bound − 1` as a field element (= K_hi * B − 1).
    z_bound_minus_1: F,
    /// Base `B` as a field element.
    base: F,
    /// Number of sumcheck variables = log_2(F * N).
    num_vars: usize,
}

impl<F: JoltField> SatDiffSlacknessParams<F> {
    /// Create new CS params, reading r₁ from the accumulator.
    pub fn new(
        node: ComputationNode,
        z_bound_minus_1: u64,
        base: u64,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let r1 = AccOpeningAccessor::new(accumulator, &node)
            .get_advice(VirtualPoly::SoftmaxExpQ)
            .0
            .r;
        let num_vars = r1.len();
        Self {
            r1,
            node,
            z_bound_minus_1: F::from_u64(z_bound_minus_1),
            base: F::from_u64(base),
            num_vars,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for SatDiffSlacknessParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Prover for the sat-diff slackness sumcheck.
pub struct SatDiffSlacknessProver<F: JoltField> {
    params: SatDiffSlacknessParams<F>,
    /// eq(r_1, *) materialized.
    eq_r1: GruenSplitEqPolynomial<F>,
    /// sat_diff MLE.
    sat_diff: MultilinearPolynomial<F>,
    /// z_hi MLE (high digit of clamped logit).
    z_hi: MultilinearPolynomial<F>,
    /// z_lo MLE (low digit of clamped logit).
    z_lo: MultilinearPolynomial<F>,
}

impl<F: JoltField> SatDiffSlacknessProver<F> {
    /// Build the prover from witness slices and shared params.
    pub fn initialize(
        sat_diff: &[i32],
        z_hi: &[i32],
        z_lo: &[i32],
        params: SatDiffSlacknessParams<F>,
    ) -> Self {
        let eq_r1 = GruenSplitEqPolynomial::new(&params.r1, BindingOrder::LowToHigh);

        Self {
            sat_diff: MultilinearPolynomial::from(sat_diff.to_vec()),
            z_hi: MultilinearPolynomial::from(z_hi.to_vec()),
            z_lo: MultilinearPolynomial::from(z_lo.to_vec()),
            params,
            eq_r1,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for SatDiffSlacknessProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(name = "SatDiffSlacknessProver::compute_message", skip_all)]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let z_bound_minus_1 = self.params.z_bound_minus_1;
        let base = self.params.base;
        let eq = &self.eq_r1;

        let [q_constant, q_quadratic] = eq.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let sd_0 = self.sat_diff.get_bound_coeff(2 * g);
            let sd_inf = self.sat_diff.get_bound_coeff(2 * g + 1) - sd_0;

            let z_hi_0 = self.z_hi.get_bound_coeff(2 * g);
            let z_hi_inf = self.z_hi.get_bound_coeff(2 * g + 1) - z_hi_0;

            let z_lo_0 = self.z_lo.get_bound_coeff(2 * g);
            let z_lo_inf = self.z_lo.get_bound_coeff(2 * g + 1) - z_lo_0;

            let c0 = sd_0 * (z_bound_minus_1 - z_hi_0 * base - z_lo_0);
            let e = sd_inf * (-base * z_hi_inf - z_lo_inf);
            [c0, e]
        });
        eq.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r1.bind(r_j);
        self.sat_diff.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.z_hi.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.z_lo.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.node)
            .into_provider(transcript, opening_point);
        provider.append_advice(VirtualPoly::SoftmaxSatDiff, self.sat_diff.final_claim());
        provider.append_advice(VirtualPoly::SoftmaxZHi, self.z_hi.final_claim());
        provider.append_advice(VirtualPoly::SoftmaxZLo, self.z_lo.final_claim());
    }
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Verifier for the complementary-slackness sumcheck.
pub struct SatDiffSlacknessVerifier<F: JoltField> {
    params: SatDiffSlacknessParams<F>,
}

impl<F: JoltField> SatDiffSlacknessVerifier<F> {
    /// Create a new SatDiff verifier.
    pub fn new(
        node: ComputationNode,
        z_bound_minus_1: u64,
        base: u64,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = SatDiffSlacknessParams::new(node, z_bound_minus_1, base, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for SatDiffSlacknessVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.node)
            .into_provider(transcript, opening_point);
        provider.append_advice(VirtualPoly::SoftmaxSatDiff);
        provider.append_advice(VirtualPoly::SoftmaxZHi);
        provider.append_advice(VirtualPoly::SoftmaxZLo);
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r2: Vec<F> = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;

        // Read the prover's claimed evaluations at r₂.
        let accessor = AccOpeningAccessor::new(accumulator, &self.params.node);
        let sat_diff_claim = accessor.get_advice(VirtualPoly::SoftmaxSatDiff).1;
        let z_hi_claim = accessor.get_advice(VirtualPoly::SoftmaxZHi).1;
        let z_lo_claim = accessor.get_advice(VirtualPoly::SoftmaxZLo).1;

        // complement(r_2) = z_bound − 1 − z_hi(r_2) * B − z_lo(r_2)
        let complement = self.params.z_bound_minus_1 - z_hi_claim * self.params.base - z_lo_claim;

        // eq(r_1, r_2) * sat_diff(r_2) * complement(r_2)
        EqPolynomial::mle(&self.params.r1, &r2) * sat_diff_claim * complement
    }
}
