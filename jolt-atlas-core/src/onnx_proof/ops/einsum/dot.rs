//! Shared sumcheck engine for einsum *contraction* patterns.
//!
//! Every einsum we prove reduces to the **same** sumcheck. After the operands are
//! "folded" at the output randomness, the claim is
//!
//! ```text
//! claim = Σ_h  L(h) · R(h) · EQ(h)
//! ```
//!
//! over the contraction(+batch) hypercube `h`. The patterns differ in only three
//! places, captured by the [`EinsumLayout`] trait:
//!
//! 1. **fold** — building `L`, `R`, and the optional batch `EQ` from the operand
//!    tensors and the node-output point ([`EinsumLayout::fold`]).
//! 2. **eq schedule** — where the `EQ` cube sits in the round order, or whether it
//!    is absent (a plain degree-2 dot product). See [`EqSchedule`].
//! 3. **scatter** — mapping the final sumcheck point back to the two operand
//!    opening points ([`EinsumLayout::operand_points`]).
//!
//! Everything else — the `Params` boilerplate, the multiply-accumulate message
//! loop, challenge ingestion, the BlindFold (`zk`) constraints — is written once,
//! here. This is the generalization of the per-variant engine that
//! [`rbmk_rbnk_bmn`](super::rbmk_rbnk_bmn) already used for three patterns.
//!
//! ## Where the claim comes from
//!
//! The sumcheck's initial claim is `input_claim` (today: the node-output opening).
//! Fusing the rescaling division + clamp into the einsum (issue #190) only changes
//! *that* value — `rescaled(r)·2^S + R(r)` instead of `output(r)` — while the
//! `Σ L·R` body stays identical. That seam is centralized in
//! [`EinsumDotParams::input_claim`].

use std::{array, sync::Arc};

use crate::{
    onnx_proof::fused_rebase,
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    tensor::Tensor,
};
use common::parallel::par_enabled;
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
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
};
use rayon::prelude::*;

#[cfg(feature = "zk")]
use joltworks::subprotocols::blindfold::{
    InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource,
};

/// The eq-poly binding schedule within the (HighToLow) sumcheck round order.
///
/// The contraction cube is bound high-to-low. The batch `EQ` cube, when present,
/// occupies a contiguous range of rounds; the two layouts in use differ in
/// whether that range is at the top or the bottom of the round order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EqSchedule {
    /// No `EQ` factor: a plain degree-2 dot product over the contraction cube.
    None,
    /// `EQ` cube occupies the **first** `log_eq` rounds (high bits) and is bound
    /// first; afterwards the cached scalar is reused.
    High {
        /// Number of rounds the `EQ` cube spans (its bit-width).
        log_eq: usize,
        /// Number of contraction bits below the `EQ` cube, used to index `EQ`.
        low_bits: usize,
    },
    /// `EQ` cube occupies the **last** `log_b` rounds (low bits) and is bound
    /// last; during the first `log_k` rounds `EQ` is read by coefficient.
    Low {
        /// Number of contraction rounds above the `EQ` cube (bound before `EQ`).
        log_k: usize,
        /// Number of rounds the `EQ` cube spans (its bit-width).
        log_b: usize,
    },
}

impl EqSchedule {
    /// Whether this schedule carries an `EQ` factor (degree 3 vs degree 2).
    pub fn has_eq(&self) -> bool {
        !matches!(self, EqSchedule::None)
    }

    /// The sumcheck degree implied by this schedule.
    pub fn degree(&self) -> usize {
        if self.has_eq() {
            3
        } else {
            2
        }
    }
}

/// Operands folded at the output randomness into MLEs over the sumcheck cube.
pub struct Folded<F: JoltField> {
    /// Left operand folded over the contraction(+batch) cube.
    pub left: MultilinearPolynomial<F>,
    /// Right operand folded over the contraction(+batch) cube.
    pub right: MultilinearPolynomial<F>,
    /// Batch eq-poly, present iff the schedule is not [`EqSchedule::None`].
    pub eq: Option<MultilinearPolynomial<F>>,
}

/// Per-pattern description of an einsum contraction sumcheck.
///
/// See the [module docs](self) for how the engine uses these hooks.
pub trait EinsumLayout<F: JoltField>: Send + Sync {
    /// Number of sumcheck rounds (`log` of the contraction(+batch) cube size).
    fn num_rounds(&self) -> usize;

    /// The eq-poly binding schedule for this pattern.
    fn schedule(&self) -> EqSchedule;

    /// Fold the two operand tensors at `r_node_output` into the sumcheck MLEs.
    fn fold(
        &self,
        left_operand: &Tensor<i32>,
        right_operand: &Tensor<i32>,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
    ) -> Folded<F>;

    /// Map the final sumcheck point back to the `(left, right)` operand opening
    /// points. `sumcheck_challenges` is the post-protocol challenge vector.
    fn operand_points(
        &self,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
        sumcheck_challenges: &[F],
    ) -> (Vec<F>, Vec<F>);

    /// The `EQ` factor multiplying `left·right` in the output claim
    /// (`F::one()` when the schedule carries no `EQ`).
    fn output_eq(
        &self,
        r_node_output: &OpeningPoint<BIG_ENDIAN, F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F;
}

/// Shared parameters for any einsum contraction sumcheck.
#[derive(Clone)]
pub struct EinsumDotParams<F: JoltField> {
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
    layout: Arc<dyn EinsumLayout<F>>,
}

impl<F: JoltField> EinsumDotParams<F> {
    /// Create params for the given pattern `layout`.
    pub fn new(
        computation_node: ComputationNode,
        layout: Arc<dyn EinsumLayout<F>>,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);
        let r_node_output = accessor.get_reduced_opening().0;
        Self {
            r_node_output,
            computation_node,
            layout,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for EinsumDotParams<F> {
    fn degree(&self) -> usize {
        self.layout.schedule().degree()
    }

    /// The sumcheck's initial claim: the raw accumulation `acc(r) = Σ_k L·R`.
    ///
    /// The einsum fuses an i64 accumulate + rescaling division + saturating
    /// clamp; [`fused_rebase::fused_input_claim`] recovers `acc(r)` from the
    /// pre-clamp `rescaled` and remainder advice .
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        fused_rebase::fused_input_claim(accumulator, &self.computation_node)
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(challenges.to_vec())
    }

    fn num_rounds(&self) -> usize {
        self.layout.num_rounds()
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::default()
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(
        &self,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        Vec::new()
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        use crate::utils::opening_access::OpeningIdBuilder;
        let builder = OpeningIdBuilder::new(&self.computation_node);
        let left = ValueSource::Opening(builder.nodeio(Target::Input(0)));
        let right = ValueSource::Opening(builder.nodeio(Target::Input(1)));
        // With an EQ factor the claim is scaled by the (eq) challenge value;
        // a plain dot product is just the operand product.
        let term = if self.layout.schedule().has_eq() {
            ProductTerm::scaled(ValueSource::Challenge(0), vec![left, right])
        } else {
            ProductTerm::product(vec![left, right])
        };
        Some(OutputClaimConstraint::sum_of_products(vec![term]))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        if self.layout.schedule().has_eq() {
            vec![self
                .layout
                .output_eq(&self.r_node_output, sumcheck_challenges)]
        } else {
            vec![]
        }
    }
}

/// Prover for any einsum contraction sumcheck.
pub struct EinsumDotProver<F: JoltField> {
    params: EinsumDotParams<F>,
    left: MultilinearPolynomial<F>,
    right: MultilinearPolynomial<F>,
    eq: Option<MultilinearPolynomial<F>>,
    /// `EQ` final claim, cached once the `EQ` cube is fully bound (High schedule).
    eq_bound_claim: Option<F>,
}

impl<F: JoltField> EinsumDotProver<F> {
    /// Initialize the prover by folding the operands at the output randomness.
    #[tracing::instrument(skip_all, name = "EinsumDotProver::initialize")]
    pub fn initialize(trace: &Trace, params: EinsumDotParams<F>) -> Self {
        let LayerData { operands, .. } = Trace::layer_data(trace, &params.computation_node);
        let [left_operand, right_operand] = operands[..] else {
            panic!("Expected two operands for einsum operation")
        };
        let Folded { left, right, eq } =
            params
                .layout
                .fold(left_operand, right_operand, &params.r_node_output);

        Self {
            params,
            left,
            right,
            eq,
            eq_bound_claim: None,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for EinsumDotProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let half = self.left.len() / 2;
        match self.params.layout.schedule() {
            EqSchedule::None => {
                let evals: [F; 2] = (0..half)
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|i| {
                        let l = self.left.sumcheck_evals(i, 2, BindingOrder::HighToLow);
                        let r = self.right.sumcheck_evals(i, 2, BindingOrder::HighToLow);
                        [l[0] * r[0], l[1] * r[1]]
                    })
                    .reduce(|| [F::zero(); 2], |a, b| array::from_fn(|i| a[i] + b[i]));
                UniPoly::from_evals_and_hint(previous_claim, &evals)
            }
            EqSchedule::High { log_eq, low_bits } => {
                let eq = self.eq.as_ref().expect("eq poly must exist");
                let evals: [F; 3] = (0..half)
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|i| {
                        let l = self
                            .left
                            .sumcheck_evals_array::<3>(i, BindingOrder::HighToLow);
                        let r = self
                            .right
                            .sumcheck_evals_array::<3>(i, BindingOrder::HighToLow);
                        let e = if round < log_eq {
                            eq.sumcheck_evals_array::<3>(i >> low_bits, BindingOrder::HighToLow)
                        } else {
                            [self.eq_bound_claim.expect("eq claim should be cached"); 3]
                        };
                        [l[0] * r[0] * e[0], l[1] * r[1] * e[1], l[2] * r[2] * e[2]]
                    })
                    .reduce(|| [F::zero(); 3], |a, b| array::from_fn(|i| a[i] + b[i]));
                UniPoly::from_evals_and_hint(previous_claim, &evals)
            }
            EqSchedule::Low { log_k, log_b } => {
                let eq = self.eq.as_ref().expect("eq poly must exist");
                let evals: [F; 3] = (0..half)
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|jh| {
                        let l = self
                            .left
                            .sumcheck_evals_array::<3>(jh, BindingOrder::HighToLow);
                        let r = self
                            .right
                            .sumcheck_evals_array::<3>(jh, BindingOrder::HighToLow);
                        let e = if round < log_k {
                            [eq.get_bound_coeff(jh & ((1 << log_b) - 1)); 3]
                        } else {
                            eq.sumcheck_evals_array::<3>(jh, BindingOrder::HighToLow)
                        };
                        [l[0] * r[0] * e[0], l[1] * r[1] * e[1], l[2] * r[2] * e[2]]
                    })
                    .reduce(|| [F::zero(); 3], |a, b| array::from_fn(|i| a[i] + b[i]));
                UniPoly::from_evals_and_hint(previous_claim, &evals)
            }
        }
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.left.bind_parallel(r_j, BindingOrder::HighToLow);
        self.right.bind_parallel(r_j, BindingOrder::HighToLow);
        match self.params.layout.schedule() {
            EqSchedule::None => {}
            EqSchedule::High { log_eq, .. } => {
                if round < log_eq {
                    let eq = self.eq.as_mut().expect("eq poly must exist");
                    eq.bind_parallel(r_j, BindingOrder::HighToLow);
                    if round == log_eq - 1 {
                        self.eq_bound_claim = Some(eq.final_claim());
                    }
                }
            }
            EqSchedule::Low { log_k, .. } => {
                if round >= log_k {
                    self.eq
                        .as_mut()
                        .expect("eq poly must exist")
                        .bind_parallel(r_j, BindingOrder::HighToLow);
                }
            }
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let sumcheck_challenges = sumcheck_challenges.into_opening();
        let (left_point, right_point) = self
            .params
            .layout
            .operand_points(&self.params.r_node_output, &sumcheck_challenges);
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .into_provider(transcript, Default::default());
        provider.append_nodeio_at(
            Target::Input(0),
            self.params.normalize_opening_point(&left_point),
            self.left.final_claim(),
        );
        provider.append_nodeio_at(
            Target::Input(1),
            self.params.normalize_opening_point(&right_point),
            self.right.final_claim(),
        );
    }
}

/// Verifier for any einsum contraction sumcheck.
pub struct EinsumDotVerifier<F: JoltField> {
    params: EinsumDotParams<F>,
}

impl<F: JoltField> EinsumDotVerifier<F> {
    /// Create a verifier for the given pattern `layout`.
    pub fn new(
        computation_node: ComputationNode,
        layout: Arc<dyn EinsumLayout<F>>,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = EinsumDotParams::new(computation_node, layout, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for EinsumDotVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.params.computation_node);
        let left = accessor.get_nodeio(Target::Input(0)).1;
        let right = accessor.get_nodeio(Target::Input(1)).1;
        let eq = self
            .params
            .layout
            .output_eq(&self.params.r_node_output, sumcheck_challenges);
        left * right * eq
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let sumcheck_challenges = sumcheck_challenges.into_opening();
        let (left_point, right_point) = self
            .params
            .layout
            .operand_points(&self.params.r_node_output, &sumcheck_challenges);
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .into_provider(transcript, Default::default());
        provider.append_nodeio_at(
            Target::Input(0),
            self.params.normalize_opening_point(&left_point),
        );
        provider.append_nodeio_at(
            Target::Input(1),
            self.params.normalize_opening_point(&right_point),
        );
    }
}
