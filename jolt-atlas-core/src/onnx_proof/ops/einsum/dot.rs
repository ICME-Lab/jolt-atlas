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
//! Everything else — the multiply-accumulate message loop, challenge ingestion —
//! is written once, here.
//!
//! ## Where the claim comes from
//!
//! The claim algebra is the einsum's
//! [`ContractionRelation`](super::ContractionRelation): its input claim is the
//! fused rescale's `rescaled(r)·2^S + R(r)` (or the plain output opening), its
//! output claim `L·R·EQ` at the scattered operand points. This engine is the
//! relation's specialized prover; the verifier is the relation's.

use crate::onnx_proof::trace_view::TraceView;
use std::array;

use atlas_trace::{trace::LayerData, Tensor, Trace};

use super::ContractionRelation;
use crate::onnx_proof::relation::{
    append_produced, note_dispatch, relation_input_claim, AtlasRelation, RelationParams,
    RelationSetup,
};
use common::parallel::par_enabled;
use jolt_claims::SymbolicSumcheck;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{OpeningPoint, ProverOpeningAccumulator, BIG_ENDIAN},
    },
    subprotocols::sumcheck_prover::SumcheckInstanceProver,
    transcripts::Transcript,
};
use rayon::prelude::*;

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

/// The field-independent half of an einsum contraction pattern: its round geometry.
///
/// See the [module docs](self) for how the engine uses these hooks.
pub trait EinsumGeometry: Send + Sync {
    /// Number of sumcheck rounds (`log` of the contraction(+batch) cube size).
    fn num_rounds(&self) -> usize;

    /// The eq-poly binding schedule for this pattern.
    fn schedule(&self) -> EqSchedule;
}

/// The field-dependent half of an einsum contraction pattern.
pub trait EinsumLayout<F: JoltField>: EinsumGeometry {
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
        sumcheck_challenges: &[F],
    ) -> F;
}

/// The dot engine: the specialized prover of a
/// [`ContractionRelation`](super::ContractionRelation). Completeness-only: the verifier is the
/// relation's, and `relation_prover_equiv_contraction` checks these messages against the
/// relation's reference messages.
pub struct EinsumDotProver<F: JoltField> {
    relation: ContractionRelation,
    setup: RelationSetup<F>,
    schedule: EqSchedule,
    left: MultilinearPolynomial<F>,
    right: MultilinearPolynomial<F>,
    eq: Option<MultilinearPolynomial<F>>,
    /// `EQ` final claim, cached once the `EQ` cube is fully bound (High schedule).
    eq_bound_claim: Option<F>,
}

impl<F: JoltField> EinsumDotProver<F> {
    /// Initialize the prover by folding the operands at the output randomness.
    #[tracing::instrument(skip_all, name = "EinsumDotProver::initialize")]
    pub fn initialize(trace: &Trace, params: RelationParams<F, ContractionRelation>) -> Self {
        let (relation, setup) = params.into_parts();
        note_dispatch(&relation);
        let layout = relation.layout::<F>();
        let LayerData { operands, .. } = TraceView::layer_data(trace, relation.node());
        let [left_operand, right_operand] = operands[..] else {
            panic!("Expected two operands for einsum operation")
        };
        let r_node_output = OpeningPoint::new(setup.incoming_point.clone());
        let Folded { left, right, eq } = layout.fold(left_operand, right_operand, &r_node_output);
        let schedule = layout.schedule();
        let eq_bound_claim = match schedule {
            EqSchedule::High { log_eq: 0, .. } => {
                Some(eq.as_ref().expect("eq poly must exist").final_claim())
            }
            _ => None,
        };

        Self {
            relation,
            setup,
            schedule,
            left,
            right,
            eq,
            eq_bound_claim,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for EinsumDotProver<F> {
    fn degree(&self) -> usize {
        self.relation.degree()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        relation_input_claim(&self.relation, &self.setup, accumulator)
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        append_produced(
            &self.relation,
            &self.setup,
            accumulator,
            transcript,
            sumcheck_challenges,
            &[self.left.final_claim(), self.right.final_claim()],
        );
    }
}

impl<F: JoltField> EinsumDotProver<F> {
    fn round_message(&mut self, round: usize, previous_claim: F) -> UnivariatePoly<F> {
        let half = self.left.len() / 2;
        match self.schedule {
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
                UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
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
                UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
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
                UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
            }
        }
    }

    fn bind(&mut self, r: F, round: usize) {
        let r_j = F::challenge_from_field(r);
        self.left.bind_parallel(r_j, BindingOrder::HighToLow);
        self.right.bind_parallel(r_j, BindingOrder::HighToLow);
        match self.schedule {
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
}

impl<F: JoltField> ProveRounds<F> for EinsumDotProver<F> {
    fn num_rounds(&self) -> usize {
        self.relation.rounds()
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
        self.bind(bind, self.relation.rounds() - 1);
        Ok(())
    }
}
