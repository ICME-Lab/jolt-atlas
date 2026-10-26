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

use atlas_trace::Tensor;

use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{OpeningPoint, BIG_ENDIAN},
    },
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
