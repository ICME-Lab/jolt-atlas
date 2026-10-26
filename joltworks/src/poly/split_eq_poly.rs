//! Atlas extensions of upstream's split eq ([`jolt_poly::GruenSplitEqPolynomial`], the
//! Dao-Thaler + Gruen optimization, https://eprint.iacr.org/2024/1210.pdf): the delayed-reduction
//! fold our round kernels use, and the Gruen round messages with their endpoint closures.

use common::parallel::par_enabled;
use num_traits::Zero;
use rayon::prelude::*;

use crate::field::JoltField;

/// Test hook: when set, every Gruen round message also evaluates the endpoint it recovers by
/// division through the member's endpoint closure, and asserts the two agree. The closures run
/// for real only when the eq factor vanishes at an endpoint, which random points never reach; this
/// checks them on ordinary points. Set it only in a test that runs in its own process (nextest).
#[cfg(feature = "test-feature")]
pub static CHECK_GRUEN_ENDPOINTS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Under [`CHECK_GRUEN_ENDPOINTS`], assert `message(x) = l(x)·q_at(x)` at the endpoint `x`.
#[inline]
pub fn check_gruen_endpoint<F: JoltField>(
    eq: &jolt_poly::GruenSplitEqPolynomial<F>,
    message: &Result<jolt_poly::UnivariatePoly<F>, F>,
    at_one: bool,
    q_at: impl FnOnce() -> F,
) {
    #[cfg(feature = "test-feature")]
    if CHECK_GRUEN_ENDPOINTS.load(std::sync::atomic::Ordering::Relaxed) {
        if let Ok(message) = message {
            let (l_zero, l_one) = eq.current_linear_evals();
            let (x, l) = if at_one {
                (F::one(), l_one)
            } else {
                (F::zero(), l_zero)
            };
            assert_eq!(
                message.evaluate(x),
                l * q_at(),
                "Gruen endpoint closure disagrees with the division at {x}"
            );
        }
    }
    #[cfg(not(feature = "test-feature"))]
    let _ = (eq, message, at_one, q_at);
}

/// The delayed-reduction fold over upstream's split eq, and the Gruen round messages built on it.
///
/// Each `x_out` lane accumulates `e_in · value` unreduced, reduces once, scales by `e_out`
/// unreduced, and the lanes reduce once at the end.
pub trait SplitEqUnreducedFold<F: JoltField> {
    /// `Σ_g eq(g) · per_g_values(g)` over the unbound groups, reducing once per `x_out` and once
    /// at the end.
    fn par_fold_out_in_unreduced<const NUM_OUT: usize>(
        &self,
        per_g_values: &(impl Fn(usize) -> [F; NUM_OUT] + Sync + Send),
    ) -> [F; NUM_OUT];

    /// The cubic round message `l(X)·q(X)` of `Σ_g eq(g)·q_g(X)` with each `q_g` quadratic:
    /// `per_g(g) = [q_g(0), q_g's X² coefficient]`, and `at_one(g) = q_g(1)`, folded only when
    /// the eq factor vanishes at 1. The error carries the actual `s(0) + s(1)`.
    fn gruen_round_deg_3(
        &self,
        per_g: &(impl Fn(usize) -> [F; 2] + Sync + Send),
        at_one: &(impl Fn(usize) -> F + Sync + Send),
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, F> {
        let [q_constant, q_quadratic] = self.par_fold_out_in_unreduced::<2>(per_g);
        self.gruen_round_deg_3_from(q_constant, q_quadratic, at_one, previous_claim)
    }

    /// [`Self::gruen_round_deg_3`] from an already folded `q(0)` and X² coefficient.
    fn gruen_round_deg_3_from(
        &self,
        q_constant: F,
        q_quadratic: F,
        at_one: &(impl Fn(usize) -> F + Sync + Send),
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, F>;

    /// The quadratic round message `l(X)·q(X)` of `Σ_g eq(g)·q_g(X)` with each `q_g` linear:
    /// `at_one(g) = q_g(1)`, and `at_zero(g) = q_g(0)`, folded only when the eq factor vanishes
    /// at 0. The error carries the actual `s(0) + s(1)`.
    fn gruen_round_deg_2(
        &self,
        at_one: &(impl Fn(usize) -> F + Sync + Send),
        at_zero: &(impl Fn(usize) -> F + Sync + Send),
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, F> {
        self.gruen_round_deg_2_scaled(at_one, at_zero, F::one(), previous_claim)
    }

    /// [`Self::gruen_round_deg_2`] of `scale · Σ_g eq(g)·q_g(X)`, scaling the folded sums
    /// rather than every `q_g`.
    fn gruen_round_deg_2_scaled(
        &self,
        at_one: &(impl Fn(usize) -> F + Sync + Send),
        at_zero: &(impl Fn(usize) -> F + Sync + Send),
        scale: F,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, F>;
}

impl<F: JoltField> SplitEqUnreducedFold<F> for jolt_poly::GruenSplitEqPolynomial<F> {
    fn gruen_round_deg_3_from(
        &self,
        q_constant: F,
        q_quadratic: F,
        at_one: &(impl Fn(usize) -> F + Sync + Send),
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, F> {
        let q_one = || {
            let [q_one] = self.par_fold_out_in_unreduced::<1>(&|g| [at_one(g)]);
            q_one
        };
        let message = self.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim, q_one);
        check_gruen_endpoint(self, &message, true, q_one);
        message
    }

    fn gruen_round_deg_2_scaled(
        &self,
        at_one: &(impl Fn(usize) -> F + Sync + Send),
        at_zero: &(impl Fn(usize) -> F + Sync + Send),
        scale: F,
        previous_claim: F,
    ) -> Result<jolt_poly::UnivariatePoly<F>, F> {
        let [q_one] = self.par_fold_out_in_unreduced::<1>(&|g| [at_one(g)]);
        let q_one = q_one * scale;
        let q_zero = || {
            let [q_zero] = self.par_fold_out_in_unreduced::<1>(&|g| [at_zero(g)]);
            q_zero * scale
        };
        let message = self.gruen_poly_deg_2(q_one, previous_claim, q_zero);
        check_gruen_endpoint(self, &message, false, q_zero);
        message
    }

    #[inline]
    fn par_fold_out_in_unreduced<const NUM_OUT: usize>(
        &self,
        per_g_values: &(impl Fn(usize) -> [F; NUM_OUT] + Sync + Send),
    ) -> [F; NUM_OUT] {
        let e_out = self.e_out_current();
        let e_in = self.e_in_current();
        (0..e_out.len())
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|x_out| {
                let mut inner = [F::UnreducedProductAccum::zero(); NUM_OUT];
                for (x_in, e_in) in e_in.iter().enumerate() {
                    let vals = per_g_values(self.group_index(x_out, x_in));
                    for k in 0..NUM_OUT {
                        inner[k] += e_in.mul_to_product_accum(vals[k]);
                    }
                }
                let mut outer = [F::UnreducedProductAccum::zero(); NUM_OUT];
                for k in 0..NUM_OUT {
                    outer[k] = e_out[x_out].mul_to_product_accum(F::reduce_product_accum(inner[k]));
                }
                outer
            })
            .reduce_with(|mut a, b| {
                for k in 0..NUM_OUT {
                    a[k] += b[k];
                }
                a
            })
            .expect("par_fold_out_in_unreduced: empty E_out; invariant violation")
            .map(F::reduce_product_accum)
    }
}
