//! The Gruen split eq polynomial ([`GruenSplitEqPolynomial`], the Dao-Thaler + Gruen
//! optimization, https://eprint.iacr.org/2024/1210.pdf), the delayed-reduction fold our round
//! kernels use over it, and the Gruen round messages with their endpoint closures.

use common::parallel::par_enabled;
use num_traits::Zero;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::{eq::EqPolynomial, multilinear::BindingOrder, univariate::UnivariatePoly};

/// The split eq polynomial of Dao-Thaler + Gruen (https://eprint.iacr.org/2024/1210.pdf):
/// `eq(w, ·)` over the unbound variables, kept as an outer and an inner prefix table and the
/// scalar accumulated over the bound rounds, so a round kernel folds `e_out ⊗ e_in` and the
/// current variable's linear factor `l(X)` multiplies in at the end.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GruenSplitEqPolynomial<F: JoltField> {
    current_index: usize,
    current_scalar: F,
    point: Vec<F>,
    e_in_vec: Vec<Vec<F>>,
    e_out_vec: Vec<Vec<F>>,
    binding_order: BindingOrder,
}

impl<F: JoltField> GruenSplitEqPolynomial<F> {
    pub fn new(point: &[F], binding_order: BindingOrder) -> Self {
        if point.is_empty() {
            return Self {
                current_index: 0,
                current_scalar: F::one(),
                point: Vec::new(),
                e_in_vec: vec![vec![F::one()]],
                e_out_vec: vec![vec![F::one()]],
                binding_order,
            };
        }

        match binding_order {
            BindingOrder::LowToHigh => {
                let split = point.len() / 2;
                let head = &point[..point.len() - 1];
                let (out_point, in_point) = head.split_at(split.min(head.len()));
                let (e_out_vec, e_in_vec) = rayon::join(
                    || EqPolynomial::<F>::evals_cached(out_point),
                    || EqPolynomial::<F>::evals_cached(in_point),
                );
                Self {
                    current_index: point.len(),
                    current_scalar: F::one(),
                    point: point.to_vec(),
                    e_in_vec,
                    e_out_vec,
                    binding_order,
                }
            }
            BindingOrder::HighToLow => {
                let split = point.len() / 2;
                let tail = &point[1..];
                let (in_point, out_point) = tail.split_at(split.min(tail.len()));
                let (e_in_vec, e_out_vec) = rayon::join(
                    || EqPolynomial::<F>::evals_cached_rev(in_point),
                    || EqPolynomial::<F>::evals_cached_rev(out_point),
                );
                Self {
                    current_index: 0,
                    current_scalar: F::one(),
                    point: point.to_vec(),
                    e_in_vec,
                    e_out_vec,
                    binding_order,
                }
            }
        }
    }

    pub fn current_scalar(&self) -> F {
        self.current_scalar
    }

    pub fn current_linear_evals(&self) -> (F, F) {
        let point = match self.binding_order {
            BindingOrder::LowToHigh => self.point[self.current_index - 1],
            BindingOrder::HighToLow => self.point[self.current_index],
        };
        let at_one = self.current_scalar * point;
        (self.current_scalar - at_one, at_one)
    }

    pub fn e_in_current(&self) -> &[F] {
        &self.e_in_vec[self.e_in_vec.len() - 1]
    }

    pub fn e_out_current(&self) -> &[F] {
        &self.e_out_vec[self.e_out_vec.len() - 1]
    }

    pub fn e_in_current_len(&self) -> usize {
        self.e_in_current().len()
    }

    pub fn bind(&mut self, challenge: F) {
        if self.point.is_empty() {
            return;
        }

        match self.binding_order {
            BindingOrder::LowToHigh => {
                let point = self.point[self.current_index - 1];
                let product = point * challenge;
                self.current_scalar *= F::one() - point - challenge + product + product;
                self.current_index -= 1;
                if self.point.len() / 2 < self.current_index && self.e_in_vec.len() > 1 {
                    let _ = self.e_in_vec.pop();
                } else if 0 < self.current_index && self.e_out_vec.len() > 1 {
                    let _ = self.e_out_vec.pop();
                }
            }
            BindingOrder::HighToLow => {
                let point = self.point[self.current_index];
                let product = point * challenge;
                self.current_scalar *= F::one() - point - challenge + product + product;
                self.current_index += 1;
                if self.current_index <= self.point.len() / 2 && self.e_in_vec.len() > 1 {
                    let _ = self.e_in_vec.pop();
                } else if self.current_index <= self.point.len() && self.e_out_vec.len() > 1 {
                    let _ = self.e_out_vec.pop();
                }
            }
        }
    }

    /// Computes the cubic `s = l*q` from `q(0)`, its leading coefficient,
    /// and the sumcheck hint. The missing endpoint is evaluated only when
    /// `l(1)` vanishes. An error carries the actual endpoint sum.
    pub fn gruen_poly_deg_3(
        &self,
        q_constant: F,
        q_quadratic_coeff: F,
        s_0_plus_s_1: F,
        q_at_one: impl FnOnce() -> F,
    ) -> Result<UnivariatePoly<F>, F> {
        let (eq_eval_0, eq_eval_1) = self.current_linear_evals();
        if self.current_scalar.is_zero() {
            return Self::zero_round(4, s_0_plus_s_1);
        }
        let eq_m = eq_eval_1 - eq_eval_0;
        let eq_eval_2 = eq_eval_1 + eq_m;
        let eq_eval_3 = eq_eval_2 + eq_m;
        let cubic_eval_0 = eq_eval_0 * q_constant;
        let cubic_eval_1 = s_0_plus_s_1 - cubic_eval_0;
        let quadratic_eval_1 = if let Some(inverse) = eq_eval_1.inverse() {
            cubic_eval_1 * inverse
        } else {
            let endpoint = q_at_one();
            let actual = cubic_eval_0 + eq_eval_1 * endpoint;
            if actual != s_0_plus_s_1 {
                return Err(actual);
            }
            endpoint
        };
        let e_times_2 = q_quadratic_coeff + q_quadratic_coeff;
        let quadratic_eval_2 = quadratic_eval_1 + quadratic_eval_1 - q_constant + e_times_2;
        let quadratic_eval_3 =
            quadratic_eval_2 + quadratic_eval_1 - q_constant + e_times_2 + e_times_2;
        Ok(UnivariatePoly::interpolate_over_integers(&[
            cubic_eval_0,
            cubic_eval_1,
            eq_eval_2 * quadratic_eval_2,
            eq_eval_3 * quadratic_eval_3,
        ]))
    }

    /// Toom samples are `q(1)..q(d-1), q`'s leading coefficient, with `d>=2`.
    /// `q_evals` must contain at least two entries.
    /// The missing `q(0)` is evaluated only when `l(0)` vanishes.
    pub fn gruen_poly_from_evals(
        &self,
        q_evals: &[F],
        s_0_plus_s_1: F,
        q_at_zero: impl FnOnce() -> F,
    ) -> Result<UnivariatePoly<F>, F> {
        if self.current_scalar.is_zero() {
            return Self::zero_round(q_evals.len() + 2, s_0_plus_s_1);
        }
        let q_zero = self.recover_q_zero(q_evals[0], s_0_plus_s_1, q_at_zero)?;
        let mut full_q_evals = Vec::with_capacity(q_evals.len() + 1);
        full_q_evals.push(q_zero);
        full_q_evals.extend_from_slice(q_evals);
        let q_coeffs = UnivariatePoly::from_evals_toom(&full_q_evals).into_coefficients();
        Ok(self.multiply_linear_factor(&q_coeffs))
    }

    /// Degree-two message for a linear inner factor, with lazy `q(0)` recovery.
    pub fn gruen_poly_deg_2(
        &self,
        q_one: F,
        s_0_plus_s_1: F,
        q_at_zero: impl FnOnce() -> F,
    ) -> Result<UnivariatePoly<F>, F> {
        if self.current_scalar.is_zero() {
            return Self::zero_round(3, s_0_plus_s_1);
        }
        let q_zero = self.recover_q_zero(q_one, s_0_plus_s_1, q_at_zero)?;
        Ok(self.multiply_linear_factor(&[q_zero, q_one - q_zero]))
    }

    fn recover_q_zero(&self, q_one: F, hint: F, q_at_zero: impl FnOnce() -> F) -> Result<F, F> {
        let (l_zero, l_one) = self.current_linear_evals();
        if let Some(inverse) = l_zero.inverse() {
            Ok((hint - l_one * q_one) * inverse)
        } else {
            let q_zero = q_at_zero();
            let actual = l_zero * q_zero + l_one * q_one;
            if actual == hint {
                Ok(q_zero)
            } else {
                Err(actual)
            }
        }
    }

    fn multiply_linear_factor(&self, q_coeffs: &[F]) -> UnivariatePoly<F> {
        let (l_zero, l_one) = self.current_linear_evals();
        let l_slope = l_one - l_zero;
        let mut coefficients = vec![F::zero(); q_coeffs.len() + 1];
        for (index, q_coeff) in q_coeffs.iter().copied().enumerate() {
            coefficients[index] += q_coeff * l_zero;
            coefficients[index + 1] += q_coeff * l_slope;
        }
        UnivariatePoly::new(coefficients)
    }

    fn zero_round(coefficients: usize, hint: F) -> Result<UnivariatePoly<F>, F> {
        if hint.is_zero() {
            Ok(UnivariatePoly::new(vec![F::zero(); coefficients]))
        } else {
            Err(F::zero())
        }
    }

    /// The full eq table over the unbound variables, scaled by the bound rounds' factors.
    pub fn merge(&self) -> Vec<F> {
        let unbound = match self.binding_order {
            BindingOrder::LowToHigh => &self.point[..self.current_index],
            BindingOrder::HighToLow => &self.point[self.current_index..],
        };
        EqPolynomial::<F>::evals_with_scaling(unbound, Some(self.current_scalar))
    }

    #[inline(always)]
    pub fn group_index(&self, x_out: usize, x_in: usize) -> usize {
        let in_bits = self.e_in_current_len().trailing_zeros() as usize;
        (x_out << in_bits) | x_in
    }

    #[inline]
    pub fn par_fold_out_in<
        OuterAcc: Send,
        InnerAcc: Send,
        MakeInner: Fn() -> InnerAcc + Sync + Send,
        InnerStep: Fn(&mut InnerAcc, usize, usize, F) + Sync + Send,
        OuterStep: Fn(usize, F, InnerAcc) -> OuterAcc + Sync + Send,
        Merge: Fn(OuterAcc, OuterAcc) -> OuterAcc + Sync + Send,
    >(
        &self,
        make_inner: MakeInner,
        inner_step: InnerStep,
        outer_step: OuterStep,
        merge: Merge,
    ) -> OuterAcc {
        let e_out = self.e_out_current();
        let e_in = self.e_in_current();
        (0..e_out.len())
            .into_par_iter()
            .map(|x_out| {
                let mut inner_acc = make_inner();
                for (x_in, &e_in) in e_in.iter().enumerate() {
                    let row = self.group_index(x_out, x_in);
                    inner_step(&mut inner_acc, row, x_in, e_in);
                }
                outer_step(x_out, e_out[x_out], inner_acc)
            })
            .reduce_with(merge)
            .unwrap_or_else(|| {
                let inner_acc = make_inner();
                outer_step(0, F::zero(), inner_acc)
            })
    }
}

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
    eq: &GruenSplitEqPolynomial<F>,
    message: &Result<UnivariatePoly<F>, F>,
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

/// The delayed-reduction fold over the split eq, and the Gruen round messages built on it.
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
    ) -> Result<UnivariatePoly<F>, F> {
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
    ) -> Result<UnivariatePoly<F>, F>;

    /// The quadratic round message `l(X)·q(X)` of `Σ_g eq(g)·q_g(X)` with each `q_g` linear:
    /// `at_one(g) = q_g(1)`, and `at_zero(g) = q_g(0)`, folded only when the eq factor vanishes
    /// at 0. The error carries the actual `s(0) + s(1)`.
    fn gruen_round_deg_2(
        &self,
        at_one: &(impl Fn(usize) -> F + Sync + Send),
        at_zero: &(impl Fn(usize) -> F + Sync + Send),
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, F> {
        self.gruen_round_deg_2_scaled(at_one, at_zero, F::one(), previous_claim)
    }

    /// The round message `l(X)·q(X)` of `Σ_g eq(g)·q_g(X)` from `q`'s Toom samples:
    /// `per_g(g) = [q_g(1), …, q_g(N−1), q_g's leading coefficient]`, and `at_zero(g) = q_g(0)`,
    /// folded only when the eq factor vanishes at 0. The error carries the actual `s(0) + s(1)`.
    fn gruen_round_from_evals<const N: usize>(
        &self,
        per_g: &(impl Fn(usize) -> [F; N] + Sync + Send),
        at_zero: &(impl Fn(usize) -> F + Sync + Send),
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, F>;

    /// [`Self::gruen_round_deg_2`] of `scale · Σ_g eq(g)·q_g(X)`, scaling the folded sums
    /// rather than every `q_g`.
    fn gruen_round_deg_2_scaled(
        &self,
        at_one: &(impl Fn(usize) -> F + Sync + Send),
        at_zero: &(impl Fn(usize) -> F + Sync + Send),
        scale: F,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, F>;
}

impl<F: JoltField> SplitEqUnreducedFold<F> for GruenSplitEqPolynomial<F> {
    fn gruen_round_from_evals<const N: usize>(
        &self,
        per_g: &(impl Fn(usize) -> [F; N] + Sync + Send),
        at_zero: &(impl Fn(usize) -> F + Sync + Send),
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, F> {
        let evals = self.par_fold_out_in_unreduced::<N>(per_g);
        let q_zero = || {
            let [q_zero] = self.par_fold_out_in_unreduced::<1>(&|g| [at_zero(g)]);
            q_zero
        };
        let message = self.gruen_poly_from_evals(&evals, previous_claim, q_zero);
        check_gruen_endpoint(self, &message, false, q_zero);
        message
    }

    fn gruen_round_deg_3_from(
        &self,
        q_constant: F,
        q_quadratic: F,
        at_one: &(impl Fn(usize) -> F + Sync + Send),
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, F> {
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
    ) -> Result<UnivariatePoly<F>, F> {
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

#[cfg(test)]
mod tests {
    use crate::field::fp128::Fp128 as Fr;
    use jolt_field::{Field, Ring};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;
    use std::cell::Cell;

    use super::*;

    fn random_point(len: usize, seed: u64) -> Vec<Fr> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..len).map(|_| Fr::random(&mut rng)).collect()
    }

    fn eq_factor(w: Fr, c: Fr) -> Fr {
        (Fr::one() - w) * (Fr::one() - c) + w * c
    }

    /// The current round's eq factor `l(X) = scalar * ((1-w)(1-X) + wX)`, built by hand from the
    /// point coordinate and an independently tracked scalar, never from the struct's internals.
    fn hand_built_linear(scalar: Fr, w: Fr) -> UnivariatePoly<Fr> {
        UnivariatePoly::new(vec![scalar * (Fr::one() - w), scalar * (w + w - Fr::one())])
    }

    /// The coordinate of `point` bound in round `round`.
    fn round_variable(point: &[Fr], round: usize, order: BindingOrder) -> Fr {
        match order {
            BindingOrder::LowToHigh => point[point.len() - 1 - round],
            BindingOrder::HighToLow => point[round],
        }
    }

    /// Every unbound row's `e_out · e_in · l(b)` is the full eq table over the unbound variables,
    /// scaled by the bound rounds' factors.
    #[test]
    fn split_tables_and_linear_factor_factor_the_bound_eq_table() {
        for (order, seed) in [
            (BindingOrder::LowToHigh, 811u64),
            (BindingOrder::HighToLow, 919),
        ] {
            let point = random_point(8, seed);
            let challenges = random_point(5, seed + 1);
            let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, order);
            let mut scalar = Fr::one();
            for (round, challenge) in challenges.into_iter().enumerate() {
                let unbound = match order {
                    BindingOrder::LowToHigh => &point[..point.len() - round],
                    BindingOrder::HighToLow => &point[round..],
                };
                let full = EqPolynomial::<Fr>::evals_with_scaling(unbound, Some(scalar));
                let half = full.len() / 2;
                let (linear_0, linear_1) = split.current_linear_evals();
                let e_out = split.e_out_current();
                let e_in = split.e_in_current();
                for x_out in 0..e_out.len() {
                    for x_in in 0..e_in.len() {
                        let row = split.group_index(x_out, x_in);
                        let (dense_0, dense_1) = match order {
                            BindingOrder::LowToHigh => (full[2 * row], full[2 * row + 1]),
                            BindingOrder::HighToLow => {
                                let out_bits = e_out.len().trailing_zeros() as usize;
                                let dense_row = (x_in << out_bits) | x_out;
                                (full[dense_row], full[dense_row + half])
                            }
                        };
                        let head = e_out[x_out] * e_in[x_in];
                        assert_eq!(head * linear_0, dense_0, "{order:?} round {round}");
                        assert_eq!(head * linear_1, dense_1, "{order:?} round {round}");
                    }
                }
                scalar *= eq_factor(round_variable(&point, round, order), challenge);
                split.bind(challenge);
                assert_eq!(split.current_scalar(), scalar, "{order:?} round {round}");
            }
        }
    }

    #[test]
    fn gruen_poly_deg_3_equals_hand_built_linear_times_quadratic() {
        for (order, seed) in [
            (BindingOrder::LowToHigh, 1301u64),
            (BindingOrder::HighToLow, 1409),
        ] {
            let point = random_point(6, seed);
            let challenges = random_point(3, seed + 1);
            let q = UnivariatePoly::new(random_point(3, seed + 2));

            let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, order);
            let mut hand_scalar = Fr::one();
            for (round, challenge) in challenges.into_iter().enumerate() {
                let w = round_variable(&point, round, order);
                let l = hand_built_linear(hand_scalar, w);
                let hint = l.evaluate(Fr::zero()) * q.evaluate(Fr::zero())
                    + l.evaluate(Fr::one()) * q.evaluate(Fr::one());
                let s = split
                    .gruen_poly_deg_3(q.coefficients()[0], q.coefficients()[2], hint, || {
                        q.evaluate(Fr::one())
                    })
                    .unwrap();
                assert_eq!(s.coefficients().len(), 4, "{order:?} round {round}");
                // s and l*q both have degree <= 3, so agreement on 8 points forces equality.
                for x in (0..8u64).map(Fr::from_u64) {
                    assert_eq!(
                        s.evaluate(x),
                        l.evaluate(x) * q.evaluate(x),
                        "{order:?} round {round}"
                    );
                }
                hand_scalar *= eq_factor(w, challenge);
                split.bind(challenge);
            }
        }
    }

    #[test]
    fn gruen_poly_from_evals_equals_hand_built_linear_times_q() {
        for (order, seed) in [
            (BindingOrder::LowToHigh, 2003u64),
            (BindingOrder::HighToLow, 2087),
        ] {
            // gruen_poly_from_evals requires degree >= 2: q_evals[0] must be q(1).
            for degree in [2usize, 3] {
                let point = random_point(5, seed + degree as u64);
                let challenges = random_point(2, seed + 10 + degree as u64);
                let q = UnivariatePoly::new(random_point(degree + 1, seed + 20 + degree as u64));

                let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, order);
                let mut hand_scalar = Fr::one();
                for (round, &challenge) in challenges.iter().enumerate() {
                    hand_scalar *= eq_factor(round_variable(&point, round, order), challenge);
                    split.bind(challenge);
                }

                let w = round_variable(&point, challenges.len(), order);
                let l = hand_built_linear(hand_scalar, w);
                // Toom layout: [q(1), ..., q(degree-1), leading coefficient].
                let mut q_evals: Vec<Fr> = (1..degree as u64)
                    .map(|x| q.evaluate(Fr::from_u64(x)))
                    .collect();
                q_evals.push(q.coefficients()[degree]);
                let hint = l.evaluate(Fr::zero()) * q.evaluate(Fr::zero())
                    + l.evaluate(Fr::one()) * q.evaluate(Fr::one());

                let s = split
                    .gruen_poly_from_evals(&q_evals, hint, || q.evaluate(Fr::zero()))
                    .unwrap();
                assert_eq!(s.coefficients().len(), degree + 2, "{order:?} deg {degree}");
                for x in (0..2 * degree as u64 + 3).map(Fr::from_u64) {
                    assert_eq!(
                        s.evaluate(x),
                        l.evaluate(x) * q.evaluate(x),
                        "{order:?} deg {degree}"
                    );
                }
            }
        }
    }

    /// At an eq coordinate of 0 or 1 one endpoint of `l` vanishes, so the message takes the
    /// endpoint closure instead of dividing; a bound coordinate of 0 at challenge 1 zeroes the
    /// scalar, so every message is the zero round.
    #[test]
    fn exceptional_gruen_rounds_match_true_products() {
        for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            for coordinate in [Fr::zero(), Fr::one(), Fr::from_u64(7)] {
                for killed_prefix in [false, true] {
                    let split = if killed_prefix {
                        let point = match order {
                            BindingOrder::LowToHigh => vec![coordinate, Fr::zero()],
                            BindingOrder::HighToLow => vec![Fr::zero(), coordinate],
                        };
                        let mut split = GruenSplitEqPolynomial::new(&point, order);
                        split.bind(Fr::one());
                        split
                    } else {
                        GruenSplitEqPolynomial::new(&[coordinate], order)
                    };
                    let scalar = if killed_prefix { Fr::zero() } else { Fr::one() };
                    let l_zero = scalar * (Fr::one() - coordinate);
                    let l_one = scalar * coordinate;
                    for degree in [1usize, 2, 3] {
                        let q = UnivariatePoly::new(
                            (0..=degree)
                                .map(|i| Fr::from_u64(2 + 3 * i as u64))
                                .collect(),
                        );
                        let hint = l_zero * q.evaluate(Fr::zero()) + l_one * q.evaluate(Fr::one());
                        let samples = || {
                            let mut samples: Vec<Fr> = (1..degree)
                                .map(|i| q.evaluate(Fr::from_u64(i as u64)))
                                .collect();
                            samples.push(q.coefficients()[degree]);
                            samples
                        };
                        let called = Cell::new(false);
                        let polynomial = if degree == 1 {
                            split.gruen_poly_deg_2(q.evaluate(Fr::one()), hint, || {
                                called.set(true);
                                q.evaluate(Fr::zero())
                            })
                        } else {
                            split.gruen_poly_from_evals(&samples(), hint, || {
                                called.set(true);
                                q.evaluate(Fr::zero())
                            })
                        }
                        .unwrap();
                        assert_eq!(called.get(), !scalar.is_zero() && l_zero.is_zero());
                        if !scalar.is_zero() && l_zero.is_zero() {
                            let bad_hint = hint + Fr::one();
                            let rejected = if degree == 1 {
                                split.gruen_poly_deg_2(q.evaluate(Fr::one()), bad_hint, || {
                                    q.evaluate(Fr::zero())
                                })
                            } else {
                                split.gruen_poly_from_evals(&samples(), bad_hint, || {
                                    q.evaluate(Fr::zero())
                                })
                            };
                            assert_eq!(rejected, Err(hint));
                        }
                        assert_eq!(polynomial.coefficients().len(), degree + 2);
                        for x in (0..8).map(Fr::from_u64) {
                            let linear = l_zero + (l_one - l_zero) * x;
                            assert_eq!(polynomial.evaluate(x), linear * q.evaluate(x));
                        }
                        if degree == 2 {
                            called.set(false);
                            let cubic = split
                                .gruen_poly_deg_3(
                                    q.coefficients()[0],
                                    q.coefficients()[2],
                                    hint,
                                    || {
                                        called.set(true);
                                        q.evaluate(Fr::one())
                                    },
                                )
                                .unwrap();
                            assert_eq!(called.get(), !scalar.is_zero() && l_one.is_zero());
                            assert_eq!(cubic.coefficients(), polynomial.coefficients());
                            if !scalar.is_zero() && l_one.is_zero() {
                                assert_eq!(
                                    split.gruen_poly_deg_3(
                                        q.coefficients()[0],
                                        q.coefficients()[2],
                                        hint + Fr::one(),
                                        || q.evaluate(Fr::one())
                                    ),
                                    Err(hint)
                                );
                            }
                        }
                    }
                    if scalar.is_zero() {
                        let never = || panic!("zero factor must not scan");
                        assert_eq!(
                            split.gruen_poly_deg_3(Fr::one(), Fr::one(), Fr::one(), never),
                            Err(Fr::zero())
                        );
                        assert_eq!(
                            split.gruen_poly_from_evals(&[Fr::one(), Fr::one()], Fr::one(), never),
                            Err(Fr::zero())
                        );
                        assert_eq!(
                            split.gruen_poly_deg_2(Fr::one(), Fr::one(), never),
                            Err(Fr::zero())
                        );
                    }
                }
            }
        }
    }
}
