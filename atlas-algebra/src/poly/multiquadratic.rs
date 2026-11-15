use allocative::Allocative;

use crate::field::JoltField;
use crate::poly::multilinear::{BindingOrder, PolynomialBinding};

/// Multiquadratic polynomial represented by its evaluations on the grid
/// {0, 1, ∞}^num_vars in base-3 layout (z_0 least-significant / fastest-varying).
#[derive(Allocative)]
pub struct MultiquadraticPolynomial<F: JoltField> {
    num_vars: usize,
    evals: Vec<F>,
}

impl<F: JoltField> MultiquadraticPolynomial<F> {
    /// Construct a multiquadratic polynomial from its full grid of evaluations.
    /// The caller is responsible for ensuring that `evals` is laid out in base-3
    /// order with z_0 as the least-significant digit.
    pub fn new(num_vars: usize, evals: Vec<F>) -> Self {
        let expected_len = 3usize.pow(num_vars as u32);
        debug_assert!(
            evals.len() == expected_len,
            "MultiquadraticPolynomial: expected {} evals, got {}",
            expected_len,
            evals.len()
        );
        Self { num_vars, evals }
    }

    /// Number of variables in the polynomial.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Underlying evaluations on {0, 1, ∞}^num_vars.
    pub fn evals(&self) -> &[F] {
        &self.evals
    }

    /// Bind the first (least-significant) variable z_0 := r, reducing the
    /// dimension from w to w-1 and keeping the base-3 layout invariant.
    ///
    /// For each assignment to (z_1, ..., z_{w-1}), we have three stored values
    ///   f(0, ..), f(1, ..), f(∞, ..)
    /// and interpolate the unique quadratic in z_0 that matches them, then
    /// evaluate it at z_0 = r.
    pub fn bind_first_variable(&mut self, r: F::Challenge) {
        let w = self.num_vars;
        debug_assert!(w > 0);

        let new_size = 3_usize.pow((w - 1) as u32);
        let one = F::one();

        let r_term = r * (r - one);
        for new_idx in 0..new_size {
            let old_base_idx = new_idx * 3;
            let eval_at_0 = self.evals[old_base_idx]; // z_0 = 0
            let eval_at_1 = self.evals[old_base_idx + 1]; // z_0 = 1
            let eval_at_inf = self.evals[old_base_idx + 2]; // z_0 = ∞

            self.evals[new_idx] = eval_at_0 * (one - r) + eval_at_1 * r + eval_at_inf * r_term;
        }

        self.num_vars -= 1;
        self.evals.truncate(new_size);
    }
}

impl<F: JoltField> PolynomialBinding<F> for MultiquadraticPolynomial<F> {
    fn is_bound(&self) -> bool {
        self.num_vars == 0 || self.evals.len() == 1
    }

    #[tracing::instrument(skip_all, name = "MultiquadraticPolynomial::bind")]
    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        match order {
            BindingOrder::LowToHigh => self.bind_first_variable(r),
            BindingOrder::HighToLow => {
                // Not currently needed by the outer Spartan streaming code.
                unimplemented!(
                    "HighToLow binding order is not implemented for MultiquadraticPolynomial"
                )
            }
        }
    }

    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        // Window sizes are small; fall back to the sequential implementation.
        self.bind(r, order);
    }

    fn final_claim(&self) -> F {
        debug_assert!(self.is_bound());
        debug_assert_eq!(self.evals.len(), 1);
        self.evals[0]
    }
}
