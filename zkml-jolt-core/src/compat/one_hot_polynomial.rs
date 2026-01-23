//! Stub implementations for removed one_hot_polynomial types
//!
//! These types were removed from jolt-core and need to be reimplemented
//! or the calling code needs to be updated.

use jolt_core::field::JoltField;
use std::sync::{Arc, RwLock};

/// Stub for OneHotSumcheckState which was removed from jolt-core
pub struct OneHotSumcheckState<F: JoltField> {
    _r_address: Vec<F::Challenge>,
    _r_cycle: Vec<F::Challenge>,
}

impl<F: JoltField> OneHotSumcheckState<F> {
    pub fn new(r_address: &[F::Challenge], r_cycle: &[F::Challenge]) -> Self {
        Self {
            _r_address: r_address.to_vec(),
            _r_cycle: r_cycle.to_vec(),
        }
    }
}

/// Stub for OneHotPolynomialProverOpening which was removed from jolt-core
#[derive(Clone)]
pub struct OneHotPolynomialProverOpening<F: JoltField> {
    _eq_state: Arc<RwLock<OneHotSumcheckState<F>>>,
}

impl<F: JoltField> OneHotPolynomialProverOpening<F> {
    pub fn new(eq_state: Arc<RwLock<OneHotSumcheckState<F>>>) -> Self {
        Self { _eq_state: eq_state }
    }

    /// Initialize the opening with a one-hot polynomial
    pub fn initialize(&mut self, _one_hot: jolt_core::poly::one_hot_polynomial::OneHotPolynomial<F>) {
        todo!("OneHotPolynomialProverOpening::initialize needs implementation")
    }

    /// Get the final sumcheck claim
    pub fn final_sumcheck_claim(&self) -> F {
        todo!("OneHotPolynomialProverOpening::final_sumcheck_claim needs implementation")
    }

    /// Compute prover message for a round
    pub fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        todo!("OneHotPolynomialProverOpening::compute_prover_message needs implementation")
    }

    /// Bind to a challenge
    pub fn bind(&mut self, _r_j: F::Challenge, _round: usize) {
        todo!("OneHotPolynomialProverOpening::bind needs implementation")
    }
}
