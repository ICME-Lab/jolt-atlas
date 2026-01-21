//! Hiding Commitment Helpers for Zero-Knowledge BlindFold
//!
//! This module provides helper types and NIFS-specific functions for working with
//! hiding (zero-knowledge) polynomial commitments.
//!
//! The main `HidingCommitmentScheme` trait is defined in `jolt_core::poly::commitment`.
//! This module provides:
//! - `ScalarBlindingFactor`: A simple blinding factor wrapper
//! - `fold_commitments`: Helper for NIFS commitment folding
//! - `fold_error_commitments`: Helper for NIFS error term folding

use jolt_core::{field::JoltField, poly::commitment::CommitmentScheme};

/// A simple blinding factor for field element blinding.
///
/// This is a single scalar that blinds the entire commitment.
#[derive(Clone, Debug, Default)]
pub struct ScalarBlindingFactor<F: JoltField> {
    /// The scalar blinding value
    pub scalar: F,
}

impl<F: JoltField> ScalarBlindingFactor<F> {
    /// Creates a new blinding factor with the given scalar.
    pub fn new(scalar: F) -> Self {
        Self { scalar }
    }

    /// Creates a zero (non-hiding) blinding factor.
    pub fn zero() -> Self {
        Self { scalar: F::zero() }
    }
}

// =============================================================================
// NIFS Helper Functions
// =============================================================================

/// Folds two commitments homomorphically: `C' = C_1 + r * C_2`
///
/// This is used in NIFS folding to compute:
/// - `W' = W_1 + r * W_2`
/// - Intermediate commitment combinations
///
/// Works with any PCS that implements `combine_commitments`.
pub fn fold_commitments<F, PCS>(c1: &PCS::Commitment, c2: &PCS::Commitment, r: &F) -> PCS::Commitment
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    PCS::Commitment: Clone,
{
    let commitments = [c1.clone(), c2.clone()];
    let coeffs = [F::one(), *r];
    PCS::combine_commitments(&commitments, &coeffs)
}

/// Folds three commitments for error term: `C' = C_1 + r * T + r^2 * C_2`
///
/// This is used in NIFS for computing the folded error commitment:
/// `E' = E_1 + r * T + r^2 * E_2`
pub fn fold_error_commitments<F, PCS>(
    e1: &PCS::Commitment,
    t: &PCS::Commitment,
    e2: &PCS::Commitment,
    r: &F,
) -> PCS::Commitment
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    PCS::Commitment: Clone,
{
    let r_squared = *r * *r;
    let commitments = [e1.clone(), t.clone(), e2.clone()];
    let coeffs = [F::one(), *r, r_squared];
    PCS::combine_commitments(&commitments, &coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn test_scalar_blinding_factor() {
        let blinding = ScalarBlindingFactor::<Fr>::new(Fr::from(42u64));
        assert_eq!(blinding.scalar, Fr::from(42u64));

        let zero = ScalarBlindingFactor::<Fr>::zero();
        assert_eq!(zero.scalar, Fr::from(0u64));
    }
}
