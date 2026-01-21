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
//! - Mock types for testing

use jolt_core::{
    field::JoltField,
    poly::commitment::CommitmentScheme,
};
use rand_core::{CryptoRng, RngCore};

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

/// Mock hiding commitment scheme for testing.
///
/// This implements `HidingCommitmentScheme` using a simple mock approach
/// that doesn't provide actual cryptographic hiding, but is useful for
/// testing the BlindFold protocol logic.
#[derive(Clone, Debug)]
pub struct MockHidingCommitment<F: JoltField> {
    /// The "committed" value (not actually hidden in mock)
    pub value: F,
    /// The blinding factor used
    pub blinding: F,
}

impl<F: JoltField> Default for MockHidingCommitment<F> {
    fn default() -> Self {
        Self {
            value: F::zero(),
            blinding: F::zero(),
        }
    }
}

impl<F: JoltField> MockHidingCommitment<F> {
    /// Creates a new mock commitment.
    pub fn new(value: F, blinding: F) -> Self {
        Self { value, blinding }
    }

    /// Returns the effective commitment value (value + blinding for mock).
    pub fn effective_value(&self) -> F {
        self.value + self.blinding
    }
}

/// Mock hiding commitment scheme for testing BlindFold.
///
/// This scheme doesn't provide real cryptographic security but implements
/// all the necessary operations for testing the zero-knowledge protocol.
pub struct MockHidingScheme<F: JoltField> {
    _marker: std::marker::PhantomData<F>,
}

impl<F: JoltField> MockHidingScheme<F> {
    /// Commits to polynomial coefficients with blinding.
    pub fn commit_with_blinding(
        coeffs: &[F],
        blinding: &ScalarBlindingFactor<F>,
    ) -> MockHidingCommitment<F> {
        // Mock commitment: hash of coefficients (simplified to sum)
        let value: F = coeffs.iter().fold(F::zero(), |acc, c| acc + *c);
        MockHidingCommitment::new(value, blinding.scalar)
    }

    /// Samples a random blinding factor.
    pub fn sample_blinding<R: RngCore + CryptoRng>(rng: &mut R) -> ScalarBlindingFactor<F> {
        ScalarBlindingFactor::new(F::random(rng))
    }

    /// Combines blinding factors linearly.
    pub fn combine_blindings(
        blindings: &[ScalarBlindingFactor<F>],
        coeffs: &[F],
    ) -> ScalarBlindingFactor<F> {
        let combined: F = blindings
            .iter()
            .zip(coeffs.iter())
            .map(|(b, c)| b.scalar * *c)
            .fold(F::zero(), |acc, x| acc + x);
        ScalarBlindingFactor::new(combined)
    }

    /// Folds two commitments: C' = C_1 + r * C_2
    pub fn fold_commitments(
        c1: &MockHidingCommitment<F>,
        c2: &MockHidingCommitment<F>,
        r: &F,
    ) -> MockHidingCommitment<F> {
        MockHidingCommitment::new(
            c1.value + *r * c2.value,
            c1.blinding + *r * c2.blinding,
        )
    }

    /// Adds blinding to an existing commitment.
    pub fn add_blinding(
        commitment: &MockHidingCommitment<F>,
        additional_blinding: &ScalarBlindingFactor<F>,
        r: &F,
    ) -> MockHidingCommitment<F> {
        MockHidingCommitment::new(
            commitment.value,
            commitment.blinding + *r * additional_blinding.scalar,
        )
    }
}

/// Wrapper to make an existing CommitmentScheme into a HidingCommitmentScheme.
///
/// This adds blinding support to any PCS by tracking blinding factors separately.
/// The actual hiding is achieved by including blinding in the Fiat-Shamir transcript.
#[derive(Clone, Debug)]
pub struct HidingWrapper<PCS: CommitmentScheme> {
    _marker: std::marker::PhantomData<PCS>,
}

/// Blinding factor for the HidingWrapper.
#[derive(Clone, Debug, Default)]
pub struct WrapperBlindingFactor<F: JoltField> {
    /// The main blinding scalar
    pub scalar: F,
    /// Additional blinding for multi-polynomial scenarios
    pub auxiliary: Vec<F>,
}

impl<F: JoltField> WrapperBlindingFactor<F> {
    /// Creates a new blinding factor.
    pub fn new(scalar: F) -> Self {
        Self {
            scalar,
            auxiliary: Vec::new(),
        }
    }

    /// Creates a blinding factor with auxiliary values.
    pub fn with_auxiliary(scalar: F, auxiliary: Vec<F>) -> Self {
        Self { scalar, auxiliary }
    }
}

impl<PCS: CommitmentScheme> HidingWrapper<PCS> {
    /// Creates a new hiding wrapper.
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<PCS: CommitmentScheme> Default for HidingWrapper<PCS> {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper methods for blinding factor management with any PCS.
///
/// This provides utility functions for working with blinding factors
/// without requiring a full `HidingCommitmentScheme` implementation.
///
/// For true cryptographic hiding, the underlying PCS should have native
/// support for Pedersen-style blinding. This helper provides the interface
/// for tracking blinding factors and combining them correctly.
impl<PCS: CommitmentScheme> HidingWrapper<PCS> {
    /// Samples a random blinding factor.
    pub fn sample_blinding<R: RngCore + CryptoRng>(rng: &mut R) -> ScalarBlindingFactor<PCS::Field> {
        ScalarBlindingFactor::new(PCS::Field::random(rng))
    }

    /// Combines multiple blinding factors with coefficients.
    ///
    /// Given blindings `[b_1, ..., b_n]` and coefficients `[c_1, ..., c_n]`,
    /// computes `sum_i(c_i * b_i)`.
    pub fn combine_blindings(
        blindings: &[ScalarBlindingFactor<PCS::Field>],
        coeffs: &[PCS::Field],
    ) -> ScalarBlindingFactor<PCS::Field> {
        let combined = blindings
            .iter()
            .zip(coeffs.iter())
            .map(|(b, c)| b.scalar * *c)
            .fold(PCS::Field::from_u64(0), |acc, x| acc + x);
        ScalarBlindingFactor::new(combined)
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
pub fn fold_commitments<F, PCS>(
    c1: &PCS::Commitment,
    c2: &PCS::Commitment,
    r: &F,
) -> PCS::Commitment
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
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_scalar_blinding_factor() {
        let blinding = ScalarBlindingFactor::<Fr>::new(Fr::from(42u64));
        assert_eq!(blinding.scalar, Fr::from(42u64));

        let zero = ScalarBlindingFactor::<Fr>::zero();
        assert_eq!(zero.scalar, Fr::from(0u64));
    }

    #[test]
    fn test_mock_hiding_commitment() {
        let commitment = MockHidingCommitment::<Fr>::new(Fr::from(10u64), Fr::from(5u64));
        assert_eq!(commitment.value, Fr::from(10u64));
        assert_eq!(commitment.blinding, Fr::from(5u64));
        assert_eq!(commitment.effective_value(), Fr::from(15u64));
    }

    #[test]
    fn test_mock_hiding_scheme_commit() {
        let coeffs = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
        let blinding = ScalarBlindingFactor::new(Fr::from(100u64));

        let commitment = MockHidingScheme::commit_with_blinding(&coeffs, &blinding);

        // Sum of coefficients = 1 + 2 + 3 = 6
        assert_eq!(commitment.value, Fr::from(6u64));
        assert_eq!(commitment.blinding, Fr::from(100u64));
    }

    #[test]
    fn test_mock_hiding_scheme_sample_blinding() {
        let mut rng = StdRng::seed_from_u64(12345);

        let b1 = MockHidingScheme::<Fr>::sample_blinding(&mut rng);
        let b2 = MockHidingScheme::<Fr>::sample_blinding(&mut rng);

        // Blindings should be different
        assert_ne!(b1.scalar, b2.scalar);
        // Blindings should be non-zero (with overwhelming probability)
        assert_ne!(b1.scalar, Fr::from(0u64));
    }

    #[test]
    fn test_mock_hiding_scheme_combine_blindings() {
        let b1 = ScalarBlindingFactor::new(Fr::from(10u64));
        let b2 = ScalarBlindingFactor::new(Fr::from(20u64));
        let coeffs = vec![Fr::from(2u64), Fr::from(3u64)];

        let combined = MockHidingScheme::combine_blindings(&[b1, b2], &coeffs);

        // 10 * 2 + 20 * 3 = 20 + 60 = 80
        assert_eq!(combined.scalar, Fr::from(80u64));
    }

    #[test]
    fn test_mock_hiding_scheme_fold_commitments() {
        let c1 = MockHidingCommitment::new(Fr::from(10u64), Fr::from(1u64));
        let c2 = MockHidingCommitment::new(Fr::from(20u64), Fr::from(2u64));
        let r = Fr::from(3u64);

        let combined = MockHidingScheme::fold_commitments(&c1, &c2, &r);

        // value: 10 + 3 * 20 = 70
        assert_eq!(combined.value, Fr::from(70u64));
        // blinding: 1 + 3 * 2 = 7
        assert_eq!(combined.blinding, Fr::from(7u64));
    }

    #[test]
    fn test_hiding_preserves_homomorphism() {
        // Test that folding hiding commitments works correctly:
        // Commit(m1, r1) + Commit(m2, r2) * c should equal Commit(m1 + c*m2, r1 + c*r2)

        let m1 = vec![Fr::from(5u64), Fr::from(10u64)];
        let m2 = vec![Fr::from(3u64), Fr::from(7u64)];
        let r1 = ScalarBlindingFactor::new(Fr::from(11u64));
        let r2 = ScalarBlindingFactor::new(Fr::from(13u64));
        let c = Fr::from(2u64);

        // Commit then fold
        let c1 = MockHidingScheme::commit_with_blinding(&m1, &r1);
        let c2 = MockHidingScheme::commit_with_blinding(&m2, &r2);
        let folded_via_commits = MockHidingScheme::fold_commitments(&c1, &c2, &c);

        // Fold then commit
        let m_folded: Vec<Fr> = m1.iter().zip(m2.iter()).map(|(a, b)| *a + c * *b).collect();
        let r_folded = ScalarBlindingFactor::new(r1.scalar + c * r2.scalar);
        let commit_of_fold = MockHidingScheme::commit_with_blinding(&m_folded, &r_folded);

        // Should be equal (homomorphism)
        assert_eq!(folded_via_commits.value, commit_of_fold.value);
        assert_eq!(folded_via_commits.blinding, commit_of_fold.blinding);
    }

    #[test]
    fn test_wrapper_blinding_factor() {
        let bf = WrapperBlindingFactor::<Fr>::new(Fr::from(42u64));
        assert_eq!(bf.scalar, Fr::from(42u64));
        assert!(bf.auxiliary.is_empty());

        let bf_aux = WrapperBlindingFactor::with_auxiliary(
            Fr::from(10u64),
            vec![Fr::from(1u64), Fr::from(2u64)],
        );
        assert_eq!(bf_aux.scalar, Fr::from(10u64));
        assert_eq!(bf_aux.auxiliary.len(), 2);
    }
}
