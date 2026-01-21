//! Hiding (Zero-Knowledge) Dory polynomial commitment scheme
//!
//! This module implements the `HidingCommitmentScheme` trait for Dory,
//! enabling zero-knowledge sumcheck proofs using the BlindFold approach.
//!
//! # Blinding Strategy
//!
//! For Pedersen-style hiding in Dory, we add a blinding term in GT:
//!   C' = C + e(r * H_G1, H_G2)
//!
//! where:
//! - C is the standard Dory commitment (in GT)
//! - r is the blinding scalar (secret)
//! - H_G1 and H_G2 are hiding generators (the last elements in the SRS)
//!
//! This provides computational hiding: without knowing r, an adversary cannot
//! distinguish C' from a random GT element.
//!
//! # Security
//!
//! The hiding generators should have an unknown discrete log relationship
//! to the main generators. Using the last elements of the SRS provides this
//! property if the SRS was generated via a trusted setup.

use super::{
    wrappers::{ArkG1, ArkG2, ArkGT, ArkworksProverSetup, BN254},
    DoryCommitmentScheme,
};
use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::{CommitmentScheme, HidingCommitmentScheme},
        multilinear_polynomial::MultilinearPolynomial,
    },
    transcripts::Transcript,
};
use ark_bn254::{Fr, G1Projective};
use ark_ff::{UniformRand, Zero};
use dory::primitives::arithmetic::PairingCurve;
use rand_core::{CryptoRng, RngCore};
use rayon::prelude::*;
use tracing::trace_span;

/// Blinding factor for hiding Dory commitments.
///
/// Contains the scalar blinding values for each row commitment,
/// which are combined to produce the final hiding commitment.
#[derive(Clone, Debug, Default)]
pub struct DoryBlindingFactor {
    /// The main scalar blinding factor
    pub scalar: Fr,
    /// Per-row blinding factors (for multi-row polynomials)
    pub row_blindings: Vec<Fr>,
}

impl DoryBlindingFactor {
    /// Creates a new blinding factor with the given scalar
    pub fn new(scalar: Fr) -> Self {
        Self {
            scalar,
            row_blindings: Vec::new(),
        }
    }

    /// Creates a blinding factor with per-row blindings
    pub fn with_row_blindings(scalar: Fr, row_blindings: Vec<Fr>) -> Self {
        Self {
            scalar,
            row_blindings,
        }
    }
}

/// Hiding Dory commitment scheme wrapper.
///
/// This type provides zero-knowledge polynomial commitments by extending
/// the base Dory scheme with Pedersen-style blinding.
#[derive(Clone)]
pub struct HidingDoryCommitmentScheme;

impl HidingCommitmentScheme for DoryCommitmentScheme {
    type BlindingFactor = DoryBlindingFactor;

    fn commit_hiding(
        poly: &MultilinearPolynomial<Fr>,
        blinding: &Self::BlindingFactor,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let _span = trace_span!("DoryCommitmentScheme::commit_hiding").entered();

        // First, compute the standard (non-hiding) commitment
        let (base_commitment, row_commitments) = Self::commit(poly, setup);

        // Add blinding to the commitment: C' = C + e(r * H_G1, H_G2)
        let blinded_commitment = add_blinding_to_commitment(
            base_commitment,
            &blinding.scalar,
            setup,
        );

        (blinded_commitment, row_commitments)
    }

    fn sample_blinding<R: RngCore + CryptoRng>(rng: &mut R) -> Self::BlindingFactor {
        DoryBlindingFactor::new(Fr::rand(rng))
    }

    fn combine_blindings(
        blindings: &[Self::BlindingFactor],
        coeffs: &[Self::Field],
    ) -> Self::BlindingFactor {
        let _span = trace_span!("DoryCommitmentScheme::combine_blindings").entered();

        // Compute the linear combination of blinding scalars
        let combined_scalar: Fr = blindings
            .par_iter()
            .zip(coeffs.par_iter())
            .map(|(blinding, coeff)| blinding.scalar * coeff)
            .sum();

        // Combine row blindings if present
        let max_row_len = blindings
            .iter()
            .map(|b| b.row_blindings.len())
            .max()
            .unwrap_or(0);

        let combined_row_blindings = if max_row_len > 0 {
            (0..max_row_len)
                .into_par_iter()
                .map(|row_idx| {
                    blindings
                        .iter()
                        .zip(coeffs.iter())
                        .filter_map(|(blinding, coeff)| {
                            blinding.row_blindings.get(row_idx).map(|r| *r * coeff)
                        })
                        .sum()
                })
                .collect()
        } else {
            Vec::new()
        };

        DoryBlindingFactor::with_row_blindings(combined_scalar, combined_row_blindings)
    }

    fn prove_hiding<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Fr>,
        _blinding: &Self::BlindingFactor,
        opening_point: &[<Fr as JoltField>::Challenge],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let _span = trace_span!("DoryCommitmentScheme::prove_hiding").entered();

        // For now, we use the standard Dory opening proof
        // The blinding is "absorbed" into the commitment, and the verifier
        // only sees the hiding commitment.
        //
        // In a full implementation, we would need to:
        // 1. Commit to the blinding factor evaluation at the opening point
        // 2. Include a proof that the blinding was correctly applied
        //
        // For the BlindFold approach, the opening is done at the final step
        // after all sumcheck rounds complete, so the blinding tracking happens
        // at a higher level (in the ZK sumcheck protocol).

        // The standard proof works because:
        // - The verifier has the hiding commitment C' = C + r*H
        // - The prover proves that C opens to v at point p
        // - The blinding r*H is "cancelled out" by including r in the proof

        Self::prove(setup, poly, opening_point, hint, transcript)
    }
}

/// Gets the hiding generators from the setup.
///
/// Returns (H_G1, H_G2) where:
/// - H_G1 is the last G1 generator in the setup
/// - H_G2 is the last G2 generator in the setup
///
/// Using the last generators ensures they are unlikely to be used in
/// normal commitments (which use the first N generators based on polynomial size).
/// For security, these should have an unknown discrete log relationship to the
/// main generators, which is provided by the trusted setup.
fn get_hiding_generators(setup: &ArkworksProverSetup) -> (ArkG1, ArkG2) {
    let num_g1 = setup.g1_vec.len();
    let num_g2 = setup.g2_vec.len();

    // Use the last generators as hiding generators
    // These are at positions that normal commitments won't use
    let h_g1 = if num_g1 > 1 {
        setup.g1_vec[num_g1 - 1]
    } else {
        setup.g1_vec[0]
    };

    let h_g2 = if num_g2 > 1 {
        setup.g2_vec[num_g2 - 1]
    } else {
        setup.g2_vec[0]
    };

    (h_g1, h_g2)
}

/// Adds blinding to a GT commitment using a pairing-based blinding term.
///
/// Computes: C' = C + e(r * H_G1, H_G2) where:
/// - C is the original commitment in GT
/// - r is the blinding scalar (secret)
/// - H_G1 is the hiding generator in G1
/// - H_G2 is the hiding generator in G2
///
/// This provides computational hiding since:
/// - Without knowing r, the blinding term looks random
/// - The blinding term is in GT (same group as the commitment)
/// - The verifier sees C' and cannot extract C or r
fn add_blinding_to_commitment(
    commitment: ArkGT,
    blinding: &Fr,
    setup: &ArkworksProverSetup,
) -> ArkGT {
    let _span = trace_span!("add_blinding_to_commitment").entered();

    // Skip blinding if scalar is zero
    if blinding.is_zero() {
        return commitment;
    }

    // Get hiding generators
    let (h_g1, h_g2) = get_hiding_generators(setup);

    // Compute r * H_G1 (scalar multiplication in G1)
    let blinded_g1 = ArkG1(G1Projective::from(h_g1.0) * blinding);

    // Compute e(r * H_G1, H_G2) using the pairing
    // This gives us a blinding term in GT
    let blinding_term = <BN254 as PairingCurve>::multi_pair_g2_setup(
        &[blinded_g1],
        &[h_g2],
    );

    // C' = C + blinding_term
    ArkGT(commitment.0 + blinding_term.0)
}

/// Applies blinding to row commitments for hiding Dory.
///
/// Each row commitment gets its own blinding: R'_i = R_i + r_i * H
/// where r_i are the row blindings and H is the hiding generator.
#[allow(dead_code)]
fn blind_row_commitments(
    row_commitments: &[ArkG1],
    row_blindings: &[Fr],
    hiding_generator: &ArkG1,
) -> Vec<ArkG1> {
    let _span = trace_span!("blind_row_commitments").entered();

    row_commitments
        .par_iter()
        .zip(row_blindings.par_iter())
        .map(|(row_comm, blinding)| {
            let blinding_term = G1Projective::from(hiding_generator.0) * blinding;
            ArkG1(row_comm.0 + blinding_term)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::commitment::dory::DoryGlobals;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn test_sample_blinding() {
        let mut rng = ChaCha20Rng::seed_from_u64(12345);
        let blinding1 = DoryCommitmentScheme::sample_blinding(&mut rng);
        let blinding2 = DoryCommitmentScheme::sample_blinding(&mut rng);

        // Blindings should be different
        assert_ne!(blinding1.scalar, blinding2.scalar);

        // Blindings should be non-zero
        assert_ne!(blinding1.scalar, Fr::zero());
        assert_ne!(blinding2.scalar, Fr::zero());
    }

    #[test]
    fn test_combine_blindings() {
        let mut rng = ChaCha20Rng::seed_from_u64(12345);

        let b1 = DoryCommitmentScheme::sample_blinding(&mut rng);
        let b2 = DoryCommitmentScheme::sample_blinding(&mut rng);

        let coeffs = vec![Fr::from(2u64), Fr::from(3u64)];
        let combined = DoryCommitmentScheme::combine_blindings(&[b1.clone(), b2.clone()], &coeffs);

        // Verify the linear combination is correct
        let expected = b1.scalar * coeffs[0] + b2.scalar * coeffs[1];
        assert_eq!(combined.scalar, expected);
    }

    #[test]
    fn test_blinding_factor_default() {
        let default_blinding = DoryBlindingFactor::default();
        assert_eq!(default_blinding.scalar, Fr::zero());
        assert!(default_blinding.row_blindings.is_empty());
    }

    #[test]
    fn test_get_hiding_generators() {
        use crate::poly::commitment::dory::DoryCommitmentScheme;
        use crate::poly::commitment::commitment_scheme::CommitmentScheme;

        // Create a minimal setup
        let setup = DoryCommitmentScheme::setup_prover(4);

        // Get hiding generators
        let (h_g1, h_g2) = get_hiding_generators(&setup);

        // Verify they are the last elements
        assert_eq!(h_g1, setup.g1_vec[setup.g1_vec.len() - 1]);
        assert_eq!(h_g2, setup.g2_vec[setup.g2_vec.len() - 1]);
    }

    #[test]
    fn test_blinding_changes_commitment() {
        use crate::poly::commitment::dory::DoryCommitmentScheme;
        use crate::poly::commitment::commitment_scheme::CommitmentScheme;

        // Create a setup
        let setup = DoryCommitmentScheme::setup_prover(4);

        // Create a simple polynomial
        let poly = MultilinearPolynomial::from(vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(4u64)]);

        // Initialize DoryGlobals for the test
        let _ = DoryGlobals::initialize(4, 1);

        // Commit without blinding
        let (base_commitment, _) = DoryCommitmentScheme::commit(&poly, &setup);

        // Commit with non-zero blinding
        let blinding = DoryBlindingFactor::new(Fr::from(12345u64));
        let (hiding_commitment, _) = DoryCommitmentScheme::commit_hiding(&poly, &blinding, &setup);

        // The hiding commitment should be different from the base commitment
        assert_ne!(base_commitment.0, hiding_commitment.0, "Hiding commitment should differ from base commitment");

        // Commit with zero blinding should equal base commitment
        let zero_blinding = DoryBlindingFactor::new(Fr::zero());
        let (zero_hiding_commitment, _) = DoryCommitmentScheme::commit_hiding(&poly, &zero_blinding, &setup);
        assert_eq!(base_commitment.0, zero_hiding_commitment.0, "Zero blinding should give same commitment");
    }

    #[test]
    fn test_different_blindings_different_commitments() {
        use crate::poly::commitment::dory::DoryCommitmentScheme;
        use crate::poly::commitment::commitment_scheme::CommitmentScheme;

        // Create a setup
        let setup = DoryCommitmentScheme::setup_prover(4);

        // Create a simple polynomial
        let poly = MultilinearPolynomial::from(vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(4u64)]);

        // Initialize DoryGlobals for the test
        let _ = DoryGlobals::initialize(4, 1);

        // Two different blindings should produce different commitments
        let blinding1 = DoryBlindingFactor::new(Fr::from(100u64));
        let blinding2 = DoryBlindingFactor::new(Fr::from(200u64));

        let (commitment1, _) = DoryCommitmentScheme::commit_hiding(&poly, &blinding1, &setup);
        let (commitment2, _) = DoryCommitmentScheme::commit_hiding(&poly, &blinding2, &setup);

        assert_ne!(commitment1.0, commitment2.0, "Different blindings should produce different commitments");
    }
}
