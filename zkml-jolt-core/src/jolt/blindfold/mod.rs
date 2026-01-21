//! BlindFold Module for Zero-Knowledge Sumcheck
//!
//! This module implements the BlindFold approach for zero-knowledge proofs,
//! which combines ZK sumcheck with Nova-style NIFS (Non-Interactive Folding Scheme).
//!
//! # Overview
//!
//! The BlindFold approach achieves zero-knowledge by:
//! 1. Committing to round polynomials instead of sending plaintext coefficients
//! 2. Using a succinct verifier R1CS circuit with NIFS folding
//! 3. Folding real instances with random satisfying instances to hide witnesses
//!
//! # Module Structure
//!
//! - `relaxed_r1cs`: Relaxed R1CS representation (Az ∘ Bz = u·Cz + E)
//! - `nifs`: Non-Interactive Folding Scheme for combining instances
//! - `verifier_circuit`: Succinct verifier R1CS circuit encoding sumcheck checks
//! - `random_instance`: Generator for random satisfying instances
//! - `hiding_commitment`: Hiding (zero-knowledge) commitment scheme
//!
//! # References
//!
//! - Nova: Recursive SNARKs without trusted setup
//! - Spartan2: NIFS-based folding
//! - Jolt PR #1205: BlindFold implementation

pub mod hiding_commitment;
pub mod nifs;
pub mod random_instance;
pub mod relaxed_r1cs;
pub mod verifier_circuit;

#[cfg(test)]
mod tests;

pub use hiding_commitment::{
    HidingWrapper, MockHidingCommitment, MockHidingScheme, ScalarBlindingFactor,
    WrapperBlindingFactor,
};
pub use nifs::{HidingNIFS, HidingNIFSProof, InstanceBlindingFactors, NIFSProof, NIFS};
pub use random_instance::RandomInstanceGenerator;
pub use relaxed_r1cs::{RelaxedR1CSInstance, RelaxedR1CSWitness};
pub use verifier_circuit::VerifierR1CSCircuit;
