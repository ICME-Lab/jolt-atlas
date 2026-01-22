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
//! - `hiding_commitment`: NIFS helper functions for commitment folding
//!
//! # References
//!
//! - Nova: Recursive SNARKs without trusted setup
//! - Spartan2: NIFS-based folding
//! - Jolt PR #1205: BlindFold implementation

pub mod blindfold_protocol;
pub mod hiding_commitment;
pub mod nifs;
pub mod random_instance;
pub mod relaxed_r1cs;
pub mod verifier_circuit;

#[cfg(test)]
mod tests;

// Core BlindFold protocol
pub use blindfold_protocol::{
    create_committed_instance, verify_commitment_opening, BlindFoldProof, BlindFoldProtocol,
    HidingBlindFoldProof, HidingBlindFoldProtocol, SumcheckBlindFold,
};

// Hiding commitment helpers
pub use hiding_commitment::ScalarBlindingFactor;

// NIFS folding
pub use nifs::{HidingNIFS, HidingNIFSProof, InstanceBlindingFactors, NIFSProof, NIFS};

// Random instance generation
pub use random_instance::RandomInstanceGenerator;

// Relaxed R1CS types
pub use relaxed_r1cs::{R1CSMatrices, RelaxedR1CSInstance, RelaxedR1CSWitness, SparseMatrix};

// Verifier circuit
pub use verifier_circuit::{VariableIndices, VerifierR1CSCircuit, VerifierWitness};
