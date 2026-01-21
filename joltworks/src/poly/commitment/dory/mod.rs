//! Dory polynomial commitment scheme
//!
//! This module provides a Dory commitment scheme implementation that bridges
//! between Jolt's types and final-dory's arkworks backend.
//!
//! # Hiding (Zero-Knowledge) Support
//!
//! The `hiding_dory` submodule implements the `HidingCommitmentScheme` trait,
//! enabling zero-knowledge sumcheck proofs using the BlindFold approach.

mod commitment_scheme;
mod dory_globals;
pub mod hiding_dory;
mod jolt_dory_routines;
mod wrappers;

#[cfg(test)]
mod tests;

pub use commitment_scheme::DoryCommitmentScheme;
pub use dory_globals::{DoryContext, DoryGlobals};
pub use hiding_dory::DoryBlindingFactor;
pub use jolt_dory_routines::{JoltG1Routines, JoltG2Routines};
pub use wrappers::{
    ArkDoryProof, ArkFr, ArkG1, ArkG2, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup,
    JoltFieldWrapper, BN254,
};
