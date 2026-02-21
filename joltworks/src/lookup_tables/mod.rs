//! Lookup tables with prefix-suffix MLE decomposition for efficient proving.
//!
//! This module implements lookup tables used in ONNX proof generation. Each lookup table
//! stores precomputed values that can be queried during proof construction. To efficiently
//! prove reads into these tables, we use a **prefix-suffix decomposition** strategy:
//!
//! ## Prefix-Suffix Decomposition
//!
//! Rather than materializing and summing over the full lookup table (which would require
//! iterating over 2^(2*XLEN) entries), we decompose the lookup table's multilinear extension
//! (MLE) into prefix and suffix components:
//!
//! - **Prefixes**: Capture high-order bits and accumulate contributions during sum-check
//! - **Suffixes**: Capture low-order bits with efficient MLE evaluation
//!
//! The full lookup table MLE at a point can be reconstructed by combining the prefix and
//! suffix evaluations using the [`PrefixSuffixDecompositionTrait::combine`] method.
//!
//! ## Sum-Check Protocol
//!
//! During the prefix-suffix read-checking sum-check protocol, we prove correct reads from
//! the lookup tables by:
//! 1. Evaluating prefix MLEs using checkpoints updated every two rounds
//! 2. Evaluating suffix MLEs over smaller boolean hypercubes
//! 3. Combining these evaluations to reconstruct the full table value
//!
//! This decomposition dramatically reduces prover complexity by avoiding iteration over
//! the full 2^(2*XLEN) table.

use and::AndTable;
use or::OrTable;
use prefixes::{PrefixEval, Prefixes};
use suffixes::{SuffixEval, Suffixes};
use xor::XorTable;

use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use derive_more::From;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter, IntoStaticStr};

#[cfg(test)]
/// Test utilities for verifying lookup table correctness.
pub mod test;

/// Core trait for all Jolt lookup tables.
///
/// Provides methods to materialize table entries and evaluate the multilinear
/// extension (MLE) of the lookup table at a given point.
pub trait JoltLookupTable: Clone + Debug + Send + Sync + Serialize {
    /// Materializes the entire lookup table for this instruction (assuming an 8-bit word size).
    #[cfg(test)]
    fn materialize(&self) -> Vec<u64> {
        (0..1 << 16)
            .map(|i| self.materialize_entry(i as u64))
            .collect()
    }

    /// Materialize the entry at the given `index` in the lookup table for this instruction.
    fn materialize_entry(&self, index: u64) -> u64;

    /// Evaluates the MLE of this lookup table on the given point `r`.
    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>;
}

/// Trait for lookup tables that support prefix-suffix MLE decomposition.
///
/// This trait extends [`JoltLookupTable`] with methods to decompose the table's MLE
/// into prefix and suffix components. During the sum-check protocol for proving reads,
/// the prover evaluates prefix_mles and suffix_mles separately and combines them,
/// avoiding iteration over the full 2^(2*XLEN) lookup table.
pub trait PrefixSuffixDecompositionTrait<const XLEN: usize>: JoltLookupTable + Default {
    /// Returns the suffix components used in this lookup table's decomposition.
    fn suffixes(&self) -> Vec<Suffixes>;
    /// Returns the prefix components used in this lookup table's decomposition.
    fn prefixes(&self) -> Vec<Prefixes>;
    /// Combines prefix and suffix evaluations to reconstruct the full lookup table MLE value.
    ///
    /// This method implements the recombination logic that reconstructs the full table
    /// evaluation from its decomposed prefix and suffix MLE evaluations.
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F;

    // TODO: Modify `prefix_suffix_test` to use above `combine` method & then rm this method
    /// Test-only version of combine method.
    #[cfg(test)]
    fn combine_test<F: JoltField>(
        &self,
        prefixes: &[PrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F;
    /// Generates a random lookup table index for testing.
    #[cfg(test)]
    fn random_lookup_index(rng: &mut rand::rngs::StdRng) -> u64 {
        rand::Rng::gen(rng)
    }
}

/// Prefix components for lookup table MLE decomposition.
pub mod prefixes;
/// Suffix components for lookup table MLE decomposition.
pub mod suffixes;

/// Bitwise AND lookup table.
pub mod and;
/// Bitwise OR lookup table.
pub mod or;
/// ReLU (Rectified Linear Unit) activation lookup table.
pub mod relu;
/// Unsigned less-than comparison lookup table.
pub mod unsigned_less_than;
/// Bitwise XOR lookup table.
pub mod xor;

/// Enum of all available lookup tables in the ONNX proof system.
///
/// Each variant corresponds to a specific operation (AND, OR, XOR) and contains
/// the associated lookup table implementation.
#[derive(
    Copy, Clone, Debug, From, Serialize, Deserialize, EnumIter, EnumCountMacro, IntoStaticStr,
)]
#[repr(u8)]
pub enum LookupTables<const XLEN: usize> {
    /// Bitwise AND operation lookup table
    And(AndTable<XLEN>),
    /// Bitwise OR operation lookup table
    Or(OrTable<XLEN>),
    /// Bitwise XOR operation lookup table
    Xor(XorTable<XLEN>),
}

impl<const XLEN: usize> LookupTables<XLEN> {
    /// Returns the discriminant index of the given lookup table variant.
    pub fn enum_index(table: &Self) -> usize {
        // Discriminant: https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting
        let byte = unsafe { *(table as *const Self as *const u8) };
        byte as usize
    }

    /// Materializes the entire lookup table (test only).
    #[cfg(test)]
    pub fn materialize(&self) -> Vec<u64> {
        match self {
            LookupTables::And(table) => table.materialize(),
            LookupTables::Or(table) => table.materialize(),
            LookupTables::Xor(table) => table.materialize(),
        }
    }

    /// Materializes a single entry of the lookup table at the given index.
    pub fn materialize_entry(&self, index: u64) -> u64 {
        match self {
            LookupTables::And(table) => table.materialize_entry(index),
            LookupTables::Or(table) => table.materialize_entry(index),
            LookupTables::Xor(table) => table.materialize_entry(index),
        }
    }

    /// Evaluates the multilinear extension (MLE) of the lookup table at point `r`.
    pub fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        match self {
            LookupTables::And(table) => table.evaluate_mle(r),
            LookupTables::Or(table) => table.evaluate_mle(r),
            LookupTables::Xor(table) => table.evaluate_mle(r),
        }
    }

    /// Returns the prefix components used in this lookup table's decomposition.
    pub fn prefixes(&self) -> Vec<Prefixes> {
        match self {
            LookupTables::And(table) => table.prefixes(),
            LookupTables::Or(table) => table.prefixes(),
            LookupTables::Xor(table) => table.prefixes(),
        }
    }

    /// Returns the suffix components used in this lookup table's decomposition.
    pub fn suffixes(&self) -> Vec<Suffixes> {
        match self {
            LookupTables::And(table) => table.suffixes(),
            LookupTables::Or(table) => table.suffixes(),
            LookupTables::Xor(table) => table.suffixes(),
        }
    }

    /// Combines prefix and suffix evaluations to reconstruct the full lookup table MLE value.
    pub fn combine<F: JoltField>(
        &self,
        prefixes: &[PrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F {
        match self {
            LookupTables::And(table) => table.combine(prefixes, suffixes),
            LookupTables::Or(table) => table.combine(prefixes, suffixes),
            LookupTables::Xor(table) => table.combine(prefixes, suffixes),
        }
    }

    /// Test-only version of combine method.
    #[cfg(test)]
    pub fn combine_test<F: JoltField>(
        &self,
        prefixes: &[PrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F {
        match self {
            LookupTables::And(table) => table.combine_test(prefixes, suffixes),
            LookupTables::Or(table) => table.combine_test(prefixes, suffixes),
            LookupTables::Xor(table) => table.combine_test(prefixes, suffixes),
        }
    }
}
