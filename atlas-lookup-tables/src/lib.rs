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

#![allow(
    clippy::assertions_on_result_states,
    clippy::from_over_into,
    clippy::len_without_is_empty,
    clippy::needless_range_loop,
    clippy::new_without_default,
    clippy::too_long_first_doc_paragraph,
    long_running_const_eval,
    non_snake_case,
    type_alias_bounds
)]

use prefixes::{PrefixEval, Prefixes};
use suffixes::{SuffixEval, Suffixes};

use atlas_algebra::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use serde::Serialize;
use std::fmt::Debug;

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

/// Clamps the input to a specified range: floor-at-0 (`SoftmaxClampTable`) or natively
/// symmetric (`ClampTable`/`SaturationTable`/`ActivationClampTable`), via one generic `ClampSpec`.
pub mod clamp;
/// Piecewise-linear tanh, erf, sigmoid, sin and cos tables with a prefix-suffix decomposition.
/// Not read by any op.
pub mod piecewise_linear;
/// ReLU (Rectified Linear Unit) activation lookup table.
pub mod relu;
/// Unsigned right-shift-by-a-fixed-bound lookup table, used to downscale the Cos/Sin trig
/// tables.
pub mod right_shift;
/// Unsigned absolute value lookup table.
///
/// Computes `abs(x)` by interpreting the input as a signed integer and returning
/// its unsigned magnitude.
pub mod unsigned_abs;
