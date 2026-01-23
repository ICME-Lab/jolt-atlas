//! Stub implementations for removed utility functions
//!
//! These functions were removed from jolt-core.

use jolt_core::field::JoltField;

/// Stub for unsafe_zero_slice which was removed from jolt-core
///
/// This function zeros out a slice in place.
pub fn unsafe_zero_slice<F: JoltField>(slice: &mut [F]) {
    for elem in slice.iter_mut() {
        *elem = F::zero();
    }
}

/// Stub for small_value constants
pub mod small_value {
    /// Number of small value optimization rounds
    pub const NUM_SVO_ROUNDS: usize = 4;
}
