//! Stub implementations for removed spartan_interleaved_poly types
//!
//! These types were removed from jolt-core.

/// Stub for SparseCoefficient which was removed from jolt-core
#[derive(Clone, Copy, Debug)]
pub struct SparseCoefficient<T> {
    pub index: usize,
    pub value: T,
}

impl<T> SparseCoefficient<T> {
    pub fn new(index: usize, value: T) -> Self {
        Self { index, value }
    }
}

impl<T: Copy> From<(usize, T)> for SparseCoefficient<T> {
    fn from((index, value): (usize, T)) -> Self {
        Self { index, value }
    }
}
