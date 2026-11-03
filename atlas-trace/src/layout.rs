//! Raw dims and their power-of-two padding.

use serde::{Deserialize, Serialize};

/// How a compiled model's tensors are laid out in memory. Model-wide: one
/// value per [`crate::Model`], set by [`crate::CompileOptions::pad_pow2`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Layout {
    /// Tensors have their raw dims.
    Raw,
    /// Tensors are zero-padded per dimension to powers of two.
    Padded,
}

impl Layout {
    /// The dims a tensor with raw dims `raw` has under this layout.
    pub fn physical(self, raw: &[usize]) -> Vec<usize> {
        match self {
            Layout::Raw => raw.to_vec(),
            Layout::Padded => padded_dims(raw),
        }
    }
}

/// Every dim of `raw` rounded up to the next power of two.
pub fn padded_dims(raw: &[usize]) -> Vec<usize> {
    raw.iter().map(|d| d.next_power_of_two()).collect()
}
