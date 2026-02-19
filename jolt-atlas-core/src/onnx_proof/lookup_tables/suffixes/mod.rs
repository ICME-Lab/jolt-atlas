//! Suffix components of lookup table MLE decompositions.
//!
//! Suffixes capture the low-order bits of lookup table inputs and provide efficient
//! MLE evaluation over small boolean hypercubes. During the prefix-suffix sum-check
//! protocol, suffix MLEs are evaluated and combined with prefix contributions to
//! reconstruct the full lookup table evaluation without materializing the entire table.

use joltworks::{field::JoltField, utils::lookup_bits::LookupBits};
use num_derive::FromPrimitive;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use crate::onnx_proof::lookup_tables::suffixes::{
    and::AndSuffix, less_than::LessThanSuffix, lower_word_no_msb::LowerWordNoMsbSuffix,
    one::OneSuffix, or::OrSuffix, relu::ReluSuffix, xor::XorSuffix,
};

/// Bitwise AND suffix implementation.
pub mod and;
/// Less-than comparison suffix implementation.
pub mod less_than;
/// Lower word without MSB suffix implementation.
pub mod lower_word_no_msb;
/// Constant one suffix implementation.
pub mod one;
/// Bitwise OR suffix implementation.
pub mod or;
/// ReLU activation suffix implementation.
pub mod relu;
/// Bitwise XOR suffix implementation.
pub mod xor;

/// Trait for suffix components that support sparse-dense MLE evaluation.
///
/// Suffixes evaluate MLEs efficiently over small boolean hypercubes representing
/// the low-order bits of lookup table inputs.
pub trait SparseDenseSuffix: 'static + Sync {
    /// Evaluates the MLE for this suffix on the bitvector `b`, where
    /// `b` represents `b.len()` variables, each assuming a Boolean value.
    fn suffix_mle(b: LookupBits) -> u32;
}

/// An enum containing all suffixes used by Jolt's instruction lookup tables.
#[repr(u8)]
#[derive(EnumCountMacro, EnumIter, FromPrimitive)]
pub enum Suffixes {
    /// Bitwise AND suffix
    And,
    /// Less-than comparison suffix
    LessThan,
    /// Lower word without MSB suffix
    LowerWordNoMSB,
    /// Constant one suffix
    One,
    /// Bitwise OR suffix
    Or,
    /// ReLU activation suffix
    Relu,
    /// Bitwise XOR suffix
    Xor,
}

/// Type alias for suffix evaluation results in the field.
pub type SuffixEval<F: JoltField> = F;

impl Suffixes {
    /// Evaluates the MLE for this suffix on the bitvector `b`, where
    /// `b` represents `b.len()` variables, each assuming a Boolean value.
    pub fn suffix_mle<const XLEN: usize>(&self, b: LookupBits) -> u32 {
        match self {
            Suffixes::And => AndSuffix::suffix_mle(b),
            Suffixes::One => OneSuffix::suffix_mle(b),
            Suffixes::Or => OrSuffix::suffix_mle(b),
            Suffixes::LessThan => LessThanSuffix::suffix_mle(b),
            Suffixes::LowerWordNoMSB => LowerWordNoMsbSuffix::<XLEN>::suffix_mle(b),
            Suffixes::Relu => ReluSuffix::<XLEN>::suffix_mle(b),
            Suffixes::Xor => XorSuffix::suffix_mle(b),
        }
    }
}
