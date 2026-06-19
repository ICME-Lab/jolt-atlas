//! Suffix components of lookup table MLE decompositions.
//!
//! Suffixes capture the low-order bits of lookup table inputs and provide efficient
//! MLE evaluation over small boolean hypercubes. During the prefix-suffix sum-check
//! protocol, suffix MLEs are evaluated and combined with prefix contributions to
//! reconstruct the full lookup table evaluation without materializing the entire table.

use crate::{
    field::JoltField,
    lookup_tables::suffixes::{
        lower_msb_upper_eqo_low::LowerMsbUpperEqoLowSuffix, neg_relu::NegReluSuffix,
        not_lower_msb_upper_eqz::NotLowerMsbUpperEqzSuffix,
        not_lower_msb_upper_eqz_low::NotLowerMsbUpperEqzLowSuffix, word_no_msb::WordNoMsbSuffix,
    },
    utils::lookup_bits::LookupBits,
};
use num_derive::FromPrimitive;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use self::{
    and::AndSuffix, less_than::LessThanSuffix, one::OneSuffix, or::OrSuffix, xor::XorSuffix,
};

/// Bitwise AND suffix implementation.
pub mod and;
/// Less-than comparison suffix implementation.
pub mod less_than;
/// `m * upper_eqo * low` suffix implementation, used in `sat_clamp` decomposition.
pub mod lower_msb_upper_eqo_low;
/// Negated ReLU suffix (Relu(-x)): `neg_relu(x) = max(-x, 0)`.
pub mod neg_relu;
/// `(1-m) * upper_eqz` suffix implementation, used in `sat_clamp` decomposition.
pub mod not_lower_msb_upper_eqz;
/// `(1-m) * upper_eqz * low` suffix implementation, used in `sat_clamp` decomposition.
pub mod not_lower_msb_upper_eqz_low;
/// Constant one suffix implementation.
pub mod one;
/// Bitwise OR suffix implementation.
pub mod or;
/// Lower word without MSB suffix implementation.
pub mod word_no_msb;
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

macro_rules! impl_sparse_dense_suffix {
    ($($name:ident : $suffix:ident),* $(,)?) => {
        /// An enum containing all suffixes used by Jolt's instruction lookup tables.
        #[repr(u8)]
        #[derive(EnumCountMacro, EnumIter, FromPrimitive)]
        pub enum Suffixes {
            $($name),*
        }

        impl Suffixes {
            /// Evaluates the MLE for this suffix on the bitvector `b`, where
            /// `b` represents `b.len()` variables, each assuming a Boolean value.
            pub fn suffix_mle<const XLEN: usize>(&self, b: LookupBits) -> u32 {
                match self {
                    $(Suffixes::$name => $suffix::<XLEN>::suffix_mle(b),)*
                }
            }
        }
    };
}

impl_sparse_dense_suffix!(
    And                     : AndSuffix,                    // Bitwise AND suffix
    LessThan                : LessThanSuffix,               // Less-than comparison suffix
    WordNoMSB               : WordNoMsbSuffix,              // Lower word without MSB suffix
    One                     : OneSuffix,                    // Constant one suffix
    Or                      : OrSuffix,                     // Bitwise OR suffix
    Xor                     : XorSuffix,                    // Bitwise XOR suffix
    NegRelu                 : NegReluSuffix,                // Suffix for Relu(-x) table
    NotLowerMsbUpperEqz     : NotLowerMsbUpperEqzSuffix,    // `(1-m) * upper_eqz` suffix, used in `sat_clamp` decomposition
    NotLowerMsbUpperEqzLow  : NotLowerMsbUpperEqzLowSuffix, // `(1-m) * upper_eqz * low` suffix, used in `sat_clamp` decomposition
    LowerMsbUpperEqoLow     : LowerMsbUpperEqoLowSuffix,    // `m * upper_eqo * low` suffix, used in `sat_clamp` decomposition
);

/// Type alias for suffix evaluation results in the field.
pub type SuffixEval<F: JoltField> = F;
