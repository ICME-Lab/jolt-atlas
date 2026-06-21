//! Suffix components of lookup table MLE decompositions.
//!
//! Suffixes capture the low-order bits of lookup table inputs and provide efficient
//! MLE evaluation over small boolean hypercubes. During the prefix-suffix sum-check
//! protocol, suffix MLEs are evaluated and combined with prefix contributions to
//! reconstruct the full lookup table evaluation without materializing the entire table.

use crate::{
    field::JoltField,
    lookup_tables::{
        clamp::{CLAMP_OP1_LOWER, CLAMP_OP2_LOWER, CLAMP_OPS_UPPER},
        suffixes::neg_relu::NegReluSuffix,
    },
    utils::lookup_bits::LookupBits,
};
use num_derive::FromPrimitive;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use self::{
    and::AndSuffix, less_than::LessThanSuffix, lower_word_no_msb::LowerWordNoMsbSuffix,
    one::OneSuffix, or::OrSuffix, relu::ReluSuffix, word_lt_bound::WordLtBoundSuffix,
    xor::XorSuffix, zero_gt_bound::ZeroGtBoundSuffix,
};

/// Bitwise AND suffix implementation.
pub mod and;
/// Less-than comparison suffix implementation.
pub mod less_than;
/// Lower word without MSB suffix implementation.
pub mod lower_word_no_msb;
/// Negated ReLU suffix (Relu(-x)): `neg_relu(x) = max(-x, 0)`.
pub mod neg_relu;
/// Constant one suffix implementation.
pub mod one;
/// Bitwise OR suffix implementation.
pub mod or;
/// ReLU activation suffix implementation.
pub mod relu;
pub mod word_lt_bound;
/// Bitwise XOR suffix implementation.
pub mod xor;
/// Suffix that checks all bits with significance >= bound are zero.
pub mod zero_gt_bound;

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
        #[derive(EnumCountMacro, EnumIter, FromPrimitive, Debug, Clone, Copy)]
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
    And                 : AndSuffix,                // Bitwise AND suffix
    LessThan            : LessThanSuffix,           // Less-than comparison suffix
    LowerWordNoMSB      : LowerWordNoMsbSuffix,     // Lower word without MSB suffix
    One                 : OneSuffix,                // Constant one suffix
    Or                  : OrSuffix,                 // Bitwise OR suffix
    Relu                : ReluSuffix,               // ReLU activation suffix
    Xor                 : XorSuffix,                // Bitwise XOR suffix
    NegRelu             : NegReluSuffix,            // Suffix for Relu(-x) table
    OpsWordLtHigh       : OpsWordLtHighSuffix,      // ∑_{i >= HighBound} x_i * 2^i
    Op1WordLtLow        : Op1WordLtLowSuffix,       // ∑_{i >= Op1LowBound} x_i * 2^i
    Op2WordLtLow        : Op2WordLtLowSuffix,       // ∑_{i >= Op2LowBound} x_i * 2^i
    OpsZeroGtHigh       : OpsZeroGtHighSuffix,      // ∀ i >= HighBound, x_i == 0
    Op1ZeroGtLow        : Op1ZeroGtLowSuffix,       // ∀ i >= Op1LowBound, x_i == 0
    Op2ZeroGtLow        : Op2ZeroGtLowSuffix,       // ∀ i >= Op2LowBound, x_i == 0
);

/// Type alias for suffix evaluation results in the field.
pub type SuffixEval<F: JoltField> = F;

// Type aliases for specific suffix implementations with configured parameters.
type OpsWordLtHighSuffix<const XLEN: usize> = WordLtBoundSuffix<XLEN, CLAMP_OPS_UPPER>;
type Op1WordLtLowSuffix<const XLEN: usize> = WordLtBoundSuffix<XLEN, CLAMP_OP1_LOWER>;
type Op2WordLtLowSuffix<const XLEN: usize> = WordLtBoundSuffix<XLEN, CLAMP_OP2_LOWER>;
type OpsZeroGtHighSuffix<const XLEN: usize> = ZeroGtBoundSuffix<XLEN, CLAMP_OPS_UPPER>;
type Op1ZeroGtLowSuffix<const XLEN: usize> = ZeroGtBoundSuffix<XLEN, CLAMP_OP1_LOWER>;
type Op2ZeroGtLowSuffix<const XLEN: usize> = ZeroGtBoundSuffix<XLEN, CLAMP_OP2_LOWER>;
