use jolt_core::utils::lookup_bits::LookupBits;
use jolt_core::zkvm::lookup_table::suffixes::SparseDenseSuffix;

use jolt_suffixes::{
    AndSuffix, DivByZeroSuffix, EqSuffix, GreaterThanSuffix, LeftOperandIsZeroSuffix,
    LeftShiftSuffix, LessThanSuffix, LowerWordSuffix, LsbSuffix, OneSuffix, OrSuffix,
    RightOperandIsZeroSuffix, RightShiftHelperSuffix, RightShiftPaddingSuffix, RightShiftSuffix,
    SignExtensionSuffix, UpperWordSuffix, XorSuffix,
};
use relu::ReluSuffix;

use num_derive::FromPrimitive;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

/// An enum containing all suffixes used by Jolt's instruction lookup tables.
#[repr(u8)]
#[derive(EnumCountMacro, EnumIter, FromPrimitive)]
pub enum Suffixes {
    And,
    DivByZero,
    Eq,
    GreaterThan,
    LeftOperandIsZero,
    LeftShift,
    LessThan,
    LowerWord,
    Lsb,
    One,
    Or,
    // Pow2,
    Relu,
    RightOperandIsZero,
    RightShift,
    RightShiftHelper,
    RightShiftPadding,
    SignExtension,
    UpperWord,
    Xor,
}

pub type SuffixEval<F> = jolt_core::zkvm::lookup_table::suffixes::SuffixEval<F>;

impl Suffixes {
    /// Evaluates the MLE for this suffix on the bitvector `b`, where
    /// `b` represents `b.len()` variables, each assuming a Boolean value.
    pub fn suffix_mle<const WORD_SIZE: usize>(&self, b: LookupBits) -> u32 {
        match self {
            Suffixes::And => AndSuffix::suffix_mle(b),
            Suffixes::DivByZero => DivByZeroSuffix::suffix_mle(b),
            Suffixes::Eq => EqSuffix::suffix_mle(b),
            Suffixes::GreaterThan => GreaterThanSuffix::suffix_mle(b),
            Suffixes::LeftOperandIsZero => LeftOperandIsZeroSuffix::suffix_mle(b),
            Suffixes::LeftShift => LeftShiftSuffix::suffix_mle(b),
            Suffixes::LessThan => LessThanSuffix::suffix_mle(b),
            Suffixes::LowerWord => LowerWordSuffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::Lsb => LsbSuffix::suffix_mle(b),
            Suffixes::One => OneSuffix::suffix_mle(b),
            Suffixes::Or => OrSuffix::suffix_mle(b),
            // Suffixes::Pow2 => Pow2Suffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::Relu => ReluSuffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::RightOperandIsZero => RightOperandIsZeroSuffix::suffix_mle(b),
            Suffixes::RightShift => RightShiftSuffix::suffix_mle(b),
            Suffixes::RightShiftHelper => RightShiftHelperSuffix::suffix_mle(b),
            Suffixes::RightShiftPadding => RightShiftPaddingSuffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::SignExtension => SignExtensionSuffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::UpperWord => UpperWordSuffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::Xor => XorSuffix::suffix_mle(b),
        }
    }
}

mod jolt_suffixes;
mod relu;
