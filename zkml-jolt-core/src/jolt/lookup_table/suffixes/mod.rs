use jolt_core::utils::lookup_bits::LookupBits;
use jolt_core::zkvm::lookup_table::suffixes::SparseDenseSuffix;

use jolt_core::zkvm::lookup_table::suffixes::{
    Suffixes as JoltSuffixes, and::AndSuffix, div_by_zero::DivByZeroSuffix, eq::EqSuffix,
    gt::GreaterThanSuffix, left_is_zero::LeftOperandIsZeroSuffix, left_shift::LeftShiftSuffix,
    lower_word::LowerWordSuffix, lsb::LsbSuffix, lt::LessThanSuffix, one::OneSuffix, or::OrSuffix,
    pow2::Pow2Suffix, right_is_zero::RightOperandIsZeroSuffix, right_shift::RightShiftSuffix,
    right_shift_helper::RightShiftHelperSuffix, right_shift_padding::RightShiftPaddingSuffix,
    sign_extension::SignExtensionSuffix, upper_word::UpperWordSuffix, xor::XorSuffix,
};
use relu::ReluSuffix;

use num_derive::FromPrimitive;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

// pub trait SparseDenseSuffix: 'static + Sync {
//     /// Evaluates the MLE for this suffix on the bitvector `b`, where
//     /// `b` represents `b.len()` variables, each assuming a Boolean value.
//     fn suffix_mle(b: LookupBits) -> u32;
// }

/// An enum containing all suffixes used by Jolt's instruction lookup tables.
#[repr(u8)]
#[derive(EnumCountMacro, EnumIter, FromPrimitive)]
pub enum Suffixes {
    One,
    And,
    Xor,
    Or,
    UpperWord,
    LowerWord,
    LessThan,
    GreaterThan,
    Eq,
    LeftOperandIsZero,
    RightOperandIsZero,
    Lsb,
    DivByZero,
    Pow2,
    RightShiftPadding,
    RightShift,
    RightShiftHelper,
    SignExtension,
    LeftShift,
    Relu,
}

pub use jolt_core::zkvm::lookup_table::suffixes::SuffixEval;

impl From<JoltSuffixes> for Suffixes {
    fn from(suffix: JoltSuffixes) -> Self {
        match suffix {
            JoltSuffixes::One => Suffixes::One,
            JoltSuffixes::And => Suffixes::And,
            JoltSuffixes::Xor => Suffixes::Xor,
            JoltSuffixes::Or => Suffixes::Or,
            JoltSuffixes::UpperWord => Suffixes::UpperWord,
            JoltSuffixes::LowerWord => Suffixes::LowerWord,
            JoltSuffixes::LessThan => Suffixes::LessThan,
            JoltSuffixes::GreaterThan => Suffixes::GreaterThan,
            JoltSuffixes::Eq => Suffixes::Eq,
            JoltSuffixes::LeftOperandIsZero => Suffixes::LeftOperandIsZero,
            JoltSuffixes::RightOperandIsZero => Suffixes::RightOperandIsZero,
            JoltSuffixes::Lsb => Suffixes::Lsb,
            JoltSuffixes::DivByZero => Suffixes::DivByZero,
            JoltSuffixes::Pow2 => Suffixes::Pow2,
            JoltSuffixes::RightShiftPadding => Suffixes::RightShiftPadding,
            JoltSuffixes::RightShift => Suffixes::RightShift,
            JoltSuffixes::RightShiftHelper => Suffixes::RightShiftHelper,
            JoltSuffixes::SignExtension => Suffixes::SignExtension,
            JoltSuffixes::LeftShift => Suffixes::LeftShift,
        }
    }
}

// pub type SuffixEval<F: JoltField> = F;

impl Suffixes {
    /// Evaluates the MLE for this suffix on the bitvector `b`, where
    /// `b` represents `b.len()` variables, each assuming a Boolean value.
    pub fn suffix_mle<const WORD_SIZE: usize>(&self, b: LookupBits) -> u32 {
        match self {
            Suffixes::One => OneSuffix::suffix_mle(b),
            Suffixes::And => AndSuffix::suffix_mle(b),
            Suffixes::Or => OrSuffix::suffix_mle(b),
            Suffixes::Xor => XorSuffix::suffix_mle(b),
            Suffixes::UpperWord => UpperWordSuffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::LowerWord => LowerWordSuffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::LessThan => LessThanSuffix::suffix_mle(b),
            Suffixes::GreaterThan => GreaterThanSuffix::suffix_mle(b),
            Suffixes::Eq => EqSuffix::suffix_mle(b),
            Suffixes::LeftOperandIsZero => LeftOperandIsZeroSuffix::suffix_mle(b),
            Suffixes::RightOperandIsZero => RightOperandIsZeroSuffix::suffix_mle(b),
            Suffixes::Lsb => LsbSuffix::suffix_mle(b),
            Suffixes::DivByZero => DivByZeroSuffix::suffix_mle(b),
            Suffixes::Pow2 => Pow2Suffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::RightShiftPadding => RightShiftPaddingSuffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::RightShift => RightShiftSuffix::suffix_mle(b),
            Suffixes::RightShiftHelper => RightShiftHelperSuffix::suffix_mle(b),
            Suffixes::SignExtension => SignExtensionSuffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::LeftShift => LeftShiftSuffix::suffix_mle(b),
            Suffixes::Relu => ReluSuffix::<WORD_SIZE>::suffix_mle(b),
        }
    }
}

pub mod relu;
