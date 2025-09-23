pub use jolt_core::zkvm::lookup_table::suffixes::{
    Suffixes as JoltSuffixes, and::AndSuffix, div_by_zero::DivByZeroSuffix, eq::EqSuffix,
    gt::GreaterThanSuffix, left_is_zero::LeftOperandIsZeroSuffix, left_shift::LeftShiftSuffix,
    lower_word::LowerWordSuffix, lsb::LsbSuffix, lt::LessThanSuffix, one::OneSuffix, or::OrSuffix,
    pow2::Pow2Suffix, right_is_zero::RightOperandIsZeroSuffix, right_shift::RightShiftSuffix,
    right_shift_helper::RightShiftHelperSuffix, right_shift_padding::RightShiftPaddingSuffix,
    sign_extension::SignExtensionSuffix, upper_word::UpperWordSuffix, xor::XorSuffix,
};

use super::Suffixes;

impl From<JoltSuffixes> for Suffixes {
    fn from(suffix: JoltSuffixes) -> Self {
        match suffix {
            JoltSuffixes::And => Suffixes::And,
            JoltSuffixes::DivByZero => Suffixes::DivByZero,
            JoltSuffixes::Eq => Suffixes::Eq,
            JoltSuffixes::GreaterThan => Suffixes::GreaterThan,
            JoltSuffixes::LeftOperandIsZero => Suffixes::LeftOperandIsZero,
            JoltSuffixes::LeftShift => Suffixes::LeftShift,
            JoltSuffixes::LessThan => Suffixes::LessThan,
            JoltSuffixes::LowerWord => Suffixes::LowerWord,
            JoltSuffixes::Lsb => Suffixes::Lsb,
            JoltSuffixes::One => Suffixes::One,
            JoltSuffixes::Or => Suffixes::Or,
            JoltSuffixes::Pow2 => Suffixes::Pow2,
            JoltSuffixes::RightOperandIsZero => Suffixes::RightOperandIsZero,
            JoltSuffixes::RightShift => Suffixes::RightShift,
            JoltSuffixes::RightShiftHelper => Suffixes::RightShiftHelper,
            JoltSuffixes::RightShiftPadding => Suffixes::RightShiftPadding,
            JoltSuffixes::SignExtension => Suffixes::SignExtension,
            JoltSuffixes::UpperWord => Suffixes::UpperWord,
            JoltSuffixes::Xor => Suffixes::Xor,
        }
    }
}
