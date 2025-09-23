use super::Suffixes as AtlasSuffixes;
use jolt_core::zkvm::lookup_table::suffixes::Suffixes as JoltSuffixes;

pub use jolt_core::zkvm::lookup_table::suffixes::{
    and::AndSuffix, div_by_zero::DivByZeroSuffix, eq::EqSuffix, gt::GreaterThanSuffix,
    left_is_zero::LeftOperandIsZeroSuffix, left_shift::LeftShiftSuffix,
    lower_word::LowerWordSuffix, lsb::LsbSuffix, lt::LessThanSuffix, one::OneSuffix, or::OrSuffix,
    pow2::Pow2Suffix, right_is_zero::RightOperandIsZeroSuffix, right_shift::RightShiftSuffix,
    right_shift_helper::RightShiftHelperSuffix, right_shift_padding::RightShiftPaddingSuffix,
    sign_extension::SignExtensionSuffix, upper_word::UpperWordSuffix, xor::XorSuffix,
};

impl From<JoltSuffixes> for AtlasSuffixes {
    fn from(suffix: JoltSuffixes) -> Self {
        match suffix {
            JoltSuffixes::And => AtlasSuffixes::And,
            JoltSuffixes::DivByZero => AtlasSuffixes::DivByZero,
            JoltSuffixes::Eq => AtlasSuffixes::Eq,
            JoltSuffixes::GreaterThan => AtlasSuffixes::GreaterThan,
            JoltSuffixes::LeftOperandIsZero => AtlasSuffixes::LeftOperandIsZero,
            JoltSuffixes::LeftShift => AtlasSuffixes::LeftShift,
            JoltSuffixes::LessThan => AtlasSuffixes::LessThan,
            JoltSuffixes::LowerWord => AtlasSuffixes::LowerWord,
            JoltSuffixes::Lsb => AtlasSuffixes::Lsb,
            JoltSuffixes::One => AtlasSuffixes::One,
            JoltSuffixes::Or => AtlasSuffixes::Or,
            JoltSuffixes::Pow2 => AtlasSuffixes::Pow2,
            JoltSuffixes::RightOperandIsZero => AtlasSuffixes::RightOperandIsZero,
            JoltSuffixes::RightShift => AtlasSuffixes::RightShift,
            JoltSuffixes::RightShiftHelper => AtlasSuffixes::RightShiftHelper,
            JoltSuffixes::RightShiftPadding => AtlasSuffixes::RightShiftPadding,
            JoltSuffixes::SignExtension => AtlasSuffixes::SignExtension,
            JoltSuffixes::UpperWord => AtlasSuffixes::UpperWord,
            JoltSuffixes::Xor => AtlasSuffixes::Xor,
        }
    }
}
