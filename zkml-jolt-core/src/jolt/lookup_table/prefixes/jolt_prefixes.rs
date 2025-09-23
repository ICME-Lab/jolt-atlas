use jolt_core::{field::JoltField, utils::lookup_bits::LookupBits};

use jolt_core::zkvm::lookup_table::prefixes::{
    PrefixCheckpoint as JoltPrefixCheckpoint, Prefixes, SparseDensePrefix,
};
pub use jolt_core::zkvm::lookup_table::prefixes::{
    and::AndPrefix, div_by_zero::DivByZeroPrefix, eq::EqPrefix,
    left_is_zero::LeftOperandIsZeroPrefix, left_msb::LeftMsbPrefix, left_shift::LeftShiftPrefix,
    left_shift_helper::LeftShiftHelperPrefix, lower_word::LowerWordPrefix, lsb::LsbPrefix,
    lt::LessThanPrefix, negative_divisor_equals_remainder::NegativeDivisorEqualsRemainderPrefix,
    negative_divisor_greater_than_remainder::NegativeDivisorGreaterThanRemainderPrefix,
    negative_divisor_zero_remainder::NegativeDivisorZeroRemainderPrefix, or::OrPrefix,
    positive_remainder_equals_divisor::PositiveRemainderEqualsDivisorPrefix,
    positive_remainder_less_than_divisor::PositiveRemainderLessThanDivisorPrefix, pow2::Pow2Prefix,
    right_is_zero::RightOperandIsZeroPrefix, right_msb::RightMsbPrefix,
    right_shift::RightShiftPrefix, sign_extension::SignExtensionPrefix,
    upper_word::UpperWordPrefix, xor::XorPrefix,
};

use crate::jolt::lookup_table::prefixes::{AtlasPrefixCheckpoint, AtlasSparseDensePrefix};

impl<F> From<AtlasPrefixCheckpoint<F>> for JoltPrefixCheckpoint<F>
where
    F: JoltField,
{
    fn from(cp: AtlasPrefixCheckpoint<F>) -> Self {
        JoltPrefixCheckpoint::from(cp.0)
    }
}

impl<F> From<JoltPrefixCheckpoint<F>> for AtlasPrefixCheckpoint<F>
where
    F: JoltField,
{
    fn from(cp: JoltPrefixCheckpoint<F>) -> Self {
        AtlasPrefixCheckpoint::from(cp.0)
    }
}

use super::Prefixes as AtlasPrefixes;

impl From<Prefixes> for AtlasPrefixes {
    fn from(prefix: Prefixes) -> Self {
        match prefix {
            Prefixes::And => AtlasPrefixes::And,
            Prefixes::DivByZero => AtlasPrefixes::DivByZero,
            Prefixes::Eq => AtlasPrefixes::Eq,
            Prefixes::LeftOperandIsZero => AtlasPrefixes::LeftOperandIsZero,
            Prefixes::LeftOperandMsb => AtlasPrefixes::LeftOperandMsb,
            Prefixes::LeftShift => AtlasPrefixes::LeftShift,
            Prefixes::LeftShiftHelper => AtlasPrefixes::LeftShiftHelper,
            Prefixes::LessThan => AtlasPrefixes::LessThan,
            Prefixes::LowerWord => AtlasPrefixes::LowerWord,
            Prefixes::Lsb => AtlasPrefixes::Lsb,
            Prefixes::NegativeDivisorEqualsRemainder => {
                AtlasPrefixes::NegativeDivisorEqualsRemainder
            }
            Prefixes::NegativeDivisorGreaterThanRemainder => {
                AtlasPrefixes::NegativeDivisorGreaterThanRemainder
            }
            Prefixes::NegativeDivisorZeroRemainder => AtlasPrefixes::NegativeDivisorZeroRemainder,
            Prefixes::Or => AtlasPrefixes::Or,
            Prefixes::PositiveRemainderEqualsDivisor => {
                AtlasPrefixes::PositiveRemainderEqualsDivisor
            }
            Prefixes::PositiveRemainderLessThanDivisor => {
                AtlasPrefixes::PositiveRemainderLessThanDivisor
            }
            Prefixes::Pow2 => AtlasPrefixes::Pow2,
            Prefixes::RightOperandIsZero => AtlasPrefixes::RightOperandIsZero,
            Prefixes::RightOperandMsb => AtlasPrefixes::RightOperandMsb,
            Prefixes::RightShift => AtlasPrefixes::RightShift,
            Prefixes::SignExtension => AtlasPrefixes::SignExtension,
            Prefixes::UpperWord => AtlasPrefixes::UpperWord,
            Prefixes::Xor => AtlasPrefixes::Xor,
        }
    }
}

macro_rules! impl_sparse_dense_atlas {
    (
        $($prefix_type:ident$(<$generic_const:ident>)?),+ $(,)?
    ) => {
        $(
            impl<$(const $generic_const: usize,)? F: JoltField> AtlasSparseDensePrefix<F> for $prefix_type$(<$generic_const>)? {
                fn prefix_mle(
                    checkpoints: &[AtlasPrefixCheckpoint<F>],
                    r_x: Option<F>,
                    c: u32,
                    b: LookupBits,
                    j: usize,
                ) -> F {
                    let checkpoints: Vec<JoltPrefixCheckpoint<F>> =
                        checkpoints.iter().map(|&cp| cp.into()).collect();
                    let mle =
                        <$prefix_type$(<$generic_const>)? as SparseDensePrefix<F>>::prefix_mle(&checkpoints, r_x, c, b, j);
                    mle
                }

                fn update_prefix_checkpoint(
                    checkpoints: &[AtlasPrefixCheckpoint<F>],
                    r_x: F,
                    r_y: F,
                    j: usize,
                ) -> AtlasPrefixCheckpoint<F> {
                    let checkpoints: Vec<JoltPrefixCheckpoint<F>> =
                        checkpoints.iter().map(|&cp| cp.into()).collect();
                    let cp = <$prefix_type$(<$generic_const>)? as SparseDensePrefix<F>>::update_prefix_checkpoint(
                        &checkpoints,
                        r_x,
                        r_y,
                        j,
                    );
                    cp.into()
                }
            }
        )*
    };
}

impl_sparse_dense_atlas!(
    AndPrefix<WORD_SIZE>,
    DivByZeroPrefix,
    EqPrefix,
    LeftMsbPrefix,
    LeftOperandIsZeroPrefix,
    LeftShiftHelperPrefix,
    LeftShiftPrefix<WORD_SIZE>,
    LessThanPrefix,
    LowerWordPrefix<WORD_SIZE>,
    LsbPrefix<WORD_SIZE>,
    NegativeDivisorEqualsRemainderPrefix,
    NegativeDivisorGreaterThanRemainderPrefix,
    NegativeDivisorZeroRemainderPrefix,
    OrPrefix<WORD_SIZE>,
    PositiveRemainderEqualsDivisorPrefix,
    PositiveRemainderLessThanDivisorPrefix,
    Pow2Prefix<WORD_SIZE>,
    RightMsbPrefix,
    RightOperandIsZeroPrefix,
    RightShiftPrefix,
    SignExtensionPrefix<WORD_SIZE>,
    UpperWordPrefix<WORD_SIZE>,
    XorPrefix<WORD_SIZE>,
);
