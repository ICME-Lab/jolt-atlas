use jolt_core::{field::JoltField, utils::lookup_bits::LookupBits};

use crate::jolt::lookup_table::prefixes::{
    PrefixCheckpoint as AtlasPrefixCheckpoint, SparseDensePrefix as AtlasSparseDensePrefix,
};
use jolt_core::zkvm::lookup_table::prefixes::{
    PrefixCheckpoint as JoltPrefixCheckpoint, Prefixes as JoltPrefixes,
    SparseDensePrefix as JoltSparseDensePrefix,
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

impl From<JoltPrefixes> for AtlasPrefixes {
    fn from(prefix: JoltPrefixes) -> Self {
        match prefix {
            JoltPrefixes::And => AtlasPrefixes::And,
            JoltPrefixes::DivByZero => AtlasPrefixes::DivByZero,
            JoltPrefixes::Eq => AtlasPrefixes::Eq,
            JoltPrefixes::LeftOperandIsZero => AtlasPrefixes::LeftOperandIsZero,
            JoltPrefixes::LeftOperandMsb => AtlasPrefixes::LeftOperandMsb,
            JoltPrefixes::LeftShift => AtlasPrefixes::LeftShift,
            JoltPrefixes::LeftShiftHelper => AtlasPrefixes::LeftShiftHelper,
            JoltPrefixes::LessThan => AtlasPrefixes::LessThan,
            JoltPrefixes::LowerWord => AtlasPrefixes::LowerWord,
            JoltPrefixes::Lsb => AtlasPrefixes::Lsb,
            JoltPrefixes::NegativeDivisorEqualsRemainder => {
                AtlasPrefixes::NegativeDivisorEqualsRemainder
            }
            JoltPrefixes::NegativeDivisorGreaterThanRemainder => {
                AtlasPrefixes::NegativeDivisorGreaterThanRemainder
            }
            JoltPrefixes::NegativeDivisorZeroRemainder => {
                AtlasPrefixes::NegativeDivisorZeroRemainder
            }
            JoltPrefixes::Or => AtlasPrefixes::Or,
            JoltPrefixes::PositiveRemainderEqualsDivisor => {
                AtlasPrefixes::PositiveRemainderEqualsDivisor
            }
            JoltPrefixes::PositiveRemainderLessThanDivisor => {
                AtlasPrefixes::PositiveRemainderLessThanDivisor
            }
            JoltPrefixes::Pow2 => AtlasPrefixes::Pow2,
            JoltPrefixes::RightOperandIsZero => AtlasPrefixes::RightOperandIsZero,
            JoltPrefixes::RightOperandMsb => AtlasPrefixes::RightOperandMsb,
            JoltPrefixes::RightShift => AtlasPrefixes::RightShift,
            JoltPrefixes::SignExtension => AtlasPrefixes::SignExtension,
            JoltPrefixes::UpperWord => AtlasPrefixes::UpperWord,
            JoltPrefixes::Xor => AtlasPrefixes::Xor,
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
                        <$prefix_type$(<$generic_const>)? as JoltSparseDensePrefix<F>>::prefix_mle(&checkpoints, r_x, c, b, j);
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
                    let cp = <$prefix_type$(<$generic_const>)? as JoltSparseDensePrefix<F>>::update_prefix_checkpoint(
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
