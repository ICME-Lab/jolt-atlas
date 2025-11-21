use jolt_core::field::JoltField;
use strum::{EnumCount, IntoEnumIterator};

// Export from jolt and implement Atlas traits
pub use jolt_core::zkvm::lookup_table::{
    and::AndTable, equal::EqualTable, halfword_alignment::HalfwordAlignmentTable,
    movsign::MovsignTable, not_equal::NotEqualTable, or::OrTable, pow2::Pow2Table,
    range_check::RangeCheckTable, shift_right_bitmask::ShiftRightBitmaskTable,
    signed_greater_than_equal::SignedGreaterThanEqualTable, signed_less_than::SignedLessThanTable,
    unsigned_greater_than_equal::UnsignedGreaterThanEqualTable,
    unsigned_less_than::UnsignedLessThanTable,
    unsigned_less_than_equal::UnsignedLessThanEqualTable, upper_word::UpperWordTable,
    valid_div0::ValidDiv0Table, valid_signed_remainder::ValidSignedRemainderTable,
    valid_unsigned_remainder::ValidUnsignedRemainderTable, virtual_rotr::VirtualRotrTable,
    virtual_sra::VirtualSRATable, virtual_srl::VirtualSRLTable, xor::XorTable,
};

use crate::jolt::lookup_table::{
    AtlasLookupTable, PrefixEval as AtlasPrefixEval,
    PrefixSuffixDecomposition as AtlasPrefixSuffixDecomposition, SuffixEval as AtlasSuffixEval,
    prefixes::Prefixes as AtlasPrefixes, suffixes::Suffixes as AtlasSuffixes,
};
use jolt_core::zkvm::lookup_table::{
    JoltLookupTable, PrefixSuffixDecomposition as JoltPrefixSuffixDecomposition,
    prefixes::{PrefixEval as JoltPrefixEval, Prefixes as JoltPrefixes},
};

macro_rules! impl_atlas_traits {
    (
        $($table_type:ident<$generic_const:ident>), + $(,)?
    ) => {
        $(
            impl<const $generic_const: usize> AtlasLookupTable for $table_type<$generic_const> {
                fn materialize_entry(&self, index: u64) -> u64 {
                    JoltLookupTable::materialize_entry(self, index)
                }

                fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
                    JoltLookupTable::evaluate_mle(self, r)
                }
            }

            impl<const $generic_const: usize> AtlasPrefixSuffixDecomposition<$generic_const> for $table_type<$generic_const> {
                fn suffixes(&self) -> Vec<AtlasSuffixes> {
                    <Self as JoltPrefixSuffixDecomposition<$generic_const>>::suffixes(self).into_iter().map(|s| AtlasSuffixes::try_from(s).unwrap()).collect()
                }

                fn combine<F: JoltField>(&self, prefixes: &[AtlasPrefixEval<F>], suffixes: &[AtlasSuffixEval<F>]) -> F {
                    let prefixes = map_atlas_prefixes_to_jolt(prefixes);
                   <Self as JoltPrefixSuffixDecomposition<$generic_const>>::combine(self, &prefixes, suffixes)
                }
            }
        )*
    };
}

impl_atlas_traits!(
    AndTable<WORD_SIZE>,
    EqualTable<WORD_SIZE>,
    HalfwordAlignmentTable<WORD_SIZE>,
    MovsignTable<WORD_SIZE>,
    NotEqualTable<WORD_SIZE>,
    OrTable<WORD_SIZE>,
    Pow2Table<WORD_SIZE>,
    RangeCheckTable<WORD_SIZE>,
    ShiftRightBitmaskTable<WORD_SIZE>,
    SignedGreaterThanEqualTable<WORD_SIZE>,
    SignedLessThanTable<WORD_SIZE>,
    UnsignedGreaterThanEqualTable<WORD_SIZE>,
    UnsignedLessThanEqualTable<WORD_SIZE>,
    UnsignedLessThanTable<WORD_SIZE>,
    ValidDiv0Table<WORD_SIZE>,
    ValidUnsignedRemainderTable<WORD_SIZE>,
    ValidSignedRemainderTable<WORD_SIZE>,
    VirtualRotrTable<WORD_SIZE>,
    VirtualSRATable<WORD_SIZE>,
    VirtualSRLTable<WORD_SIZE>,
    UpperWordTable<WORD_SIZE>,
    XorTable<WORD_SIZE>,
);

/// Remap PrefixEval array to Jolt's ordering
fn map_atlas_prefixes_to_jolt<F: JoltField>(
    atlas_prefixes: &[AtlasPrefixEval<F>],
) -> Vec<JoltPrefixEval<F>> {
    let mut jolt_prefixes: Vec<JoltPrefixEval<F>> = vec![F::zero().into(); JoltPrefixes::COUNT];
    for cp in JoltPrefixes::iter() {
        if let Ok(atlas_cp) = AtlasPrefixes::try_from(&cp) {
            jolt_prefixes[cp as usize] = atlas_prefixes[atlas_cp].into();
        }
    }
    jolt_prefixes
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;

    use super::*;

    #[test]
    fn test_prefix_mapping() {
        let mut atlas_prefix: Vec<AtlasPrefixEval<Fr>> =
            vec![Fr::from_u64(0).into(); AtlasPrefixes::COUNT];

        for (i, cp) in AtlasPrefixes::iter().enumerate() {
            atlas_prefix[cp as usize] = Fr::from(i as u64).into();
        }
        let jolt_prefix = map_atlas_prefixes_to_jolt(&atlas_prefix);

        for cp in JoltPrefixes::iter() {
            if let Ok(atlas_cp) = AtlasPrefixes::try_from(&cp) {
                assert_eq!(jolt_prefix[cp as usize].0, Fr::from(atlas_cp as u64));
            }
        }
    }
}
