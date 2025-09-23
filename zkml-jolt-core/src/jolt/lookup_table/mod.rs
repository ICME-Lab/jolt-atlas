pub mod jolt_tables;

use crate::jolt::lookup_table::{
    prefixes::AtlasPrefixEval,
    suffixes::{SuffixEval, Suffixes},
};

use crate::jolt::lookup_table::jolt_tables::{
    AndTable, EqualTable, HalfwordAlignmentTable, MovsignTable, NotEqualTable, OrTable, Pow2Table,
    RangeCheckTable, ShiftRightBitmaskTable, SignedGreaterThanEqualTable, SignedLessThanTable,
    UnsignedGreaterThanEqualTable, UnsignedLessThanEqualTable, UnsignedLessThanTable,
    UpperWordTable, ValidDiv0Table, ValidSignedRemainderTable, ValidUnsignedRemainderTable,
    VirtualRotrTable, VirtualSRATable, VirtualSRLTable, XorTable,
};
use relu::ReLUTable;

use serde::{Deserialize, Serialize};
use std::marker::Sync;
use strum::EnumCount;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use derive_more::From;
use jolt_core::field::JoltField;
use std::fmt::Debug;

pub trait AtlasLookupTable: Clone + Debug + Send + Sync + Serialize {
    /// Materializes the entire lookup table for this instruction (assuming an 8-bit word size).
    #[cfg(test)]
    fn materialize(&self) -> Vec<u64> {
        (0..1 << 16).map(|i| self.materialize_entry(i)).collect()
    }

    /// Materialize the entry at the given `index` in the lookup table for this instruction.
    fn materialize_entry(&self, index: u64) -> u64;

    /// Evaluates the MLE of this lookup table on the given point `r`.
    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F;
}

pub trait AtlasPrefixSuffixDecomposition<const WORD_SIZE: usize>:
    AtlasLookupTable + Default
{
    fn suffixes(&self) -> Vec<Suffixes>;
    fn combine<F: JoltField>(&self, prefixes: &[AtlasPrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F;
    #[cfg(test)]
    fn random_lookup_index(rng: &mut rand::rngs::StdRng) -> u64 {
        rand::RngCore::next_u64(rng)
    }
}

#[cfg(test)]
pub mod test;

pub const NUM_LOOKUP_TABLES: usize = AtlasLookupTables::<32>::COUNT;

#[derive(Copy, Clone, Debug, From, Serialize, Deserialize, EnumIter, EnumCountMacro)]
#[repr(u8)]
pub enum AtlasLookupTables<const WORD_SIZE: usize> {
    RangeCheck(RangeCheckTable<WORD_SIZE>),
    And(AndTable<WORD_SIZE>),
    Or(OrTable<WORD_SIZE>),
    Xor(XorTable<WORD_SIZE>),
    Equal(EqualTable<WORD_SIZE>),
    SignedGreaterThanEqual(SignedGreaterThanEqualTable<WORD_SIZE>),
    UnsignedGreaterThanEqual(UnsignedGreaterThanEqualTable<WORD_SIZE>),
    NotEqual(NotEqualTable<WORD_SIZE>),
    SignedLessThan(SignedLessThanTable<WORD_SIZE>),
    UnsignedLessThan(UnsignedLessThanTable<WORD_SIZE>),
    Movsign(MovsignTable<WORD_SIZE>),
    UpperWord(UpperWordTable<WORD_SIZE>),
    LessThanEqual(UnsignedLessThanEqualTable<WORD_SIZE>),
    ValidSignedRemainder(ValidSignedRemainderTable<WORD_SIZE>),
    ValidUnsignedRemainder(ValidUnsignedRemainderTable<WORD_SIZE>),
    ValidDiv0(ValidDiv0Table<WORD_SIZE>),
    HalfwordAlignment(HalfwordAlignmentTable<WORD_SIZE>),
    Pow2(Pow2Table<WORD_SIZE>),
    ShiftRightBitmask(ShiftRightBitmaskTable<WORD_SIZE>),
    VirtualSRL(VirtualSRLTable<WORD_SIZE>),
    VirtualSRA(VirtualSRATable<WORD_SIZE>),
    VirtualROTRI(VirtualRotrTable<WORD_SIZE>),
    Relu(ReLUTable<WORD_SIZE>),
}

impl<const WORD_SIZE: usize> AtlasLookupTables<WORD_SIZE> {
    pub fn enum_index(table: &Self) -> usize {
        // Discriminant: https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting
        let byte = unsafe { *(table as *const Self as *const u8) };
        byte as usize
    }

    // #[cfg(test)]
    // pub fn materialize(&self) -> Vec<u64> {
    //     match self {
    //         LookupTables::RangeCheck(table) => table.materialize(),
    //         LookupTables::And(table) => table.materialize(),
    //         LookupTables::Or(table) => table.materialize(),
    //         LookupTables::Xor(table) => table.materialize(),
    //         LookupTables::Equal(table) => table.materialize(),
    //         LookupTables::SignedGreaterThanEqual(table) => table.materialize(),
    //         LookupTables::UnsignedGreaterThanEqual(table) => table.materialize(),
    //         LookupTables::NotEqual(table) => table.materialize(),
    //         LookupTables::SignedLessThan(table) => table.materialize(),
    //         LookupTables::UnsignedLessThan(table) => table.materialize(),
    //         LookupTables::Movsign(table) => table.materialize(),
    //         LookupTables::UpperWord(table) => table.materialize(),
    //         LookupTables::LessThanEqual(table) => table.materialize(),
    //         LookupTables::ValidSignedRemainder(table) => table.materialize(),
    //         LookupTables::ValidUnsignedRemainder(table) => table.materialize(),
    //         LookupTables::ValidDiv0(table) => table.materialize(),
    //         LookupTables::HalfwordAlignment(table) => table.materialize(),
    //         LookupTables::Pow2(table) => table.materialize(),
    //         LookupTables::ShiftRightBitmask(table) => table.materialize(),
    //         LookupTables::VirtualSRL(table) => table.materialize(),
    //         LookupTables::VirtualSRA(table) => table.materialize(),
    //         LookupTables::VirtualROTRI(table) => table.materialize(),
    //     }
    // }

    pub fn materialize_entry(&self, index: u64) -> u64 {
        match self {
            AtlasLookupTables::RangeCheck(table) => table.materialize_entry(index),
            AtlasLookupTables::And(table) => table.materialize_entry(index),
            AtlasLookupTables::Or(table) => table.materialize_entry(index),
            AtlasLookupTables::Xor(table) => table.materialize_entry(index),
            AtlasLookupTables::Equal(table) => table.materialize_entry(index),
            AtlasLookupTables::SignedGreaterThanEqual(table) => table.materialize_entry(index),
            AtlasLookupTables::UnsignedGreaterThanEqual(table) => table.materialize_entry(index),
            AtlasLookupTables::NotEqual(table) => table.materialize_entry(index),
            AtlasLookupTables::SignedLessThan(table) => table.materialize_entry(index),
            AtlasLookupTables::UnsignedLessThan(table) => table.materialize_entry(index),
            AtlasLookupTables::Movsign(table) => table.materialize_entry(index),
            AtlasLookupTables::UpperWord(table) => table.materialize_entry(index),
            AtlasLookupTables::LessThanEqual(table) => table.materialize_entry(index),
            AtlasLookupTables::ValidSignedRemainder(table) => table.materialize_entry(index),
            AtlasLookupTables::ValidUnsignedRemainder(table) => table.materialize_entry(index),
            AtlasLookupTables::ValidDiv0(table) => table.materialize_entry(index),
            AtlasLookupTables::HalfwordAlignment(table) => table.materialize_entry(index),
            AtlasLookupTables::Pow2(table) => table.materialize_entry(index),
            AtlasLookupTables::ShiftRightBitmask(table) => table.materialize_entry(index),
            AtlasLookupTables::VirtualSRL(table) => table.materialize_entry(index),
            AtlasLookupTables::VirtualSRA(table) => table.materialize_entry(index),
            AtlasLookupTables::VirtualROTRI(table) => table.materialize_entry(index),
            AtlasLookupTables::Relu(table) => table.materialize_entry(index),
        }
    }

    pub fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        match self {
            AtlasLookupTables::RangeCheck(table) => table.evaluate_mle(r),
            AtlasLookupTables::And(table) => table.evaluate_mle(r),
            AtlasLookupTables::Or(table) => table.evaluate_mle(r),
            AtlasLookupTables::Xor(table) => table.evaluate_mle(r),
            AtlasLookupTables::Equal(table) => table.evaluate_mle(r),
            AtlasLookupTables::SignedGreaterThanEqual(table) => table.evaluate_mle(r),
            AtlasLookupTables::UnsignedGreaterThanEqual(table) => table.evaluate_mle(r),
            AtlasLookupTables::NotEqual(table) => table.evaluate_mle(r),
            AtlasLookupTables::SignedLessThan(table) => table.evaluate_mle(r),
            AtlasLookupTables::UnsignedLessThan(table) => table.evaluate_mle(r),
            AtlasLookupTables::Movsign(table) => table.evaluate_mle(r),
            AtlasLookupTables::UpperWord(table) => table.evaluate_mle(r),
            AtlasLookupTables::LessThanEqual(table) => table.evaluate_mle(r),
            AtlasLookupTables::ValidSignedRemainder(table) => table.evaluate_mle(r),
            AtlasLookupTables::ValidUnsignedRemainder(table) => table.evaluate_mle(r),
            AtlasLookupTables::ValidDiv0(table) => table.evaluate_mle(r),
            AtlasLookupTables::HalfwordAlignment(table) => table.evaluate_mle(r),
            AtlasLookupTables::Pow2(table) => table.evaluate_mle(r),
            AtlasLookupTables::ShiftRightBitmask(table) => table.evaluate_mle(r),
            AtlasLookupTables::VirtualSRL(table) => table.evaluate_mle(r),
            AtlasLookupTables::VirtualSRA(table) => table.evaluate_mle(r),
            AtlasLookupTables::VirtualROTRI(table) => table.evaluate_mle(r),
            AtlasLookupTables::Relu(table) => table.evaluate_mle(r),
        }
    }

    pub fn suffixes(&self) -> Vec<Suffixes> {
        match self {
            AtlasLookupTables::RangeCheck(table) => table.suffixes(),
            AtlasLookupTables::And(table) => table.suffixes(),
            AtlasLookupTables::Or(table) => table.suffixes(),
            AtlasLookupTables::Xor(table) => table.suffixes(),
            AtlasLookupTables::Equal(table) => table.suffixes(),
            AtlasLookupTables::SignedGreaterThanEqual(table) => table.suffixes(),
            AtlasLookupTables::UnsignedGreaterThanEqual(table) => table.suffixes(),
            AtlasLookupTables::NotEqual(table) => table.suffixes(),
            AtlasLookupTables::SignedLessThan(table) => table.suffixes(),
            AtlasLookupTables::UnsignedLessThan(table) => table.suffixes(),
            AtlasLookupTables::Movsign(table) => table.suffixes(),
            AtlasLookupTables::UpperWord(table) => table.suffixes(),
            AtlasLookupTables::LessThanEqual(table) => table.suffixes(),
            AtlasLookupTables::ValidSignedRemainder(table) => table.suffixes(),
            AtlasLookupTables::ValidUnsignedRemainder(table) => table.suffixes(),
            AtlasLookupTables::ValidDiv0(table) => table.suffixes(),
            AtlasLookupTables::HalfwordAlignment(table) => table.suffixes(),
            AtlasLookupTables::Pow2(table) => table.suffixes(),
            AtlasLookupTables::ShiftRightBitmask(table) => table.suffixes(),
            AtlasLookupTables::VirtualSRL(table) => table.suffixes(),
            AtlasLookupTables::VirtualSRA(table) => table.suffixes(),
            AtlasLookupTables::VirtualROTRI(table) => table.suffixes(),
            AtlasLookupTables::Relu(table) => table.suffixes(),
        }
    }

    pub fn combine<F: JoltField>(
        &self,
        prefixes: &[AtlasPrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F {
        match self {
            AtlasLookupTables::RangeCheck(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::And(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::Or(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::Xor(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::Equal(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::SignedGreaterThanEqual(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::UnsignedGreaterThanEqual(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::NotEqual(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::SignedLessThan(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::UnsignedLessThan(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::Movsign(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::UpperWord(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::LessThanEqual(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::ValidSignedRemainder(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::ValidUnsignedRemainder(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::ValidDiv0(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::HalfwordAlignment(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::Pow2(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::ShiftRightBitmask(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::VirtualSRL(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::VirtualSRA(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::VirtualROTRI(table) => table.combine(prefixes, suffixes),
            AtlasLookupTables::Relu(table) => table.combine(prefixes, suffixes),
        }
    }
}

pub mod prefixes;
pub mod suffixes;

pub mod relu;
