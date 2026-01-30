use crate::onnx_proof::lookup_tables::{
    and::AndTable,
    or::OrTable,
    prefixes::{PrefixEval, Prefixes},
    suffixes::{SuffixEval, Suffixes},
    xor::XorTable,
};
use derive_more::From;
use joltworks::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter, IntoStaticStr};

#[cfg(test)]
pub mod test;

pub trait JoltLookupTable: Clone + Debug + Send + Sync + Serialize {
    /// Materializes the entire lookup table for this instruction (assuming an 8-bit word size).
    #[cfg(test)]
    fn materialize(&self) -> Vec<u64> {
        (0..1 << 16)
            .map(|i| self.materialize_entry(i as u64))
            .collect()
    }

    /// Materialize the entry at the given `index` in the lookup table for this instruction.
    fn materialize_entry(&self, index: u64) -> u64;

    /// Evaluates the MLE of this lookup table on the given point `r`.
    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>;
}

pub trait PrefixSuffixDecompositionTrait<const XLEN: usize>: JoltLookupTable + Default {
    fn suffixes(&self) -> Vec<Suffixes>;
    fn prefixes(&self) -> Vec<Prefixes>;
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F;

    // TODO: Modify `prefix_suffix_test` to use above `combine` method & then rm this method
    #[cfg(test)]
    fn combine_test<F: JoltField>(
        &self,
        prefixes: &[PrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F;
    #[cfg(test)]
    fn random_lookup_index(rng: &mut rand::rngs::StdRng) -> u64 {
        rand::Rng::gen(rng)
    }
}

pub mod prefixes;
pub mod suffixes;

pub mod and;
pub mod or;
pub mod relu;
pub mod unsigned_less_than;
pub mod xor;

#[derive(
    Copy, Clone, Debug, From, Serialize, Deserialize, EnumIter, EnumCountMacro, IntoStaticStr,
)]
#[repr(u8)]
pub enum LookupTables<const XLEN: usize> {
    And(AndTable<XLEN>),
    Or(OrTable<XLEN>),
    Xor(XorTable<XLEN>),
}

impl<const XLEN: usize> LookupTables<XLEN> {
    pub fn enum_index(table: &Self) -> usize {
        // Discriminant: https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting
        let byte = unsafe { *(table as *const Self as *const u8) };
        byte as usize
    }

    #[cfg(test)]
    pub fn materialize(&self) -> Vec<u64> {
        match self {
            LookupTables::And(table) => table.materialize(),
            LookupTables::Or(table) => table.materialize(),
            LookupTables::Xor(table) => table.materialize(),
        }
    }

    pub fn materialize_entry(&self, index: u64) -> u64 {
        match self {
            LookupTables::And(table) => table.materialize_entry(index),
            LookupTables::Or(table) => table.materialize_entry(index),
            LookupTables::Xor(table) => table.materialize_entry(index),
        }
    }

    pub fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        match self {
            LookupTables::And(table) => table.evaluate_mle(r),
            LookupTables::Or(table) => table.evaluate_mle(r),
            LookupTables::Xor(table) => table.evaluate_mle(r),
        }
    }

    pub fn prefixes(&self) -> Vec<Prefixes> {
        match self {
            LookupTables::And(table) => table.prefixes(),
            LookupTables::Or(table) => table.prefixes(),
            LookupTables::Xor(table) => table.prefixes(),
        }
    }

    pub fn suffixes(&self) -> Vec<Suffixes> {
        match self {
            LookupTables::And(table) => table.suffixes(),
            LookupTables::Or(table) => table.suffixes(),
            LookupTables::Xor(table) => table.suffixes(),
        }
    }

    pub fn combine<F: JoltField>(
        &self,
        prefixes: &[PrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F {
        match self {
            LookupTables::And(table) => table.combine(prefixes, suffixes),
            LookupTables::Or(table) => table.combine(prefixes, suffixes),
            LookupTables::Xor(table) => table.combine(prefixes, suffixes),
        }
    }

    #[cfg(test)]
    pub fn combine_test<F: JoltField>(
        &self,
        prefixes: &[PrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F {
        match self {
            LookupTables::And(table) => table.combine_test(prefixes, suffixes),
            LookupTables::Or(table) => table.combine_test(prefixes, suffixes),
            LookupTables::Xor(table) => table.combine_test(prefixes, suffixes),
        }
    }
}
