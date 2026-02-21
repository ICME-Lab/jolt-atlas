use serde::{Deserialize, Serialize};

use super::{
    prefixes::{PrefixEval, Prefixes},
    suffixes::{SuffixEval, Suffixes},
    JoltLookupTable, PrefixSuffixDecompositionTrait,
};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::uninterleave_bits,
};

/// Lookup table for bitwise OR operations.
///
/// Implements OR(x, y) = x | y using prefix-suffix decomposition.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct OrTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for OrTable<XLEN> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        (x | y).into()
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);

        let mut result = F::zero();
        for i in 0..XLEN {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            result += F::from_u64(1u64 << (XLEN - 1 - i)) * (x_i + y_i - x_i * y_i);
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecompositionTrait<XLEN> for OrTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Or]
    }

    fn prefixes(&self) -> Vec<Prefixes> {
        vec![Prefixes::Or]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one_suffix, or_suffix] = suffixes.try_into().unwrap();
        let [or_prefix] = prefixes.try_into().unwrap();
        or_prefix * one_suffix + or_suffix
    }

    #[cfg(test)]
    fn combine_test<F: JoltField>(
        &self,
        prefixes: &[PrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, or] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Or] * one + or
    }
}

#[cfg(test)]
mod test {
    use super::OrTable;
    use crate::lookup_tables::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use ark_bn254::Fr;
    use common::consts::XLEN;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, OrTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, OrTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, OrTable<XLEN>>();
    }
}
