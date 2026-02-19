use joltworks::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::uninterleave_bits,
};
use serde::{Deserialize, Serialize};

use crate::onnx_proof::lookup_tables::PrefixSuffixDecompositionTrait;

use super::{
    prefixes::{PrefixEval, Prefixes},
    suffixes::{SuffixEval, Suffixes},
    JoltLookupTable,
};

/// Lookup table for unsigned less-than comparison.
///
/// Implements LT(x, y) = 1 if x < y (unsigned), 0 otherwise.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct UnsignedLessThanTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for UnsignedLessThanTable<XLEN> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        (x < y).into()
    }
    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);

        // \sum_i (1 - x_i) * y_i * \prod_{j < i} ((1 - x_j) * (1 - y_j) + x_j * y_j)
        let mut result = F::zero();
        let mut eq_term = F::one();
        for i in 0..XLEN {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            result += (F::one() - x_i) * y_i * eq_term;
            eq_term *= x_i * y_i + (F::one() - x_i) * (F::one() - y_i);
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecompositionTrait<XLEN> for UnsignedLessThanTable<XLEN> {
    fn prefixes(&self) -> Vec<Prefixes> {
        vec![Prefixes::Eq, Prefixes::LessThan]
    }

    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LessThan]
    }

    #[cfg(test)]
    fn combine_test<F: JoltField>(
        &self,
        prefixes: &[PrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F {
        let [one, less_than] = suffixes.try_into().unwrap();
        prefixes[Prefixes::LessThan] * one + prefixes[Prefixes::Eq] * less_than
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [one, less_than_suffix] = suffixes.try_into().unwrap();
        let [eq, less_than_prefix] = prefixes.try_into().unwrap();
        less_than_prefix * one + eq * less_than_suffix
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use common::consts::XLEN;

    use crate::onnx_proof::lookup_tables::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };

    use super::UnsignedLessThanTable;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, UnsignedLessThanTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, UnsignedLessThanTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, UnsignedLessThanTable<XLEN>>();
    }
}
