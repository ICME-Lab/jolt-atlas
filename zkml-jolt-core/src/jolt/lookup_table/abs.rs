use jolt_core::field::JoltField;
use serde::{Deserialize, Serialize};

use super::AtlasLookupTable;
use super::PrefixSuffixDecomposition;
use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct AbsTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> AtlasLookupTable for AbsTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let sign_bit = 1 << (WORD_SIZE - 1);
        if sign_bit & index == 0 {
            index % (1 << WORD_SIZE)
        } else {
            // In two's complement, -x = (!x) + 1
            // Where ! is the bitwise NOT operator
            ((!index).wrapping_add(1)) % (1 << WORD_SIZE)
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);

        let mut positive_case = F::zero();
        // if x < 0, abs(x) = -x = (!x) + 1
        let mut negative_case = F::one();
        for i in 0..WORD_SIZE - 1 {
            positive_case += F::from_u64(1 << i) * r[r.len() - 1 - i];
            negative_case += F::from_u64(1 << i) * (F::one() - r[r.len() - 1 - i]);
        }

        // r[WORD_SIZE] is the sign bit, we select which of positive or negative case we keep according to it.
        positive_case * (F::one() - r[WORD_SIZE]) + negative_case * r[WORD_SIZE]
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for AbsTable<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Relu, Suffixes::AbsNegativeCase]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, relu, abs_negative_case] = suffixes.try_into().unwrap();

        prefixes[Prefixes::Abs] * one
            + prefixes[Prefixes::NotUnaryMsb] * relu
            + prefixes[Prefixes::UnaryMsb] * abs_negative_case
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::AbsTable;
    use crate::jolt::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, AbsTable<32>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, AbsTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, AbsTable<32>>();
    }
}
