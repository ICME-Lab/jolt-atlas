use std::fmt::Debug;

use crate::onnx_proof::lookup_tables::{
    prefixes::{PrefixEval, Prefixes},
    suffixes::{SuffixEval, Suffixes},
    JoltLookupTable, PrefixSuffixDecompositionTrait,
};
use joltworks::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use serde::{Deserialize, Serialize};

// const TABLE_SIZE: usize = 64;

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ClampTable<
    const X_LEN: usize,
    const LOWER_BOUND_LOG: usize,
    const UPPER_BOUND_LOG: usize,
>;

impl<const X_LEN: usize, const L: usize, const U: usize> JoltLookupTable
    for ClampTable<X_LEN, L, U>
{
    fn materialize_entry(&self, index: u64) -> u64 {
        if index > 1 << U {
            1 << U
        } else if index > 1 << L {
            index
        } else {
            1 << L
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        let indexed_r: Vec<_> = r.iter().enumerate().collect();
        let ubound_index = 2 * X_LEN - U - 1; // index corresponding to the set bit in the upper bound
        let lbound_index = 2 * X_LEN - L - 1; // index corresponding to the set bit in the lower bound

        // this mle returns 1 if all bits with `significance >= upper bound` are zero.
        let is_higher_zero_hbound: F = indexed_r[..ubound_index + 1]
            .iter()
            .map(|(_, &r_i)| F::one() - r_i)
            .product();
        let is_higher_zero_lbound: F = is_higher_zero_hbound
            * indexed_r[ubound_index + 1..lbound_index + 1]
                .iter()
                .map(|(_, &r_i)| F::one() - r_i)
                .product::<F>();

        // this mle returns the value of the lower word, only including bits with `significance < lower bound`
        let lbound_word: F = indexed_r[lbound_index + 1..]
            .iter()
            .map(|(i, &r_i)| {
                let exponent = 2 * X_LEN - i - 1;
                r_i * F::from_u64(1 << exponent)
            })
            .sum();
        // this mle returns the value of the lower word, only including bits with `significance < upper bound`
        let hbound_word: F = lbound_word
            + indexed_r[ubound_index + 1..lbound_index + 1]
                .iter()
                .map(|(i, &r_i)| {
                    let exponent = 2 * X_LEN - i - 1;
                    r_i * F::from_u64(1 << exponent)
                })
                .sum::<F>();

        // Initialize the result with bounded value 2^U
        let mut res = F::from_i64(1 << U);
        // If the higher word is zero, means that x < 2^U.
        // Hence we return the lower word as the result, and compensate the initial value by subtracting 2^U.
        res += is_higher_zero_hbound
            * (hbound_word - F::from_i64(1 << U)
                + is_higher_zero_lbound * (F::from_i64(1 << L) - lbound_word));

        res
    }
}

impl<const X_LEN: usize, const L: usize, const U: usize> PrefixSuffixDecompositionTrait<X_LEN>
    for ClampTable<X_LEN, L, U>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Relu]
    }

    fn prefixes(&self) -> Vec<Prefixes> {
        vec![Prefixes::NotMsb, Prefixes::LowerWordNoMsb]
    }

    #[cfg(test)]
    fn combine_test<F: JoltField>(
        &self,
        prefixes: &[PrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F {
        let [one, relu] = suffixes.try_into().unwrap();
        prefixes[Prefixes::NotMsb] * prefixes[Prefixes::LowerWordNoMsb] * one
            + relu * prefixes[Prefixes::NotMsb]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [suffix_one, suffix_relu] = suffixes.try_into().unwrap();
        let [prefix_not_msb, prefix_lower_word_no_msb] = prefixes.try_into().unwrap();
        prefix_not_msb * prefix_lower_word_no_msb * suffix_one + prefix_not_msb * suffix_relu
    }
}

#[cfg(test)]
mod test {
    use crate::onnx_proof::lookup_tables::{
        clamp::ClampTable,
        test::{
            lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
        },
        JoltLookupTable,
    };
    use ark_bn254::Fr;
    use common::consts::XLEN;
    use joltworks::utils::index_to_field_bitvector;

    #[test]
    #[ignore] // Unimplemented yet
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, ClampTable<XLEN, 0, 10>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, ClampTable<8, 5, 10>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, ClampTable<XLEN, 5, 10>>();
    }
}
