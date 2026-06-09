use std::fmt::Debug;

use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    lookup_tables::{
        prefixes::{PrefixEval, Prefixes},
        suffixes::{SuffixEval, Suffixes},
        JoltLookupTable, PrefixSuffixDecompositionTrait,
    },
};
use serde::{Deserialize, Serialize};

// const TABLE_SIZE: usize = 64;

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ClampTable<
    const X_LEN: usize,
    const LOWER_BOUND_LOG: usize,
    const UPPER_BOUND_LOG: usize,
>;

/// Structure that holds the prefixes required for a specific-bounds clamp table.
struct ClampPrefixes {
    /// Whether all bits x_i, where i >= UPPER_BOUND_LOG, are zero.
    zero_gt_upper_bound: Prefixes,
    /// Whether all bits x_i, where i >= LOWER_BOUND_LOG, are zero.
    zero_gt_lower_bound: Prefixes,
    /// The value of the word formed by bits x_i, where i < UPPER_BOUND_LOG.
    word_lt_upper_bound: Prefixes,
    /// The value of the word formed by bits x_i, where i < LOWER_BOUND_LOG.
    word_lt_lower_bound: Prefixes,
}

/// Structure that holds the suffixes required for a specific-bounds clamp table.
struct ClampSuffixes {
    /// Whether all bits x_i, where i >= UPPER_BOUND_LOG, are zero.
    zero_gt_upper_bound: Suffixes,
    /// Whether all bits x_i, where i >= LOWER_BOUND_LOG, are zero.
    zero_gt_lower_bound: Suffixes,
    /// The value of the word formed by bits x_i, where i < UPPER_BOUND_LOG.
    word_lt_upper_bound: Suffixes,
    /// The value of the word formed by bits x_i, where i < LOWER_BOUND_LOG.
    word_lt_lower_bound: Suffixes,
}

/// Holds the required bounds, prefixes and suffixes for a specific clamp table.
trait ClampTableTrait<const X_LEN: usize> {
    const LOWER_BOUND_LOG: usize;
    const UPPER_BOUND_LOG: usize;

    fn table_prefixes() -> ClampPrefixes;
    fn table_suffixes() -> ClampSuffixes;
}

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
        // We split `is_higher_zero_lbound` as
        // `is_higher_zero_hbound * something_else` where `something_else`
        // only uses variables in indices `(ubound_index, lbound_index]`.
        // This is valid because the two factors depend on disjoint variable sets.
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
        res += is_higher_zero_hbound * (hbound_word - F::from_i64(1 << U))
            + is_higher_zero_lbound * (F::from_i64(1 << L) - lbound_word);

        res
    }
}

impl<T, const X_LEN: usize> PrefixSuffixDecompositionTrait<X_LEN> for T
where
    T: ClampTableTrait<X_LEN> + JoltLookupTable + Default,
{
    fn prefixes(&self) -> Vec<Prefixes> {
        let pre = Self::table_prefixes();
        vec![
            pre.zero_gt_upper_bound,
            pre.zero_gt_lower_bound,
            pre.word_lt_upper_bound,
            pre.word_lt_lower_bound,
        ]
    }

    fn suffixes(&self) -> Vec<Suffixes> {
        let suf = Self::table_suffixes();
        vec![
            suf.zero_gt_upper_bound,
            suf.zero_gt_lower_bound,
            suf.word_lt_upper_bound,
            suf.word_lt_lower_bound,
        ]
    }

    #[cfg(test)]
    fn combine_test<F: JoltField>(
        &self,
        prefixes: &[PrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F {
        let [zero_gt_upper, zero_gt_lower, word_lt_upper, word_lt_lower] =
            suffixes.try_into().unwrap();
        let const_upper_bound_value = F::from_u64(1u64 << Self::UPPER_BOUND_LOG);
        let const_lower_bound_value = F::from_u64(1u64 << Self::LOWER_BOUND_LOG);

        let ps_zero_gtupper = prefixes[self.prefixes()[0]] * zero_gt_upper;
        let ps_zero_gtlower = prefixes[self.prefixes()[1]] * zero_gt_lower;
        let ps_word_ltupper = prefixes[self.prefixes()[2]] + word_lt_upper;
        let ps_word_ltlower = prefixes[self.prefixes()[3]] + word_lt_lower;

        const_upper_bound_value
            + ps_zero_gtupper * (ps_word_ltupper - const_upper_bound_value)
            + ps_zero_gtlower * (const_lower_bound_value - ps_word_ltlower)
    }

    fn combine<F: JoltField>(&self, _prefixes: &[PrefixEval<F>], _suffixes: &[SuffixEval<F>]) -> F {
        todo!();
        // let [s_zero_gt_upper, s_zero_gt_lower, s_word_lt_upper, s_word_lt_lower] =
        //     suffixes.try_into().unwrap();
        // let [p_zero_gt_upper, p_zero_gt_lower, p_word_lt_upper, p_word_lt_lower] =
        //     prefixes.try_into().unwrap();
        // let const_upper_bound_value = F::from_u64(1u64 << Self::UPPER_BOUND_LOG);
        // let const_lower_bound_value = F::from_u64(1u64 << Self::LOWER_BOUND_LOG);

        // let ps_zero_gtupper = p_zero_gt_upper * s_zero_gt_upper;
        // let ps_zero_gtlower = p_zero_gt_lower * s_zero_gt_lower;
        // let ps_word_ltupper = p_word_lt_upper + s_word_lt_upper;
        // let ps_word_ltlower = p_word_lt_lower + s_word_lt_lower;

        // const_upper_bound_value
        //     + ps_zero_gtupper * (ps_word_ltupper - const_upper_bound_value)
        //     + ps_zero_gtlower * (const_lower_bound_value - ps_word_ltlower)
    }
}

// Simulate a common upper bound and two different lower bounds for use in two different operators.
// Examples of needing operators: Tanh/Sigmoid/Erf, Saturating for Addition/Einsum ...
pub const CLAMP_OPS_UPPER: usize = 10;
pub const CLAMP_OP1_LOWER: usize = 5;
pub const CLAMP_OP2_LOWER: usize = 0;

pub type ClampTableOp1<const X_LEN: usize> = ClampTable<X_LEN, CLAMP_OP1_LOWER, CLAMP_OPS_UPPER>;
pub type ClampTableOp2<const X_LEN: usize> = ClampTable<X_LEN, CLAMP_OP2_LOWER, CLAMP_OPS_UPPER>;

// Implementation of the ClampTableTrait for Op1's table.
impl<const X_LEN: usize> ClampTableTrait<X_LEN> for ClampTableOp1<X_LEN> {
    const LOWER_BOUND_LOG: usize = CLAMP_OP1_LOWER;
    const UPPER_BOUND_LOG: usize = CLAMP_OPS_UPPER;

    fn table_prefixes() -> ClampPrefixes {
        ClampPrefixes {
            zero_gt_upper_bound: Prefixes::OpsZeroGtHigh,
            zero_gt_lower_bound: Prefixes::Op1ZeroGtLow,
            word_lt_upper_bound: Prefixes::OpsWordLtHigh,
            word_lt_lower_bound: Prefixes::Op1WordLtLow,
        }
    }

    fn table_suffixes() -> ClampSuffixes {
        ClampSuffixes {
            zero_gt_upper_bound: Suffixes::OpsZeroGtHigh,
            zero_gt_lower_bound: Suffixes::Op1ZeroGtLow,
            word_lt_upper_bound: Suffixes::OpsWordLtHigh,
            word_lt_lower_bound: Suffixes::Op1WordLtLow,
        }
    }
}

// Implementation of the ClampTableTrait for Op2's table.
impl<const X_LEN: usize> ClampTableTrait<X_LEN> for ClampTableOp2<X_LEN> {
    const LOWER_BOUND_LOG: usize = CLAMP_OP2_LOWER;
    const UPPER_BOUND_LOG: usize = CLAMP_OPS_UPPER;

    fn table_prefixes() -> ClampPrefixes {
        ClampPrefixes {
            zero_gt_upper_bound: Prefixes::OpsZeroGtHigh,
            zero_gt_lower_bound: Prefixes::Op2ZeroGtLow,
            word_lt_upper_bound: Prefixes::OpsWordLtHigh,
            word_lt_lower_bound: Prefixes::Op2WordLtLow,
        }
    }

    fn table_suffixes() -> ClampSuffixes {
        ClampSuffixes {
            zero_gt_upper_bound: Suffixes::OpsZeroGtHigh,
            zero_gt_lower_bound: Suffixes::Op2ZeroGtLow,
            word_lt_upper_bound: Suffixes::OpsWordLtHigh,
            word_lt_lower_bound: Suffixes::Op2WordLtLow,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::lookup_tables::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_linearity_test,
        lookup_table_mle_random_test, prefix_suffix_test,
    };
    use ark_bn254::Fr;
    use common::consts::XLEN;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, ClampTableOp1<XLEN>>();
        prefix_suffix_test::<XLEN, Fr, ClampTableOp2<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, ClampTableOp1<8>>();
        lookup_table_mle_full_hypercube_test::<Fr, ClampTableOp2<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, ClampTableOp1<XLEN>>();
        lookup_table_mle_random_test::<Fr, ClampTableOp2<XLEN>>();
    }

    #[test]
    fn mle_linearity() {
        lookup_table_mle_linearity_test::<XLEN, Fr, ClampTableOp1<XLEN>>();
        lookup_table_mle_linearity_test::<XLEN, Fr, ClampTableOp2<XLEN>>();
    }
}
