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

/// Configures one bound channel (upper or lower) for a clamp table.
/// Bundles the bound exponent with the specific prefix/suffix enum variants that
/// implement the "zero above bound" indicator and "word below bound" accumulator
/// for that channel.
pub struct BoundSpec {
    /// log₂ of the bound value: the clamp boundary is 2^bound_log.
    pub bound_log: usize,
    /// Prefix variant: evaluates to 1 iff all bits with significance ≥ 2^bound_log are zero.
    pub pre_zero_gtbound: Prefixes,
    /// Prefix variant: accumulates the value of bits with significance < 2^bound_log.
    pub pre_word_ltbound: Prefixes,
    /// Suffix variant for the zero-above-bound indicator.
    pub suf_zero_gtbound: Suffixes,
    /// Suffix variant for the word-below-bound accumulator.
    pub suf_word_ltbound: Suffixes,
}

/// Maps a concrete clamp table to its prefix/suffix variant channels.
///
/// Implementing this trait is all that is needed to get a full
/// [`PrefixSuffixDecompositionTrait`] implementation via the blanket impl below.
trait ClampSpec {
    /// Spec for the upper bound channel (always present).
    fn upper_spec() -> BoundSpec;
    /// Spec for the lower bound channel.
    /// Returns `None` for upper-only clamping (no lower correction term).
    fn lower_spec() -> Option<BoundSpec>;
}

/// Core combine logic shared by both `combine` and `combine_test`.
fn clamp_combine_inner<F: JoltField>(
    upper: &BoundSpec,
    lower: Option<&BoundSpec>,
    prefixes: &[PrefixEval<F>],
    suffixes: &[SuffixEval<F>],
) -> F {
    let const_upper = F::from_u64(1u64 << upper.bound_log);
    let ps_zero_gtupper = prefixes[upper.pre_zero_gtbound] * suffixes[0];
    let ps_word_ltupper = prefixes[upper.pre_word_ltbound] + suffixes[1];

    let mut result = const_upper + ps_zero_gtupper * (ps_word_ltupper - const_upper);

    if let Some(lower) = lower {
        let const_lower = F::from_u64(1u64 << lower.bound_log);
        let ps_zero_gtlower = prefixes[lower.pre_zero_gtbound] * suffixes[2];
        let ps_word_ltlower = prefixes[lower.pre_word_ltbound] + suffixes[3];
        result += ps_zero_gtlower * (const_lower - ps_word_ltlower);
    }

    result
}

/// Blanket [`PrefixSuffixDecompositionTrait`] impl for every [`ClampSpec`] type.
///
/// Prefix/suffix order: `[zero_gt_upper, word_lt_upper]` for upper-only tables,
/// `[zero_gt_upper, word_lt_upper, zero_gt_lower, word_lt_lower]` for dual-bound.
impl<const X_LEN: usize, T> PrefixSuffixDecompositionTrait<X_LEN> for T
where
    T: ClampSpec + JoltLookupTable + Default,
{
    fn prefixes(&self) -> Vec<Prefixes> {
        let upper = Self::upper_spec();
        let mut result = vec![upper.pre_zero_gtbound, upper.pre_word_ltbound];
        if let Some(lower) = Self::lower_spec() {
            result.push(lower.pre_zero_gtbound);
            result.push(lower.pre_word_ltbound);
        }
        result
    }

    fn suffixes(&self) -> Vec<Suffixes> {
        let upper = Self::upper_spec();
        let mut result = vec![upper.suf_zero_gtbound, upper.suf_word_ltbound];
        if let Some(lower) = Self::lower_spec() {
            result.push(lower.suf_zero_gtbound);
            result.push(lower.suf_word_ltbound);
        }
        result
    }

    fn combine<F: JoltField>(&self, _prefixes: &[PrefixEval<F>], _suffixes: &[SuffixEval<F>]) -> F {
        todo!()
    }

    #[cfg(test)]
    fn combine_test<F: JoltField>(
        &self,
        prefixes: &[PrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F {
        clamp_combine_inner(
            &Self::upper_spec(),
            Self::lower_spec().as_ref(),
            prefixes,
            suffixes,
        )
    }
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ClampTableDeBou<
    const X_LEN: usize,
    const LOWER_BOUND_LOG: usize,
    const UPPER_BOUND_LOG: usize,
>;

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ClampTableSiBou<const X_LEN: usize, const UPPER_BOUND_LOG: usize>;

impl<const X_LEN: usize, const L: usize, const U: usize> JoltLookupTable
    for ClampTableDeBou<X_LEN, L, U>
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
        clamp_evaluate_mle_inner::<X_LEN, F, C>(r, U, Some(L))
    }
}

impl<const X_LEN: usize, const U: usize> JoltLookupTable for ClampTableSiBou<X_LEN, U> {
    fn materialize_entry(&self, index: u64) -> u64 {
        if index > 1 << U {
            1 << U
        } else {
            index
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        clamp_evaluate_mle_inner::<X_LEN, F, C>(r, U, None)
    }
}

/// Core MLE logic shared by all clamp variants.
fn clamp_evaluate_mle_inner<const X_LEN: usize, F, C>(
    r: &[C],
    upper_log: usize,
    lower_log: Option<usize>,
) -> F
where
    C: ChallengeFieldOps<F>,
    F: JoltField + FieldChallengeOps<C>,
{
    let indexed_r: Vec<_> = r.iter().enumerate().collect();
    let ubound_index = 2 * X_LEN - upper_log - 1;

    // Indicator that all bits with significance >= 2^U are zero.
    let is_zero_gt_upper: F = indexed_r[..ubound_index + 1]
        .iter()
        .map(|(_, &r_i)| F::one() - r_i)
        .product();

    // Word value from bits with significance < 2^U.
    let word_lt_upper: F = indexed_r[ubound_index + 1..]
        .iter()
        .map(|(i, &r_i)| {
            let exponent = 2 * X_LEN - i - 1;
            r_i * F::from_u64(1 << exponent)
        })
        .sum();

    let const_upper = F::from_i64(1 << upper_log);

    // Defaults to the upper bound
    let mut res = const_upper;
    // If input < 2^U (all high-significance bits = 0),
    // then we cancel the upper bound contribution,
    // and add the input value (which is fully represented by the lower bits).
    res += is_zero_gt_upper * (word_lt_upper - const_upper);

    if let Some(lower_log) = lower_log {
        debug_assert!(lower_log <= upper_log);
        let lbound_index = 2 * X_LEN - lower_log - 1;

        // Split lower indicator into upper indicator and the disjoint middle range.
        let is_zero_gt_lower: F = is_zero_gt_upper
            * indexed_r[ubound_index + 1..lbound_index + 1]
                .iter()
                .map(|(_, &r_i)| F::one() - r_i)
                .product::<F>();

        // Word value from bits with significance < 2^L.
        let word_lt_lower: F = indexed_r[lbound_index + 1..]
            .iter()
            .map(|(i, &r_i)| {
                let exponent = 2 * X_LEN - i - 1;
                r_i * F::from_u64(1 << exponent)
            })
            .sum();

        let const_lower = F::from_i64(1 << lower_log);

        // if input < 2^L (all bits with significance >= 2^L are zero),
        // then we cancel the input value (fully represented by the lower bits),
        // and add the lower bound.
        res += is_zero_gt_lower * (const_lower - word_lt_lower);
    }

    res
}

// Simulate a common upper bound and two different lower bounds for use in two different operators.
// Examples of needing operators: Tanh/Sigmoid/Erf, Saturating for Addition/Einsum ...
pub const CLAMP_OPS_UPPER: usize = 10;
pub const CLAMP_OP1_LOWER: usize = 5;
pub const CLAMP_OP2_LOWER: usize = 0;

pub type ClampTableOp1<const X_LEN: usize> =
    ClampTableDeBou<X_LEN, CLAMP_OP1_LOWER, CLAMP_OPS_UPPER>;
pub type ClampTableOp2<const X_LEN: usize> =
    ClampTableDeBou<X_LEN, CLAMP_OP2_LOWER, CLAMP_OPS_UPPER>;
pub type ClampTableOp3<const X_LEN: usize> = ClampTableSiBou<X_LEN, CLAMP_OPS_UPPER>;

// Implementation of ClampSpec for Op1's table (dual-bound: upper=10, lower=5).
impl<const X_LEN: usize> ClampSpec for ClampTableOp1<X_LEN> {
    fn upper_spec() -> BoundSpec {
        BoundSpec {
            bound_log: CLAMP_OPS_UPPER,
            pre_zero_gtbound: Prefixes::OpsZeroGtHigh,
            pre_word_ltbound: Prefixes::OpsWordLtHigh,
            suf_zero_gtbound: Suffixes::OpsZeroGtHigh,
            suf_word_ltbound: Suffixes::OpsWordLtHigh,
        }
    }

    fn lower_spec() -> Option<BoundSpec> {
        Some(BoundSpec {
            bound_log: CLAMP_OP1_LOWER,
            pre_zero_gtbound: Prefixes::Op1ZeroGtLow,
            pre_word_ltbound: Prefixes::Op1WordLtLow,
            suf_zero_gtbound: Suffixes::Op1ZeroGtLow,
            suf_word_ltbound: Suffixes::Op1WordLtLow,
        })
    }
}

// Implementation of ClampSpec for Op2's table (dual-bound: upper=10, lower=0).
impl<const X_LEN: usize> ClampSpec for ClampTableOp2<X_LEN> {
    fn upper_spec() -> BoundSpec {
        BoundSpec {
            bound_log: CLAMP_OPS_UPPER,
            pre_zero_gtbound: Prefixes::OpsZeroGtHigh,
            pre_word_ltbound: Prefixes::OpsWordLtHigh,
            suf_zero_gtbound: Suffixes::OpsZeroGtHigh,
            suf_word_ltbound: Suffixes::OpsWordLtHigh,
        }
    }

    fn lower_spec() -> Option<BoundSpec> {
        Some(BoundSpec {
            bound_log: CLAMP_OP2_LOWER,
            pre_zero_gtbound: Prefixes::Op2ZeroGtLow,
            pre_word_ltbound: Prefixes::Op2WordLtLow,
            suf_zero_gtbound: Suffixes::Op2ZeroGtLow,
            suf_word_ltbound: Suffixes::Op2WordLtLow,
        })
    }
}

// Implementation of ClampSpec for Op3's table (upper-only: no lower correction).
impl<const X_LEN: usize> ClampSpec for ClampTableOp3<X_LEN> {
    fn upper_spec() -> BoundSpec {
        BoundSpec {
            bound_log: CLAMP_OPS_UPPER,
            pre_zero_gtbound: Prefixes::OpsZeroGtHigh,
            pre_word_ltbound: Prefixes::OpsWordLtHigh,
            suf_zero_gtbound: Suffixes::OpsZeroGtHigh,
            suf_word_ltbound: Suffixes::OpsWordLtHigh,
        }
    }

    fn lower_spec() -> Option<BoundSpec> {
        None
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
        prefix_suffix_test::<XLEN, Fr, ClampTableOp3<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, ClampTableOp1<8>>();
        lookup_table_mle_full_hypercube_test::<Fr, ClampTableOp2<8>>();
        lookup_table_mle_full_hypercube_test::<Fr, ClampTableOp3<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, ClampTableOp1<XLEN>>();
        lookup_table_mle_random_test::<Fr, ClampTableOp2<XLEN>>();
        lookup_table_mle_random_test::<Fr, ClampTableOp3<XLEN>>();
    }

    #[test]
    fn mle_linearity() {
        lookup_table_mle_linearity_test::<XLEN, Fr, ClampTableOp1<XLEN>>();
        lookup_table_mle_linearity_test::<XLEN, Fr, ClampTableOp2<XLEN>>();
        lookup_table_mle_linearity_test::<XLEN, Fr, ClampTableOp3<XLEN>>();
    }
}
