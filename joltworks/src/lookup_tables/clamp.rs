use std::fmt::Debug;

use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    lookup_tables::{
        prefixes::{
            word_lt_bound::OpsLowerWordPrefix, zero_gt_bound::OpsHigherIsZeroPrefix, PrefixEval,
            PrefixVariant, Prefixes,
        },
        suffixes::{
            zero_gt_bound::OpsHigherIsZeroSuffix,
            zero_gt_mul_word_lt_bound::OpsHZeroMulLWordSuffix, SuffixEval, SuffixVariant, Suffixes,
        },
        JoltLookupTable, PrefixSuffixDecompositionTrait,
    },
};
use serde::{Deserialize, Serialize};

/// Maps a concrete clamp table to the prefix/suffix types implementing its
/// "zero above bound" indicator and "word below bound" accumulator for the
/// upper bound channel.
///
/// The [`Prefixes`]/[`Suffixes`] enum variants are read directly off these
/// types via [`PrefixVariant`]/[`SuffixVariant`], so it's impossible for the
/// enum variant used here to drift out of sync with the type that actually
/// implements the corresponding MLE logic.
///
/// Implementing this trait is all that is needed to get a full
/// [`PrefixSuffixDecompositionTrait`] implementation via the blanket impl below.
trait ClampSpec {
    /// Prefix type: evaluates to 1 iff all bits with significance ≥ 2^BOUND_LOG are zero.
    type ZeroGtBound: PrefixVariant;
    /// Prefix type: accumulates the value of bits with significance < 2^BOUND_LOG.
    type WordLtBound: PrefixVariant;
    /// Suffix type for the zero-above-bound indicator.
    type SufZeroGtBound: SuffixVariant;
    /// Suffix type for the word-below-bound accumulator.
    type SufZeroMulWord: SuffixVariant;
    /// log₂ of the bound value: the clamp boundary is 2^BOUND_LOG.
    const BOUND_LOG: usize;
    // /// Spec for the lower bound channel.
    // /// Returns `None` for upper-only clamping (no lower correction term).
    // fn lower_spec() -> Option<BoundSpec>;
}

/// Blanket [`PrefixSuffixDecompositionTrait`] impl for every [`ClampSpec`] type.
///
/// Prefix/suffix order: `[zero_gt_upper, zero_gt_upper_mul_word_lt_upper]` for upper-only tables,
/// `[zero_gt_upper, word_lt_upper, zero_gt_lower, word_lt_lower]` for dual-bound.
impl<const XLEN: usize, T> PrefixSuffixDecompositionTrait<XLEN> for T
where
    T: ClampSpec + JoltLookupTable + Default,
{
    fn prefixes(&self) -> Vec<Prefixes> {
        vec![T::ZeroGtBound::VARIANT, T::WordLtBound::VARIANT]
    }

    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            T::SufZeroGtBound::VARIANT,
            T::SufZeroMulWord::VARIANT,
            Suffixes::One,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let const_upper = F::from_u64(1u64 << Self::BOUND_LOG);

        debug_assert!(prefixes.len() == 2);
        debug_assert!(suffixes.len() == 3);

        let [pre_zero_gtupper, pre_word_ltupper] = prefixes.try_into().unwrap();
        let [suf_zero_gtupper, suf_zero_word_upper, suf_one] = suffixes.try_into().unwrap();

        // Phase A baseline: table is constant, but we keep one suffix channel so
        // the prefix-suffix protocol can still carry weighted sums correctly.
        // let mut result = const_upper + ps_zero_gtupper * (ps_word_ltupper - const_upper);
        const_upper * suf_one
            + pre_zero_gtupper * (suf_zero_word_upper + pre_word_ltupper * suf_one)
            - pre_zero_gtupper * const_upper * suf_zero_gtupper
    }
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ClampTableSiBou<const XLEN: usize, const UPPER_BOUND_LOG: usize>;

impl<const XLEN: usize, const U: usize> JoltLookupTable for ClampTableSiBou<XLEN, U> {
    fn materialize_entry(&self, index: u64) -> u64 {
        if index >= 1 << U {
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
        clamp_evaluate_mle_inner::<XLEN, F, C>(r, U)
    }
}

/// Core MLE logic shared by all clamp variants.
fn clamp_evaluate_mle_inner<const XLEN: usize, F, C>(
    r: &[C],
    upper_log: usize,
    // lower_log: Option<usize>,
) -> F
where
    C: ChallengeFieldOps<F>,
    F: JoltField + FieldChallengeOps<C>,
{
    let indexed_r: Vec<_> = r.iter().enumerate().collect();
    let ubound_index = XLEN - upper_log - 1;

    // Indicator that all bits with significance >= 2^U are zero.
    let is_zero_gt_upper: F = indexed_r[..ubound_index + 1]
        .iter()
        .map(|(_, &r_i)| F::one() - r_i)
        .product();

    // Word value from bits with significance < 2^U.
    let word_lt_upper: F = indexed_r[ubound_index + 1..]
        .iter()
        .map(|(i, &r_i)| {
            let exponent = XLEN - i - 1;
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

    res
}

// Simulate a common upper bound and two different lower bounds for use in two different operators.
// Examples of needing operators: Tanh/Sigmoid/Erf, Saturating for Addition/Einsum ...
pub const CLAMP_OPS_UPPER: usize = 10;
pub const CLAMP_OP1_LOWER: usize = 5;
pub const CLAMP_OP2_LOWER: usize = 0;

pub type ClampTableOp3<const XLEN: usize> = ClampTableSiBou<XLEN, CLAMP_OPS_UPPER>;

// Implementation of ClampSpec for Op3's table (upper-only: no lower correction).
impl<const XLEN: usize> ClampSpec for ClampTableOp3<XLEN> {
    type ZeroGtBound = OpsHigherIsZeroPrefix<XLEN>;
    type WordLtBound = OpsLowerWordPrefix<XLEN>;
    type SufZeroGtBound = OpsHigherIsZeroSuffix<XLEN>;
    type SufZeroMulWord = OpsHZeroMulLWordSuffix<XLEN>;
    const BOUND_LOG: usize = CLAMP_OPS_UPPER;
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        lookup_tables::test::{
            lookup_table_mle_full_hypercube_test, lookup_table_mle_linearity_test,
            lookup_table_mle_random_test, prefix_suffix_test_unary,
        },
        subprotocols::ps_shout::unary::tests::test_read_raf_sumcheck,
    };
    use ark_bn254::Fr;
    use common::consts::XLEN;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test_unary::<XLEN, Fr, ClampTableOp3<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, ClampTableOp3<16>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, ClampTableOp3<64>>();
    }

    #[test]
    fn mle_linearity() {
        lookup_table_mle_linearity_test::<XLEN, Fr, ClampTableOp3<XLEN>>();
    }

    #[test]
    fn read_raf() {
        test_read_raf_sumcheck::<ClampTableOp3<XLEN>, XLEN>();
    }
}
