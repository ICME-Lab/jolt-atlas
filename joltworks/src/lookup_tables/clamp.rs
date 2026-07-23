use std::fmt::Debug;

use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    lookup_tables::{
        prefixes::{
            higher_is_zero::ClampHigherIsZeroPrefix, lower_word::ClampLowerWordPrefix, PrefixEval,
            PrefixVariant, Prefixes,
        },
        suffixes::{
            higher_is_zero::ClampHigherIsZeroSuffix, hzero_mul_lword::ClampHZeroMulLWordSuffix,
            SuffixEval, SuffixVariant, Suffixes,
        },
        JoltLookupTable, PrefixSuffixDecompositionTrait,
    },
};
use serde::{Deserialize, Serialize};

/// Maps a concrete clamp table to the prefix/suffix types implementing its
/// "higher is zero" indicator and "lower word" accumulator for the
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
    /// Prefix type: evaluates to 1 iff all bits with significance ≥ 2^BOUND are zero.
    type HigherIsZero: PrefixVariant;
    /// Prefix type: accumulates the value of bits with significance < 2^BOUND.
    type LowerWord: PrefixVariant;
    /// Suffix type for the higher-is-zero indicator.
    type SufHigherIsZero: SuffixVariant;
    /// Suffix type for the higher-is-zero-times-lower-word accumulator.
    type SufHZeroMulLWord: SuffixVariant;
    /// log₂ of the bound value: the clamp boundary is 2^BOUND.
    const BOUND: usize;
    // /// Spec for the lower bound channel.
    // /// Returns `None` for upper-only clamping (no lower correction term).
    // fn lower_spec() -> Option<BoundSpec>;
}

/// Blanket [`PrefixSuffixDecompositionTrait`] impl for every [`ClampSpec`] type.
///
/// Prefix/suffix order: `[higher_is_zero, higher_is_zero_mul_lower_word]` for upper-only tables,
/// `[higher_is_zero, lower_word, higher_is_zero_lower, lower_word_lower]` for dual-bound.
impl<const XLEN: usize, T> PrefixSuffixDecompositionTrait<XLEN> for T
where
    T: ClampSpec + JoltLookupTable + Default,
{
    fn prefixes(&self) -> Vec<Prefixes> {
        vec![T::HigherIsZero::VARIANT, T::LowerWord::VARIANT]
    }

    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            T::SufHigherIsZero::VARIANT,
            T::SufHZeroMulLWord::VARIANT,
            Suffixes::One,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let const_upper = F::from_u64(1u64 << Self::BOUND);

        let [pre_higher_is_zero, pre_lower_word] = prefixes.try_into().unwrap();
        let [suf_higher_is_zero, suf_hzero_mul_lword, suf_one] = suffixes.try_into().unwrap();

        const_upper * suf_one
            + pre_higher_is_zero * (suf_hzero_mul_lword + pre_lower_word * suf_one)
            - pre_higher_is_zero * const_upper * suf_higher_is_zero
    }
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ClampBoundedTable<const XLEN: usize, const BOUND: usize>;

impl<const XLEN: usize, const BOUND: usize> JoltLookupTable for ClampBoundedTable<XLEN, BOUND> {
    fn materialize_entry(&self, index: u64) -> u64 {
        if index >= 1 << BOUND {
            1 << BOUND
        } else {
            index
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        let indexed_r: Vec<_> = r.iter().enumerate().collect();
        let ubound_index = XLEN - BOUND - 1;

        // Indicator that all bits with significance >= 2^U are zero.
        let is_higher_is_zero: F = indexed_r[..ubound_index + 1]
            .iter()
            .map(|(_, &r_i)| F::one() - r_i)
            .product();

        // Word value from bits with significance < 2^U.
        let lower_word: F = indexed_r[ubound_index + 1..]
            .iter()
            .map(|(i, &r_i)| {
                let exponent = XLEN - i - 1;
                r_i * F::from_u64(1 << exponent)
            })
            .sum();

        let const_upper = F::from_i64(1 << BOUND);

        // Defaults to the upper bound
        let mut res = const_upper;
        // If input < 2^BOUND (all high-significance bits = 0),
        // then we cancel the upper bound contribution,
        // and add the input value (which is fully represented by the lower bits).
        res += is_higher_is_zero * (lower_word - const_upper);

        res
    }
}

// This bound is also intended for future reuse by other saturating operators
// (e.g. Tanh/Sigmoid/Erf, saturating Addition/Einsum) sharing the same table shape.
pub const CLAMP_BOUND: usize = 10;

pub type ClampTable<const XLEN: usize> = ClampBoundedTable<XLEN, CLAMP_BOUND>;

// Implementation of ClampSpec for the clamp table (upper-only: no lower correction).
impl<const XLEN: usize> ClampSpec for ClampTable<XLEN> {
    type HigherIsZero = ClampHigherIsZeroPrefix<XLEN>;
    type LowerWord = ClampLowerWordPrefix<XLEN>;
    type SufHigherIsZero = ClampHigherIsZeroSuffix<XLEN>;
    type SufHZeroMulLWord = ClampHZeroMulLWordSuffix<XLEN>;
    const BOUND: usize = CLAMP_BOUND;
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
        prefix_suffix_test_unary::<XLEN, Fr, ClampTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, ClampTable<16>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, ClampTable<64>>();
    }

    #[test]
    fn mle_linearity() {
        lookup_table_mle_linearity_test::<XLEN, Fr, ClampTable<XLEN>>();
    }

    #[test]
    fn read_raf() {
        test_read_raf_sumcheck::<ClampTable<XLEN>, XLEN>();
    }
}
