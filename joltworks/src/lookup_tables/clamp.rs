use super::{
    prefixes::{
        higher_all_one::{
            ActivationHigherAllOnePrefix, ClampHigherAllOnePrefix, SatClampHigherAllOnePrefix,
        },
        higher_all_zero::{
            ActivationHigherAllZeroPrefix, ClampHigherAllZeroPrefix, SatClampHigherAllZeroPrefix,
            SoftmaxClampHigherAllZeroPrefix,
        },
        lower_word::{
            ActivationLowerWordPrefix, ClampLowerWordPrefix, SatClampLowerWordPrefix,
            SoftmaxClampLowerWordPrefix,
        },
        zero::ZeroPrefix,
        PrefixEval, PrefixVariant, Prefixes,
    },
    suffixes::{
        higher_all_zero::{
            ActivationHigherAllZeroSuffix, ClampHigherAllZeroSuffix, SatClampHigherAllZeroSuffix,
            SoftmaxClampHigherAllZeroSuffix,
        },
        hone_mul_lword::{
            ActivationHOneMulLWordSuffix, ClampHOneMulLWordSuffix, SatClampHOneMulLWordSuffix,
        },
        hzero_mul_lword::{
            ActivationHZeroMulLWordSuffix, ClampHZeroMulLWordSuffix, SatClampHZeroMulLWordSuffix,
            SoftmaxClampHZeroMulLWordSuffix,
        },
        one::OneSuffix,
        SuffixEval, SuffixVariant, Suffixes,
    },
    JoltLookupTable, PrefixSuffixDecompositionTrait,
};
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use common::consts::{ACTIVATION_BOUND, SOFTMAX_CLAMP_BOUND};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Maps a concrete clamp table to its prefix/suffix decomposition. Covers both floor-at-0
/// (`SYMMETRIC = false`, e.g. [`SoftmaxClampTable`]) and natively symmetric (`SYMMETRIC = true`)
/// tables with one trait — see the `HigherAllOne`/`SufHOneMulLWord` docs below for the
/// floor-at-0 case.
pub(crate) trait ClampSpec {
    /// Prefix type: evaluates to 1 iff all bits with significance >= 2^BOUND, sign included,
    /// are zero (the positive/in-range branch).
    type HigherAllZero: PrefixVariant;
    /// Prefix type: evaluates to 1 iff all bits with significance >= 2^BOUND, sign included,
    /// are one (the negative/in-range branch). [`ZeroPrefix`] for `SYMMETRIC = false`.
    type HigherAllOne: PrefixVariant;
    /// Prefix type: accumulates the value of bits with significance < 2^BOUND.
    type LowerWord: PrefixVariant;
    /// Suffix type for the higher-is-zero indicator.
    type SufHigherAllZero: SuffixVariant;
    /// Suffix type for the higher-is-zero-times-lower-word accumulator.
    type SufHZeroMulLWord: SuffixVariant;
    /// Suffix type for the higher-is-one-times-lower-word accumulator. Value is irrelevant for
    /// `SYMMETRIC = false` (its term is annihilated by `HigherAllOne`), so [`OneSuffix`] is fine.
    type SufHOneMulLWord: SuffixVariant;
    /// log₂ of the bound value: the clamp's upper bound is `2^BOUND - 1`.
    const BOUND: usize;
    /// `false` clamps to `[0, 2^BOUND - 1]` (floor-at-0); `true` clamps to
    /// `[-2^BOUND, 2^BOUND - 1]`.
    const SYMMETRIC: bool;
}

/// Blanket [`PrefixSuffixDecompositionTrait`] impl for every [`ClampSpec`] type.
///
/// The `combine` formula was derived from, and numerically verified against,
/// [`ClampBoundedTable::materialize_entry`] over all boolean corners at every prefix/suffix
/// split point, for both `SYMMETRIC = false` and `SYMMETRIC = true` (see the property tests
/// below, which exercise the real field-valued streaming-checkpoint path).
impl<const XLEN: usize, T> PrefixSuffixDecompositionTrait<XLEN> for T
where
    T: ClampSpec + JoltLookupTable + Default,
{
    fn prefixes(&self) -> Vec<Prefixes> {
        vec![
            T::HigherAllZero::VARIANT,
            T::HigherAllOne::VARIANT,
            T::LowerWord::VARIANT,
            Prefixes::Msb,
        ]
    }

    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            T::SufHigherAllZero::VARIANT,
            T::SufHZeroMulLWord::VARIANT,
            T::SufHOneMulLWord::VARIANT,
            Suffixes::One,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let const_upper = F::from_u64((1u64 << Self::BOUND) - 1);
        let lower_coeff = if Self::SYMMETRIC {
            const_upper + const_upper + F::one()
        } else {
            const_upper
        };

        let [pre_higher_all_zero, pre_higher_all_one, pre_lower_word, pre_msb] =
            prefixes.try_into().unwrap();
        let [suf_higher_all_zero, suf_hzero_mul_lword, suf_hone_mul_lword, suf_one] =
            suffixes.try_into().unwrap();

        // Default to the upper bound, or the lower bound if the sign bit is set.
        suf_one * const_upper - pre_msb * suf_one * lower_coeff
        // If the input is in [0, 2^BOUND), add lower word and cancel the upper-bound default.
            + pre_higher_all_zero
                * (suf_hzero_mul_lword + pre_lower_word * suf_one
                    - suf_higher_all_zero * const_upper)
        // If the input is in [-2^BOUND, -1], add lower word (the lower-bound default already
        // accounts for the `-2^BOUND` floor, so no further correction term is needed here).
            + pre_higher_all_one * (suf_hone_mul_lword + pre_lower_word * suf_one)
    }
}

/// A signed `XLEN`-bit input, clamped to `[0, 2^BOUND - 1]` (`SYMMETRIC = false`) or
/// `[-2^BOUND, 2^BOUND - 1]` (`SYMMETRIC = true`).
#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ClampBoundedTable<const XLEN: usize, const BOUND: usize, const SYMMETRIC: bool>;

impl<const XLEN: usize, const BOUND: usize, const SYMMETRIC: bool> JoltLookupTable
    for ClampBoundedTable<XLEN, BOUND, SYMMETRIC>
{
    fn materialize_entry(&self, index: u64) -> u64 {
        let val: i64 = match XLEN {
            8 => index as u8 as i8 as i64,
            16 => index as u16 as i16 as i64,
            32 => index as u32 as i32 as i64,
            64 => index as i64,
            _ => unimplemented!(),
        };
        let lower = if SYMMETRIC { -(1i64 << BOUND) } else { 0 };
        let upper = (1i64 << BOUND) - 1;
        val.clamp(lower, upper) as u64
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        // Only the last XLEN bits hold the input value.
        let offset = r.len() - XLEN;
        let indexed_r: Vec<_> = r[offset..].iter().enumerate().collect();
        let ubound_index = XLEN - BOUND - 1;

        // sign bit
        let msb = *indexed_r[0].1;

        // Indicator that all bits with significance >= 2^BOUND, sign included, are zero.
        let is_higher_all_zero: F = indexed_r[..ubound_index + 1]
            .iter()
            .map(|(_, &r_i)| F::one() - r_i)
            .product();

        // Indicator that all bits with significance >= 2^BOUND, sign included, are one.
        // Structurally zero for SYMMETRIC=false: see the doc comment on `ClampSpec`.
        let is_higher_all_one: F = if SYMMETRIC {
            indexed_r[..ubound_index + 1]
                .iter()
                .map(|(_, &r_i)| r_i.into())
                .product()
        } else {
            F::zero()
        };

        // Word value from bits with significance < 2^BOUND.
        let lower_word: F = indexed_r[ubound_index + 1..]
            .iter()
            .map(|(i, &r_i)| {
                let exponent = XLEN - i - 1;
                r_i * F::from_u64(1 << exponent)
            })
            .sum();

        let const_upper = F::from_i64((1 << BOUND) - 1);
        let lower_coeff = if SYMMETRIC {
            const_upper + const_upper + F::one()
        } else {
            const_upper
        };

        // Defaults to the upper bound, or the lower bound if the sign bit is set.
        const_upper - msb * lower_coeff
        // If input in [0, 2^BOUND) (all high-significance bits, sign included, are 0),
        // add lower word and cancel the upper-bound default.
            + is_higher_all_zero * (lower_word - const_upper)
            // If input in [-2^BOUND, -1] (all high-significance bits, sign included, are 1),
            // add lower word (the lower-bound default already covers the `-2^BOUND` floor).
            + is_higher_all_one * lower_word
    }
}

/// The effective bound of the ONNX `Clamp` op: it clamps into `[-2^CLAMP_BOUND, 2^CLAMP_BOUND - 1]`
/// (see `jolt_atlas_core::onnx_proof::ops::clamp`).
pub const CLAMP_BOUND: usize = 9;

/// The bound index (log2 of `i32::MAX + 1`) used to saturate an accumulation to i32's exact
/// signed range `[-2^SIGN_BIT_I32, 2^SIGN_BIT_I32 - 1]` = `[i32::MIN, i32::MAX]`.
pub const SIGN_BIT_I32: usize = 31;

/// The address width (in bits) of the saturating-accumulator clamp table: wide enough to hold
/// an i64 accumulation (e.g. the sum of two i32 operands) before it's saturated to i32's range.
pub const LARGE_XLEN: usize = 64;

/// Clamps the ONNX `Clamp` op's input natively to `[-2^CLAMP_BOUND, 2^CLAMP_BOUND - 1]`
/// (see `jolt_atlas_core::onnx_proof::ops::clamp`).
pub type ClampTable<const XLEN: usize> = ClampBoundedTable<XLEN, CLAMP_BOUND, true>;

impl<const XLEN: usize> ClampSpec for ClampTable<XLEN> {
    type HigherAllZero = ClampHigherAllZeroPrefix<XLEN>;
    type HigherAllOne = ClampHigherAllOnePrefix<XLEN>;
    type LowerWord = ClampLowerWordPrefix<XLEN>;
    type SufHigherAllZero = ClampHigherAllZeroSuffix<XLEN>;
    type SufHZeroMulLWord = ClampHZeroMulLWordSuffix<XLEN>;
    type SufHOneMulLWord = ClampHOneMulLWordSuffix<XLEN>;
    const BOUND: usize = CLAMP_BOUND;
    const SYMMETRIC: bool = true;
}

/// Saturates an i64 accumulation to i32's exact signed range `[-2^31, 2^31 - 1]` (i.e.
/// `[i32::MIN, i32::MAX]`), used by the shared saturating-arithmetic clamp
/// (`jolt_atlas_core::onnx_proof::clamp_lookups`, backing `Add`/`Sub`/`Sum`/`Einsum`/`Mul`/
/// `Square`/`Cube`/`MeanOfSquares`).
pub type SaturationTable = ClampBoundedTable<LARGE_XLEN, SIGN_BIT_I32, true>;

impl ClampSpec for SaturationTable {
    type HigherAllZero = SatClampHigherAllZeroPrefix<LARGE_XLEN>;
    type HigherAllOne = SatClampHigherAllOnePrefix<LARGE_XLEN>;
    type LowerWord = SatClampLowerWordPrefix<LARGE_XLEN>;
    type SufHigherAllZero = SatClampHigherAllZeroSuffix<LARGE_XLEN>;
    type SufHZeroMulLWord = SatClampHZeroMulLWordSuffix<LARGE_XLEN>;
    type SufHOneMulLWord = SatClampHOneMulLWordSuffix<LARGE_XLEN>;
    const BOUND: usize = SIGN_BIT_I32;
    const SYMMETRIC: bool = true;
}

/// Clamps Erf/Sigmoid/Tanh input to `[-2^ACTIVATION_BOUND, 2^ACTIVATION_BOUND - 1]` at model
/// scale [`common::consts::MODEL_SCALE`], before the small activation-table lookup.
pub type ActivationClampTable<const XLEN: usize> = ClampBoundedTable<XLEN, ACTIVATION_BOUND, true>;

impl<const XLEN: usize> ClampSpec for ActivationClampTable<XLEN> {
    type HigherAllZero = ActivationHigherAllZeroPrefix<XLEN>;
    type HigherAllOne = ActivationHigherAllOnePrefix<XLEN>;
    type LowerWord = ActivationLowerWordPrefix<XLEN>;
    type SufHigherAllZero = ActivationHigherAllZeroSuffix<XLEN>;
    type SufHZeroMulLWord = ActivationHZeroMulLWordSuffix<XLEN>;
    type SufHOneMulLWord = ActivationHOneMulLWordSuffix<XLEN>;
    const BOUND: usize = ACTIVATION_BOUND;
    const SYMMETRIC: bool = true;
}

/// Clamps softmax's `z = max_k - x` (always non-negative by construction) to
/// `[0, 2^SOFTMAX_CLAMP_BOUND - 1]` at model scale [`common::consts::MODEL_SCALE`]
/// (`jolt_atlas_core::onnx_proof::ops::softmax_last_axis::significance_clamp`).
pub type SoftmaxClampTable<const XLEN: usize> = ClampBoundedTable<XLEN, SOFTMAX_CLAMP_BOUND, false>;

impl<const XLEN: usize> ClampSpec for SoftmaxClampTable<XLEN> {
    type HigherAllZero = SoftmaxClampHigherAllZeroPrefix<XLEN>;
    type HigherAllOne = ZeroPrefix<XLEN>;
    type LowerWord = SoftmaxClampLowerWordPrefix<XLEN>;
    type SufHigherAllZero = SoftmaxClampHigherAllZeroSuffix<XLEN>;
    type SufHZeroMulLWord = SoftmaxClampHZeroMulLWordSuffix<XLEN>;
    type SufHOneMulLWord = OneSuffix<XLEN>;
    const BOUND: usize = SOFTMAX_CLAMP_BOUND;
    const SYMMETRIC: bool = false;
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        lookup_tables::test::{
            lookup_table_mle_linearity_test, prefix_suffix_test_unary,
            signed_lookup_table_mle_full_hypercube_test, signed_lookup_table_mle_random_test,
        },
        subprotocols::ps_shout::unary::tests::test_read_raf_sumcheck,
    };
    use ark_bn254::Fr;
    use common::consts::XLEN;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test_unary::<XLEN, Fr, ClampTable<XLEN>>();
        prefix_suffix_test_unary::<LARGE_XLEN, Fr, SaturationTable>();
        prefix_suffix_test_unary::<XLEN, Fr, ActivationClampTable<XLEN>>();
        prefix_suffix_test_unary::<XLEN, Fr, SoftmaxClampTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        signed_lookup_table_mle_full_hypercube_test::<Fr, ClampTable<16>>();
        // `ActivationClampTable`/`SoftmaxClampTable`'s bounds are derived from `MODEL_SCALE`
        // and can reach 16 (e.g. at `MODEL_SCALE=12`), leaving no headroom against a 16-bit
        // `XLEN` here (`XLEN-BOUND-1` underflows) — exercised instead via
        // `mle_random`/`mle_linearity`/`prefix_suffix` at the real `XLEN=32`.
    }

    #[test]
    fn mle_random() {
        signed_lookup_table_mle_random_test::<Fr, ClampTable<XLEN>>();
        signed_lookup_table_mle_random_test::<Fr, SaturationTable>();
        signed_lookup_table_mle_random_test::<Fr, ActivationClampTable<XLEN>>();
        signed_lookup_table_mle_random_test::<Fr, SoftmaxClampTable<XLEN>>();
    }

    #[test]
    fn mle_linearity() {
        lookup_table_mle_linearity_test::<XLEN, Fr, ClampTable<XLEN>>();
        lookup_table_mle_linearity_test::<LARGE_XLEN, Fr, SaturationTable>();
        lookup_table_mle_linearity_test::<XLEN, Fr, ActivationClampTable<XLEN>>();
        lookup_table_mle_linearity_test::<XLEN, Fr, SoftmaxClampTable<XLEN>>();
    }

    #[test]
    fn read_raf() {
        test_read_raf_sumcheck::<ClampTable<XLEN>, XLEN>();
        test_read_raf_sumcheck::<SaturationTable, LARGE_XLEN>();
        test_read_raf_sumcheck::<ActivationClampTable<XLEN>, XLEN>();
        test_read_raf_sumcheck::<SoftmaxClampTable<XLEN>, XLEN>();
    }
}
