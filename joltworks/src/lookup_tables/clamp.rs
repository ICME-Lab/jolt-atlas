use std::fmt::Debug;

use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    lookup_tables::{
        prefixes::{
            higher_is_zero::{ClampHigherIsZeroPrefix, SatClampHigherIsZeroPrefix},
            lower_word::{ClampLowerWordPrefix, SatClampLowerWordPrefix},
            PrefixEval, PrefixVariant, Prefixes,
        },
        suffixes::{
            higher_is_zero::{ClampHigherIsZeroSuffix, SatClampHigherIsZeroSuffix},
            hzero_mul_lword::{ClampHZeroMulLWordSuffix, SatClampHZeroMulLWordSuffix},
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
        vec![
            T::HigherIsZero::VARIANT,
            T::LowerWord::VARIANT,
            Prefixes::Msb,
        ]
    }

    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            T::SufHigherIsZero::VARIANT,
            T::SufHZeroMulLWord::VARIANT,
            Suffixes::One,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let const_upper = F::from_u64((1u64 << Self::BOUND) - 1);

        let [pre_higher_is_zero, pre_lower_word, pre_msb] = prefixes.try_into().unwrap();
        let [suf_higher_is_zero, suf_hzero_mul_lword, suf_one] = suffixes.try_into().unwrap();

        // Default to the upper bound
        suf_one * const_upper
        // If the input is < 2^BOUND, add lower word and cancel upper bound
            + pre_higher_is_zero
                * (suf_hzero_mul_lword + pre_lower_word * suf_one
                    - suf_higher_is_zero * const_upper)
        // If the input is negative, cancel upper bound
            - pre_msb * suf_one * const_upper
    }
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ClampBoundedTable<const XLEN: usize, const BOUND: usize>;

impl<const XLEN: usize, const BOUND: usize> JoltLookupTable for ClampBoundedTable<XLEN, BOUND> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let val: i64 = match XLEN {
            8 => index as u8 as i8 as i64,
            16 => index as u16 as i16 as i64,
            32 => index as u32 as i32 as i64,
            64 => index as i64,
            _ => unimplemented!(),
        };
        if val < 0 {
            0
        } else if val >= 1 << BOUND {
            (1 << BOUND) - 1
        } else {
            val as u64
        }
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

        // Indicator that all bits with significance >= 2^BOUND are zero.
        let is_higher_is_zero: F = indexed_r[..ubound_index + 1]
            .iter()
            .map(|(_, &r_i)| F::one() - r_i)
            .product();

        // Word value from bits with significance < 2^BOUND.
        let lower_word: F = indexed_r[ubound_index + 1..]
            .iter()
            .map(|(i, &r_i)| {
                let exponent = XLEN - i - 1;
                r_i * F::from_u64(1 << exponent)
            })
            .sum();

        let const_upper = F::from_i64((1 << BOUND) - 1);

        // Defaults to the upper bound
        const_upper
        // If input < 2^BOUND (all high-significance bits = 0),
        // add lower word and cancel upper bound
            + is_higher_is_zero * (lower_word - const_upper)
            // If input is negative, cancel upper bound
            - msb * const_upper
    }
}

/// The effective bound of the ONNX `Clamp` op: it clamps into `[-2^CLAMP_BOUND, 2^CLAMP_BOUND - 1]`
/// (see `jolt_atlas_core::onnx_proof::ops::clamp`, which offsets the input by `2^CLAMP_BOUND`
/// to map that range onto this table's `[0, 2^CLAMP_TABLE_BOUND)` floor-at-0 domain).
pub const CLAMP_BOUND: usize = 9;

/// The underlying floor-at-0 lookup table's own bound, one more than [`CLAMP_BOUND`] so the
/// offset range fits exactly. This bound is also intended for future reuse by other
/// saturating operators (e.g. Tanh/Sigmoid/Erf, saturating Addition/Einsum) sharing the same
/// table shape.
pub const CLAMP_TABLE_BOUND: usize = CLAMP_BOUND + 1;

pub type ClampTable<const XLEN: usize> = ClampBoundedTable<XLEN, CLAMP_TABLE_BOUND>;

// Implementation of ClampSpec for the clamp table (upper-only: no lower correction).
impl<const XLEN: usize> ClampSpec for ClampTable<XLEN> {
    type HigherIsZero = ClampHigherIsZeroPrefix<XLEN>;
    type LowerWord = ClampLowerWordPrefix<XLEN>;
    type SufHigherIsZero = ClampHigherIsZeroSuffix<XLEN>;
    type SufHZeroMulLWord = ClampHZeroMulLWordSuffix<XLEN>;
    const BOUND: usize = CLAMP_TABLE_BOUND;
}

/// `ClampBoundedTable` instantiated to replicate `SatClampTable<64>`'s saturation shape:
/// half-open floor-at-0/ceiling-at-(2^32 - 1) over the raw i64 index, intended to be
/// used with an (external, wrapper-level) offset of 2^31 -- the same offset-trick
/// pattern `SymmetricClampOperands` already uses for the ONNX `Clamp` op -- to reproduce
/// a saturating clamp to i32's exact range: [-2^31, 2^31 - 1] spans exactly 2^32 values,
/// which this half-open BOUND=32 table represents with zero waste.
pub type SatClampViaClampTable = ClampBoundedTable<64, 32>;

impl ClampSpec for SatClampViaClampTable {
    type HigherIsZero = SatClampHigherIsZeroPrefix<64>;
    type LowerWord = SatClampLowerWordPrefix<64>;
    type SufHigherIsZero = SatClampHigherIsZeroSuffix<64>;
    type SufHZeroMulLWord = SatClampHZeroMulLWordSuffix<64>;
    const BOUND: usize = 32;
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
        lookup_table_mle_random_test::<Fr, ClampTable<XLEN>>();
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

#[cfg(test)]
mod sat_clamp_via_clamp_test {
    use super::*;
    use crate::{
        lookup_tables::{
            sat_clamp::SatClampTable,
            test::{lookup_table_mle_random_test, prefix_suffix_test_unary},
        },
        subprotocols::ps_shout::unary::tests::test_read_raf_sumcheck,
    };
    use ark_bn254::Fr;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    const OFFSET: i64 = 1i64 << 31;

    /// Applies the same offset trick `SymmetricClampOperands` uses in production: shift the
    /// input by `2^31` into `SatClampViaClampTable`'s floor-at-0 domain, then shift the result
    /// back. Only valid while `val + OFFSET` doesn't overflow `i64`, which holds for any input
    /// within a wide margin of `i32`'s range -- all that matters for saturation behavior.
    // `SatClampViaClampTable` is a type alias, not a struct definition, so clippy's
    // `default_constructed_unit_structs` suggestion (dropping `::default()`) doesn't apply here.
    #[allow(clippy::default_constructed_unit_structs)]
    fn sat_via_clamp(val: i64) -> i64 {
        let shifted = (val + OFFSET) as u64;
        SatClampViaClampTable::default().materialize_entry(shifted) as i64 - OFFSET
    }

    #[test]
    fn materialize_matches_sat_clamp_table() {
        let expected = |val: i64| SatClampTable::<64>.materialize_entry(val as u64) as i64;

        for val in [
            i32::MIN as i64 - 1,
            i32::MIN as i64,
            i32::MAX as i64,
            i32::MAX as i64 + 1,
            0,
        ] {
            assert_eq!(sat_via_clamp(val), expected(val), "mismatch at val={val}");
        }

        let mut rng = StdRng::seed_from_u64(0x5a7c);
        for _ in 0..10_000 {
            // Sampled well within a margin that keeps `val + OFFSET` from overflowing `i64`,
            // while still spanning far beyond `i32`'s range on both sides.
            let val: i64 = rng.gen_range(-(1i64 << 40)..(1i64 << 40));
            assert_eq!(sat_via_clamp(val), expected(val), "mismatch at val={val}");
        }
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, SatClampViaClampTable>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test_unary::<64, Fr, SatClampViaClampTable>();
    }

    #[test]
    fn read_raf() {
        test_read_raf_sumcheck::<SatClampViaClampTable, 64>();
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::{
        lookup_tables::sat_clamp::SatClampTable,
        subprotocols::ps_shout::unary::tests::test_read_raf_sumcheck,
    };
    use std::time::{Duration, Instant};

    const ITERATIONS: u32 = 20;

    /// Runs `f` `ITERATIONS` times and prints the averaged wall-clock time per call
    /// (trace generation + prove + verify combined -- `test_read_raf_sumcheck` doesn't
    /// expose prove/verify separately).
    fn bench(label: &str, f: impl Fn()) {
        let mut total = Duration::ZERO;
        for _ in 0..ITERATIONS {
            let start = Instant::now();
            f();
            total += start.elapsed();
        }

        println!(
            "{label}: avg {:.2?} over {ITERATIONS} iterations (total {:.2?})",
            total / ITERATIONS,
            total,
        );
    }

    /// Not a correctness test: run only to eyeball timing.
    #[test]
    #[ignore]
    fn bench_sat_clamp_via_clamp() {
        bench("SatClampViaClampTable", test_read_raf_sumcheck::<SatClampViaClampTable, 64>);
    }

    /// Not a correctness test: run only to eyeball timing.
    #[test]
    #[ignore]
    fn bench_sat_clamp() {
        bench("SatClampTable<64>", test_read_raf_sumcheck::<SatClampTable<64>, 64>);
    }
}
