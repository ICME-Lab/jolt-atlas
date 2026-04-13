use std::fmt::Debug;

use super::{
    prefixes::{PrefixEval, Prefixes},
    suffixes::{SuffixEval, Suffixes},
    JoltLookupTable, PrefixSuffixDecompositionTrait,
};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    lookup_tables::{neg_relu::NegReluTable, relu::ReluTable},
    utils::math::Math,
};
use serde::{Deserialize, Serialize};

/// Unsigned absolute value lookup table: `f(x) = |x| = relu(x) + relu(-x)`.
///
/// Decomposes into [`ReluTable`] and [`NegReluTable`] for prefix-suffix proving.
#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct UnsignedAbsTable<const X_LEN: usize>;

impl<const X_LEN: usize> JoltLookupTable for UnsignedAbsTable<X_LEN> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let sign_bit: bool = ((index >> (X_LEN - 1)) & 1) == 1;
        if sign_bit {
            // Two's complement negation: -x = (!x) + 1
            // We reduce modulo 2^(X_LEN-1) first to extract the lower magnitude bits
            // from the 64-bit NOT, then add 1. This ordering is important: for i8::MIN
            // (magnitude bits all zero), !L = 2^(X_LEN-1) - 1, so (!L % 2^(X_LEN-1)) + 1
            // correctly yields 2^(X_LEN-1) (= 128 for X_LEN=8).
            (!index) % (1 << (X_LEN - 1)) + 1
        } else {
            index % (1 << (X_LEN - 1))
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * X_LEN);

        let mut positive_case = F::zero();

        // if x < 0, abs(x) = -x = (!x) + 1
        let mut negative_case = F::one();

        r.iter()
            .skip(X_LEN  /* skip high bits */ + 1 /* skip sign bit */)
            .rev()
            .enumerate()
            .for_each(|(i, &r_i)| {
                positive_case += r_i * F::from_u64(i.pow2() as u64);
                negative_case += (F::one() - r_i) * F::from_u64(i.pow2() as u64)
            });

        let sign_bit = r[X_LEN];
        positive_case * (F::one() - sign_bit) + negative_case * sign_bit
    }
}

impl<const X_LEN: usize> PrefixSuffixDecompositionTrait<X_LEN> for UnsignedAbsTable<X_LEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Relu, Suffixes::NegRelu]
    }

    fn prefixes(&self) -> Vec<Prefixes> {
        vec![
            Prefixes::NotMsb,
            Prefixes::LowerWordNoMsb,
            Prefixes::Msb,
            Prefixes::NotLowerWord,
        ]
    }

    #[cfg(test)]
    fn combine_test<F: JoltField>(
        &self,
        prefixes: &[PrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F {
        let [one, relu, neg_relu] = suffixes.try_into().unwrap();
        let relu = ReluTable::<X_LEN>.combine_test(prefixes, &[one, relu]);
        let neg_relu = NegReluTable::<X_LEN>.combine_test(prefixes, &[one, neg_relu]);
        relu + neg_relu
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [suffix_one, suffix_relu, suffix_neg_relu] = suffixes.try_into().unwrap();
        let [prefix_not_msb, prefix_lower_word_no_msb, prefix_msb, prefix_nlw] =
            prefixes.try_into().unwrap();
        let relu = ReluTable::<X_LEN>.combine(
            &[prefix_not_msb, prefix_lower_word_no_msb],
            &[suffix_one, suffix_relu],
        );
        let neg_relu = NegReluTable::<X_LEN>
            .combine(&[prefix_msb, prefix_nlw], &[suffix_one, suffix_neg_relu]);
        relu + neg_relu
    }
}

#[cfg(test)]
mod test {

    use crate::lookup_tables::{
        test::{
            lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
        },
        unsigned_abs::UnsignedAbsTable,
        JoltLookupTable,
    };
    use ark_bn254::Fr;
    use common::consts::XLEN;

    fn unsigned_abs(x: i8) -> u8 {
        x.unsigned_abs()
    }

    /// Verify the lookup table matches `i8::unsigned_abs()` for all 256 inputs.
    ///
    /// Key edge cases (two's complement, X_LEN=8):
    ///
    /// | Value | Binary      | Abs |
    /// |-------|-------------|-----|
    /// |   127 | `0111_1111` | 127 |
    /// |  -128 | `1000_0000` | 128 |
    /// |    -1 | `1111_1111` |   1 |
    /// |     0 | `0000_0000` |   0 |
    ///
    /// See the `i8::MIN` assertion below for why this table returns 128
    /// (not 0) and how that differs from ONNX semantics.
    #[test]
    fn test_unsigned_abs_table() {
        let table = UnsignedAbsTable::<8>;
        let materialized_table = table.materialize();

        // Exhaustive check against Rust's `i8::unsigned_abs()`
        for i in 0..256usize {
            assert_eq!(unsigned_abs(i as i8), materialized_table[i] as u8);
        }

        // Explicit edge cases
        assert_eq!(table.materialize_entry(0u64), 0, "abs(0) = 0");
        assert_eq!(table.materialize_entry(127u64), 127, "abs(127) = 127");
        assert_eq!(table.materialize_entry(0xFF), 1, "abs(-1) = 1");

        // i8::MIN = -128 (0b1000_0000): the critical edge case.
        //
        // NOTE: Our table returns 128 (not 0), because this table is used for
        // internal proof computations where outputs are unsigned integers
        // where we want int::MIN to return its canonical abs value in the
        // field, NOT for proving ONNX `Abs` directly. For ONNX compliance,
        // 128 doesn't fit in i8 (or negate(i32::MIN) doesn't fit in i32 in
        // the real impl), so MIN would need clamped output. One approach:
        // replace `!x + 1` with `!x + (1 - eqz)` where `eqz` is 1 when the
        // lower X_LEN-1 magnitude bits (i.e., everything except the sign bit)
        // are all zero — which uniquely identifies MIN. This skips the +1
        // only for MIN, clamping its abs to INT_MAX (127 for i8).
        assert_eq!(
            table.materialize_entry(0x80),
            128,
            "abs(i8::MIN) must be 128, not 0: field arithmetic avoids two's complement overflow"
        );
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, UnsignedAbsTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, UnsignedAbsTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, UnsignedAbsTable<XLEN>>();
    }
}
