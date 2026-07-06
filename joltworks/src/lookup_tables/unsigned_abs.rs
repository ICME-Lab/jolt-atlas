use super::{
    prefixes::{PrefixEval, Prefixes},
    suffixes::{SuffixEval, Suffixes},
    JoltLookupTable, PrefixSuffixDecompositionTrait,
};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::math::Math,
};
use serde::{Deserialize, Serialize};

/// Unsigned absolute value lookup table for the XLEN-bit layout (sign at position 0).
///
/// `abs(x) = relu(x) + neg_relu(x)` where `x` is a signed integer.
/// Uses the prefix system where the sign bit is the first variable (bit 0).
#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct UnsignedAbsTable<const X_LEN: usize>;

impl<const X_LEN: usize> JoltLookupTable for UnsignedAbsTable<X_LEN> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let sign_bit: bool = ((index >> (X_LEN - 1)) & 1) == 1;
        if sign_bit {
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
        debug_assert_eq!(r.len(), X_LEN);
        // magnitude = sum_{i=1}^{X_LEN-1} r[i] * 2^(X_LEN-1-i)
        let mut magnitude = F::zero();
        r.iter()
            .skip(1)
            .rev()
            .enumerate()
            .for_each(|(i, &r_i)| magnitude += r_i * F::from_u64(i.pow2() as u64));
        // not_magnitude = 2^(X_LEN-1) - magnitude  (the neg_relu contribution)
        let not_magnitude = F::from_u64(1 << (X_LEN - 1)) - magnitude;
        let sign_bit = r[0];
        magnitude * (F::one() - sign_bit) + not_magnitude * sign_bit
    }
}

impl<const X_LEN: usize> PrefixSuffixDecompositionTrait<X_LEN> for UnsignedAbsTable<X_LEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::WordNoMSB, Suffixes::NegRelu]
    }

    fn prefixes(&self) -> Vec<Prefixes> {
        vec![
            Prefixes::NotMsb,
            Prefixes::WordNoMsb,
            Prefixes::Msb,
            Prefixes::NotWordNoMsb,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [suffix_one, suffix_lwnm, suffix_neg_relu] = suffixes.try_into().unwrap();
        let [prefix_not_msb, prefix_lwnm, prefix_msb, prefix_nlw] = prefixes.try_into().unwrap();
        let relu = prefix_not_msb * prefix_lwnm * suffix_one + prefix_not_msb * suffix_lwnm;
        let neg_relu = prefix_msb * prefix_nlw * suffix_one + prefix_msb * suffix_neg_relu;
        relu + neg_relu
    }
}

#[cfg(test)]
mod test {
    use crate::lookup_tables::{
        test::{
            lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test,
            prefix_suffix_test_unary,
        },
        unsigned_abs::UnsignedAbsTable,
        JoltLookupTable,
    };
    use ark_bn254::Fr;

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
        prefix_suffix_test_unary::<32, Fr, UnsignedAbsTable<32>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, UnsignedAbsTable<16>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, UnsignedAbsTable<64>>();
    }
}
