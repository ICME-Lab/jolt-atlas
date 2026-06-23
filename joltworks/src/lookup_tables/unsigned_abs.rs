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
        vec![Suffixes::One, Suffixes::LowerWordNoMSB, Suffixes::NegRelu]
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
        let [one, lwnm, neg_relu] = suffixes.try_into().unwrap();
        let relu = prefixes[Prefixes::NotMsb] * prefixes[Prefixes::LowerWordNoMsb] * one
            + prefixes[Prefixes::NotMsb] * lwnm;
        let neg_relu_part = prefixes[Prefixes::Msb] * prefixes[Prefixes::NotLowerWord] * one
            + prefixes[Prefixes::Msb] * neg_relu;
        relu + neg_relu_part
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [suffix_one, suffix_lwnm, suffix_neg_relu] = suffixes.try_into().unwrap();
        let [prefix_not_msb, prefix_lwnm, prefix_msb, prefix_nlw] =
            prefixes.try_into().unwrap();
        let relu = prefix_not_msb * prefix_lwnm * suffix_one + prefix_not_msb * suffix_lwnm;
        let neg_relu =
            prefix_msb * prefix_nlw * suffix_one + prefix_msb * suffix_neg_relu;
        relu + neg_relu
    }
}

#[cfg(test)]
mod test {
    use crate::lookup_tables::{
        test::{lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test_unary},
        unsigned_abs::UnsignedAbsTable,
    };
    use ark_bn254::Fr;

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
