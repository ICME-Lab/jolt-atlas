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

/// Negated ReLU lookup table: `f(x) = max(−x, 0)`.
///
/// Returns the magnitude of `x` when negative, and `0` otherwise.
/// Used in the decomposition `abs(x) = relu(x) + neg_relu(x)`.
#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct NegReluTable<const X_LEN: usize>;

impl<const X_LEN: usize> JoltLookupTable for NegReluTable<X_LEN> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let is_negative = ((index >> (X_LEN - 1)) & 1) == 1;
        if is_negative {
            // Two's complement negation: |x| = (!x % 2^X_LEN) + 1
            (!index % (1 << X_LEN)) + 1
        } else {
            0
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        // Initialize to 1 to account for the +1 in two's complement
        // negation: -x = (!x) + 1. The loop below accumulates the (!x) part
        // as sum of (1 - r_i) * 2^i, and this starting value provides the +1.
        let mut res = F::one();
        r.iter()
            .skip(X_LEN  /* skip high bits */ + 1 /* skip sign bit */)
            .rev()
            .enumerate()
            .for_each(|(i, &r_i)| res += (F::one() - r_i) * F::from_u64(i.pow2() as u64));
        let sign_bit = r[X_LEN];
        res * sign_bit
    }
}

impl<const X_LEN: usize> PrefixSuffixDecompositionTrait<X_LEN> for NegReluTable<X_LEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::NegRelu]
    }

    fn prefixes(&self) -> Vec<Prefixes> {
        vec![Prefixes::Msb, Prefixes::NotLowerWord]
    }

    #[cfg(test)]
    fn combine_test<F: JoltField>(
        &self,
        prefixes: &[PrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F {
        let [one, suffix_neg_relu] = suffixes.try_into().unwrap();
        // Safe to multiply prefix[MSB] * prefix[NotLowerWord]:
        //   - prefix[NotLowerWord] is degree-0 in the MSB variable
        //   - prefix[MSB] is degree-1 in the MSB variable and degree-0 in the lower bits
        // so their product remains degree ≤ 1 in every variable,
        // preserving the multilinear property.
        prefixes[Prefixes::Msb] * prefixes[Prefixes::NotLowerWord] * one
            + prefixes[Prefixes::Msb] * suffix_neg_relu
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [suffix_one, suffix_nlw] = suffixes.try_into().unwrap();
        let [prefix_msb, prefix_nlw] = prefixes.try_into().unwrap();
        prefix_msb * prefix_nlw * suffix_one + prefix_msb * suffix_nlw
    }
}

#[cfg(test)]
mod test {

    use crate::lookup_tables::{
        neg_relu::NegReluTable,
        test::{
            lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
        },
    };
    use ark_bn254::Fr;
    use common::consts::XLEN;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, NegReluTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, NegReluTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, NegReluTable<XLEN>>();
    }
}
