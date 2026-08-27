use super::{
    prefixes::{
        less_than_const::{TrigEqConstPrefix, TrigLessThanConstPrefix},
        PrefixEval, PrefixVariant, Prefixes,
    },
    suffixes::{less_than_const::TrigLessThanConstSuffix, SuffixEval, SuffixVariant, Suffixes},
    JoltLookupTable, PrefixSuffixDecompositionTrait,
};
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use common::consts::TRIG_PERIOD_MODULUS;
use serde::{Deserialize, Serialize};

/// An unsigned `XLEN`-bit input `x`, compared against a compile-time constant `BOUND` (not
/// restricted to a power of two): `x < BOUND`. Single-operand analogue of
/// [`UnsignedLessThanTable`](super::unsigned_less_than::UnsignedLessThanTable), used to
/// range-check against a constant.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct LessThanConstTable<const XLEN: usize, const BOUND: u64>;

impl<const XLEN: usize, const BOUND: u64> JoltLookupTable for LessThanConstTable<XLEN, BOUND> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let masked = if XLEN == 64 {
            index
        } else {
            index & ((1u64 << XLEN) - 1)
        };
        (masked < BOUND).into()
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        let offset = r.len() - XLEN;

        // \sum_{i: k_i=1} (1 - x_i) * \prod_{j<i} (x_j if k_j=1 else (1-x_j))
        let mut result = F::zero();
        let mut eq = F::one();
        for i in 0..XLEN {
            let x_i: F = r[offset + i].into();
            if (BOUND >> (XLEN - 1 - i)) & 1 == 1 {
                result += (F::one() - x_i) * eq;
                eq *= x_i;
            } else {
                eq *= F::one() - x_i;
            }
        }
        result
    }
}

/// `LessThanConstTable` specialized to `BOUND = TRIG_PERIOD_MODULUS`, for the Cos/Sin
/// teleportation remainder's range-check (`remainder < TRIG_PERIOD_MODULUS`).
///
/// Not a blanket impl over arbitrary `BOUND`: `Prefixes`/`Suffixes` are static enums, so each
/// distinct `BOUND` needs its own registered variant (mirrors `ClampSpec` in `clamp.rs`). Add
/// another concrete impl like this one if a second `BOUND` is needed.
pub type TrigLessThanConstTable<const XLEN: usize> =
    LessThanConstTable<XLEN, { TRIG_PERIOD_MODULUS as u64 }>;

impl<const XLEN: usize> PrefixSuffixDecompositionTrait<XLEN> for TrigLessThanConstTable<XLEN> {
    fn prefixes(&self) -> Vec<Prefixes> {
        vec![
            TrigEqConstPrefix::<XLEN>::VARIANT,
            TrigLessThanConstPrefix::<XLEN>::VARIANT,
        ]
    }

    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, TrigLessThanConstSuffix::<XLEN>::VARIANT]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [eq, less_than] = prefixes.try_into().unwrap();
        let [one, less_than_suffix] = suffixes.try_into().unwrap();
        less_than * one + eq * less_than_suffix
    }
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
        prefix_suffix_test_unary::<XLEN, Fr, TrigLessThanConstTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, LessThanConstTable<16, 12345>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, TrigLessThanConstTable<XLEN>>();
    }

    #[test]
    fn mle_linearity() {
        lookup_table_mle_linearity_test::<XLEN, Fr, TrigLessThanConstTable<XLEN>>();
    }

    #[test]
    fn read_raf() {
        test_read_raf_sumcheck::<TrigLessThanConstTable<XLEN>, XLEN>();
    }
}
