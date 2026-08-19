use super::{
    prefixes::{PrefixEval, Prefixes},
    suffixes::{SuffixEval, Suffixes},
    JoltLookupTable, PrefixSuffixDecompositionTrait,
};
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use common::consts::TRIG_DOWNSCALE_BITS;
use serde::{Deserialize, Serialize};

/// Unsigned right-shift-by-[`TRIG_DOWNSCALE_BITS`] lookup table.
///
/// `RightShiftTable(x) = x >> TRIG_DOWNSCALE_BITS`, treating `x` as an unsigned `X_LEN`-bit
/// integer (no sign handling). Used by the Cos/Sin trig-table downscaling.
#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct RightShiftTable<const X_LEN: usize>;

impl<const X_LEN: usize> JoltLookupTable for RightShiftTable<X_LEN> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let index = match X_LEN {
            8 => index as u8 as u64,
            16 => index as u16 as u64,
            32 => index as u32 as u64,
            64 => index,
            _ => unimplemented!(),
        };
        index >> TRIG_DOWNSCALE_BITS
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        assert_eq!(r.len(), X_LEN);
        let mut res = F::zero();
        r.iter()
            .take(X_LEN - TRIG_DOWNSCALE_BITS as usize)
            .rev()
            .enumerate()
            .for_each(|(i, &r_i)| res += r_i * F::from_u64(1u64 << i));
        res
    }
}

impl<const X_LEN: usize> PrefixSuffixDecompositionTrait<X_LEN> for RightShiftTable<X_LEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::TrigRightShift]
    }

    fn prefixes(&self) -> Vec<Prefixes> {
        vec![Prefixes::TrigRightShift]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [prefix] = prefixes.try_into().unwrap();
        let [suffix_one, suffix] = suffixes.try_into().unwrap();
        prefix * suffix_one + suffix
    }
}

#[cfg(test)]
mod test {
    use crate::{
        lookup_tables::{
            right_shift::RightShiftTable,
            test::{
                lookup_table_mle_full_hypercube_test, lookup_table_mle_linearity_test,
                lookup_table_mle_random_test, prefix_suffix_test_unary,
            },
        },
        subprotocols::ps_shout::unary::tests::test_read_raf_sumcheck,
    };
    use ark_bn254::Fr;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test_unary::<32, Fr, RightShiftTable<32>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, RightShiftTable<16>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, RightShiftTable<64>>();
    }

    #[test]
    fn mle_linearity() {
        lookup_table_mle_linearity_test::<32, Fr, RightShiftTable<32>>();
    }

    #[test]
    fn read_raf() {
        test_read_raf_sumcheck::<RightShiftTable<32>, 32>();
    }
}
