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

/// Lookup table for ReLU (Rectified Linear Unit) activation function.
///
/// Implements ReLU(x) = max(0, x) treating the input as a signed integer.
#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ReluTable<const X_LEN: usize>;

impl<const X_LEN: usize> JoltLookupTable for ReluTable<X_LEN> {
    fn materialize_entry(&self, mut index: u64) -> u64 {
        index %= 1 << X_LEN;
        match X_LEN {
            8 => 0i8.max(index as u8 as i8) as u64,
            32 => 0i32.max(index as u32 as i32) as u64,
            _ => unimplemented!(),
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        let mut res = F::zero();
        r.iter()
            .skip(X_LEN  /* skip high bits */ + 1 /* skip sign bit */)
            .rev()
            .enumerate()
            .for_each(|(i, &r_i)| res += r_i * F::from_u64(i.pow2() as u64));
        let sign_bit = r[X_LEN];
        res * (F::one() - sign_bit)
    }
}

impl<const X_LEN: usize> PrefixSuffixDecompositionTrait<X_LEN> for ReluTable<X_LEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Relu]
    }

    fn prefixes(&self) -> Vec<Prefixes> {
        vec![Prefixes::NotMsb, Prefixes::LowerWordNoMsb]
    }

    #[cfg(test)]
    fn combine_test<F: JoltField>(
        &self,
        prefixes: &[PrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F {
        let [one, relu] = suffixes.try_into().unwrap();
        prefixes[Prefixes::NotMsb] * prefixes[Prefixes::LowerWordNoMsb] * one
            + relu * prefixes[Prefixes::NotMsb]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [suffix_one, suffix_relu] = suffixes.try_into().unwrap();
        let [prefix_not_msb, prefix_lower_word_no_msb] = prefixes.try_into().unwrap();
        prefix_not_msb * prefix_lower_word_no_msb * suffix_one + prefix_not_msb * suffix_relu
    }
}

#[cfg(test)]
mod test {
    use crate::lookup_tables::{
        relu::ReluTable,
        test::{
            lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
        },
    };
    use ark_bn254::Fr;
    use common::consts::XLEN;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, ReluTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, ReluTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, ReluTable<XLEN>>();
    }
}
