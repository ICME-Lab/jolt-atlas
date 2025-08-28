use itertools::Itertools;
use jolt_core::poly::eq_poly::EqPolynomial;
use serde::{Deserialize, Serialize};

use jolt_core::jolt::lookup_table::prefixes::{PrefixEval, Prefixes};
use jolt_core::jolt::lookup_table::suffixes::{SuffixEval, Suffixes};
use jolt_core::jolt::lookup_table::JoltLookupTable;
use jolt_core::jolt::lookup_table::PrefixSuffixDecomposition;
use jolt_core::field::JoltField;

use crate::jolt::instruction::sigmoid::SCALE;

const LUT_SIZE: usize = 112;
pub const SIGMOID_SCALED_TABLE: [u8; LUT_SIZE] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2,
    2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5,
    5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
];


// const EXPECTED_WORD_SIZE: usize = 32;
// fn build_sigmoid_table() -> Vec<u8> {
//     let size = 1 << EXPECTED_WORD_SIZE;
//     let mut lut_l = vec![7u8; size / 2];
//     let lut_r = vec![0u8; size / 2];
//     lut_l.extend_from_slice(&lut_r);
//     for i in 0..LUT_SIZE / 2 {
//         lut_l[i] = SIGMOID_SCALED_TABLE[i + LUT_SIZE / 2];
//         lut_l[size - i - 1] = SIGMOID_SCALED_TABLE[LUT_SIZE / 2 - i - 1];
//     }
//     lut_l
// }

// pub static SIGMOID_EXPANDED_SCALED_TABLE: once_cell::sync::Lazy<Vec<u8>> =
//     once_cell::sync::Lazy::new(build_sigmoid_table);


#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SigmoidTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for SigmoidTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u64) -> u64 {
        if index < (LUT_SIZE / 2) as u64 {
            SIGMOID_SCALED_TABLE[index as usize + LUT_SIZE / 2] as u64
        } else if index > u64::MAX - (LUT_SIZE / 2) as u64 {
            SIGMOID_SCALED_TABLE[LUT_SIZE / 2 - (index as u8 as usize) - 1] as u64
        } else if index < u64::MAX / 2 {
            SCALE as u64
        } else {
            0
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);
        // TODO: We only need the first half (only one input to sigmoid)
        // let r_prime = &r[..r.len() / 2];
        let r_prime = r;
        let len = 1 << r_prime.len();
        let max = F::from_u8(SCALE as u8);
        let zero = F::zero();
        let mut f_eval = vec![max; len as usize / 2];
        let f_eval_r = vec![zero; len as usize / 2];
        f_eval.extend_from_slice(&f_eval_r);
        for i in 0..LUT_SIZE / 2 {
            f_eval[i] = F::from_u8(SIGMOID_SCALED_TABLE[i + LUT_SIZE / 2]);
            f_eval[len as usize - i - 1] = F::from_u8(SIGMOID_SCALED_TABLE[LUT_SIZE / 2 - i - 1]);
        }

        // TODO: EqPolynomial::evals(r) is too slow
        let eq_evals = EqPolynomial::evals(r_prime);
        f_eval
            .iter()
            .zip_eq(eq_evals.iter())
            .map(|(x, e)| *x * e)
            .sum()
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for SigmoidTable<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LowerWord]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, lower_word] = suffixes.try_into().unwrap();
        prefixes[Prefixes::LowerWord] * one + lower_word
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::SigmoidTable;
    use jolt_core::jolt::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, SigmoidTable<32>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, SigmoidTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, SigmoidTable<32>>();
    }
}
