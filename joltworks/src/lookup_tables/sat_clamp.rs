use super::{
    prefixes::{PrefixEval, Prefixes},
    suffixes::{SuffixEval, Suffixes},
    JoltLookupTable, PrefixSuffixDecompositionTrait,
};
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// LUT for saturating clamp of signed integers to a smaller signed integer type. For example, if XLEN=64, the LUT maps i64 values to i32 values, saturating at the min/max of i32.
/// NOTE: The input is a 64-bit signed integer, and the output is a 32-bit signed integer. The LUT is parameterized by XLEN which represents the input bit width. For XLEN=64, the LUT maps i64 values to i32 values, saturating at the min/max of i32.
#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct SatClampTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for SatClampTable<XLEN> {
    fn materialize_entry(&self, index: u64) -> u64 {
        match XLEN {
            16 => (index as i16).clamp(i8::MIN as i16, i8::MAX as i16) as u64,
            64 => (index as i64).clamp(i32::MIN as i64, i32::MAX as i64) as u64,
            _ => unimplemented!(),
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), XLEN, "Input length must be  XLEN ");
        let half_xlen = XLEN / 2;
        let m = r[half_xlen];

        // `upper_eqz`/`upper_eqo` are the MLEs of "the top `xlen` bits are all 0" / "all 1".
        // `sign_extension = (1-m)*upper_eqz + m*upper_eqo` is the MLE of "those bits are the
        // sign-extension of `m`", i.e. the value fits in the lower `xlen` bits.
        let mut upper_eqz = F::one();
        let mut upper_eqo = F::one();
        for i in 0..half_xlen {
            upper_eqz *= F::one() - r[i];
            upper_eqo *= r[i].into();
        }
        let sign_extension = (F::one() - m) * upper_eqz + m * upper_eqo;

        let (min, max) = match half_xlen {
            8 => (i8::MIN as i32, i8::MAX as i32),
            32 => (i32::MIN, i32::MAX),
            _ => unimplemented!(),
        };

        // Magnitude bits below the sign-extension check bit, as an unsigned value.
        let mut low = F::zero();
        for (i, &bit) in r[half_xlen + 1..].iter().rev().enumerate() {
            low += bit * F::from_u64(1u64 << i);
        }

        let sat_val = r[0] * F::from_i32(min) + (F::one() - r[0]) * F::from_i32(max);

        sign_extension * low - F::from_i32(max) * (F::one() - m) * upper_eqz + sat_val
    }
}

impl<const XLEN: usize> PrefixSuffixDecompositionTrait<XLEN> for SatClampTable<XLEN> {
    fn prefixes(&self) -> Vec<Prefixes> {
        vec![
            Prefixes::SatVal,
            Prefixes::UpperEqz,
            Prefixes::UpperEqo,
            Prefixes::LowerMsb,
            Prefixes::NotLowerMsb,
            Prefixes::LowerWordNoMsb,
        ]
    }

    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::One,
            Suffixes::NotLowerMsbUpperEqz,
            Suffixes::NotLowerMsbUpperEqzLow,
            Suffixes::LowerMsbUpperEqoLow,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        // The prefix-suffix decomposition hard-codes the i32 split point and uses
        // `i32::MAX`, so it is only correct for a 64-bit address. (The MLE /
        // materialization paths additionally support XLEN = 16.)
        debug_assert_eq!(
            XLEN, 64,
            "SatClampTable prefix-suffix decomposition is only valid for XLEN = 64"
        );
        let [suffix_one, suffix_not_lower_msb_upper_eqz, suffix_not_lower_msb_upper_eqz_low, suffix_lower_msb_upper_eqo_low] =
            suffixes.try_into().unwrap();
        let [prefix_sat_val, prefix_upper_eqz, prefix_upper_eqo, prefix_lower_msb, prefix_not_lower_msb, prefix_lower_word_no_msb] =
            prefixes.try_into().unwrap();
        let max = i32::MAX;

        // o = upper_eqz * (1-m) * low
        let not_lower_msb_eqz = prefix_not_lower_msb * prefix_upper_eqz;
        let o = not_lower_msb_eqz * suffix_not_lower_msb_upper_eqz_low
            + prefix_lower_word_no_msb * not_lower_msb_eqz * suffix_one;

        // p = upper_eqo * m * low
        let lower_msb_eqo = prefix_lower_msb * prefix_upper_eqo;
        let p = lower_msb_eqo * suffix_lower_msb_upper_eqo_low
            + prefix_lower_word_no_msb * lower_msb_eqo * suffix_one;

        // h = MAX * (1 - m) * upper_eqz
        let h = prefix_not_lower_msb
            * prefix_upper_eqz
            * suffix_not_lower_msb_upper_eqz
            * F::from_i32(max);

        // sv = sat_val(m) = m * MIN + (1 - m) * MAX
        let sv = prefix_sat_val * suffix_one;

        // final = o + p - h + sv
        o + p - h + sv
    }
}

#[cfg(test)]
mod test {

    use crate::{
        field::JoltField,
        lookup_tables::{
            sat_clamp::SatClampTable,
            test::{
                lookup_table_mle_linearity_test, prefix_suffix_test_unary,
                read_raf_test_unary_inner, signed_lookup_table_mle_full_hypercube_test,
                signed_lookup_table_mle_random_test,
            },
        },
        poly::opening_proof::{OpeningAccumulator, OpeningId, OpeningPoint},
        subprotocols::ps_shout::{
            unary::{PrefixSuffixShoutProvider, ReadRafClaims},
            RafShoutProvider,
        },
    };
    use ark_bn254::Fr;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test_unary::<64, Fr, SatClampTable<64>>();
    }

    #[test]
    fn mle_full_hypercube() {
        signed_lookup_table_mle_full_hypercube_test::<Fr, SatClampTable<16>>();
    }

    #[test]
    fn mle_random() {
        signed_lookup_table_mle_random_test::<Fr, SatClampTable<64>>();
    }

    #[test]
    fn linearity_test() {
        lookup_table_mle_linearity_test::<64, Fr, SatClampTable<64>>();
    }

    #[test]
    fn read_raf() {
        use crate::lookup_tables::PrefixSuffixDecompositionTrait;
        use crate::poly::opening_proof::{
            SumcheckId::{self, Raf},
            BIG_ENDIAN,
        };

        use common::VirtualPoly::{self, NodeOutput, NodeOutputRa};

        struct LookupProvider;

        impl<F: JoltField> RafShoutProvider<F> for LookupProvider {
            fn ra_poly(&self) -> (VirtualPoly, SumcheckId) {
                (NodeOutputRa(0), Raf)
            }

            fn r_cycle(
                &self,
                accumulator: &dyn OpeningAccumulator<F>,
            ) -> OpeningPoint<BIG_ENDIAN, F> {
                accumulator
                    .get_virtual_polynomial_opening(OpeningId::new(NodeOutput(1), Raf))
                    .0
            }
        }

        impl<F: JoltField, T: PrefixSuffixDecompositionTrait<64>>
            PrefixSuffixShoutProvider<F, T, 64> for LookupProvider
        {
            fn read_raf_claims(&self, accumulator: &dyn OpeningAccumulator<F>) -> ReadRafClaims<F> {
                let (_, rv_claim) =
                    accumulator.get_virtual_polynomial_opening(OpeningId::new(NodeOutput(1), Raf));
                let (_, raf_claim) =
                    accumulator.get_virtual_polynomial_opening(OpeningId::new(NodeOutput(0), Raf));

                ReadRafClaims {
                    rv_claim,
                    operand_claim: raf_claim,
                }
            }
        }

        // TODO: Currently does not really test to the full i64 extent as input is hard-coded to be sampled in i32.
        read_raf_test_unary_inner::<64, Fr, SatClampTable<64>, LookupProvider>(&LookupProvider);
    }
}
