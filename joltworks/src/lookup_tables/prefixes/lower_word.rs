use super::{PrefixCheckpoint, SparseDensePrefix};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    lookup_tables::prefixes::{PrefixCheckpoints, Prefixes},
    utils::lookup_bits::LookupBits,
};

pub enum LowerWordPrefix<const XLEN: usize, const BOUND: usize, const CP_INDEX: usize> {}

impl<const XLEN: usize, const BOUND: usize, const CP_INDEX: usize, F: JoltField>
    SparseDensePrefix<F> for LowerWordPrefix<XLEN, BOUND, CP_INDEX>
{
    fn prefix_mle<C>(
        checkpoints: &PrefixCheckpoints<F>,
        r_x: Option<C>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        // j + b.len() is always >= XLEN in binary setups, where this prefix is unused.
        // In binary setups, getting the exponent is an operation that can underflow, so we exit early.
        if j + b.len() >= XLEN {
            return F::zero();
        }

        let lbound_index = XLEN - BOUND - 1;
        let mut result = checkpoints[CP_INDEX].unwrap_or(F::zero());

        if let Some(r_x) = r_x {
            if j > 0 {
                let prev_index = j - 1;
                if prev_index > lbound_index {
                    let exponent = XLEN - prev_index - 1;
                    result += F::from_u64(1 << exponent) * r_x;
                }
            }
            if j > lbound_index {
                let exponent = XLEN - j - 1;
                result += F::from_u64(1 << exponent) * F::from_u32(c);
            }
        } else if j > lbound_index {
            let exponent = XLEN - j - 1;
            result += F::from_u64(1 << exponent) * F::from_u32(c);
        }

        let b_len = b.len();
        let b_u64: u64 = b.into();
        for pos in 0..b_len {
            let global_index = j + 1 + pos;
            if global_index > lbound_index {
                let exponent = XLEN - global_index - 1;
                let bit = ((b_u64 >> (b_len - 1 - pos)) & 1) as u8;
                result += F::from_u64(1 << exponent) * F::from_u8(bit);
            }
        }

        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &PrefixCheckpoints<F>,
        r_x: C,
        r_y: C,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        // j + suffix_len is always >= XLEN in binary setups, where this prefix is unused.
        // In binary setups, getting the exponent is an operation that can underflow, so we exit early.
        if j + suffix_len >= XLEN {
            return None.into();
        }

        let lbound_index = XLEN - BOUND - 1;
        let mut result = checkpoints[CP_INDEX].unwrap_or(F::zero());

        if j > 0 {
            let prev_index = j - 1;
            if prev_index > lbound_index {
                let exponent = XLEN - prev_index - 1;
                result += F::from_u64(1 << exponent) * r_x;
            }
        }
        if j > lbound_index {
            let exponent = XLEN - j - 1;
            result += F::from_u64(1 << exponent) * r_y;
        }

        Some(result).into()
    }
}

use crate::lookup_tables::clamp::CLAMP_BOUND;

pub type ClampLowerWordPrefix<const XLEN: usize> =
    LowerWordPrefix<XLEN, CLAMP_BOUND, { Prefixes::ClampLowerWord as usize }>;
