use super::{PrefixCheckpoint, SparseDensePrefix};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    lookup_tables::prefixes::{PrefixCheckpoints, Prefixes},
    utils::lookup_bits::LookupBits,
};

/// Prefix that evaluates to 1 iff all bits with significance >= 2^BOUND are zero.
pub enum HigherIsZeroPrefix<const XLEN: usize, const BOUND: usize, const CP_INDEX: usize> {}

impl<const XLEN: usize, const BOUND: usize, const CP_INDEX: usize, F: JoltField>
    SparseDensePrefix<F> for HigherIsZeroPrefix<XLEN, BOUND, CP_INDEX>
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
        let bound_index = XLEN - BOUND - 1;
        let mut result = checkpoints[CP_INDEX].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            if j > 0 {
                let prev_index = j - 1;
                if prev_index <= bound_index {
                    result *= F::one() - r_x;
                }
            }
            if j <= bound_index {
                result *= F::one() - F::from_u32(c);
            }
        } else if j <= bound_index {
            result *= F::one() - F::from_u32(c);
        }

        let b_len = b.len();
        let b_u64: u64 = b.into();
        for pos in 0..b_len {
            let global_index = j + 1 + pos;
            if global_index <= bound_index {
                let bit = ((b_u64 >> (b_len - 1 - pos)) & 1) as u8;
                result *= F::one() - F::from_u8(bit);
            }
        }

        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &PrefixCheckpoints<F>,
        r_x: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let bound_index = XLEN - BOUND - 1;
        let mut result = checkpoints[CP_INDEX].unwrap_or(F::one());

        if j > 0 {
            let prev_index = j - 1;
            if prev_index <= bound_index {
                result *= F::one() - r_x;
            }
        }
        if j <= bound_index {
            result *= F::one() - r_y;
        }

        Some(result).into()
    }
}

use crate::lookup_tables::clamp::CLAMP_BOUND;

pub type ClampHigherIsZeroPrefix<const XLEN: usize> =
    HigherIsZeroPrefix<XLEN, CLAMP_BOUND, { Prefixes::ClampHigherIsZero as usize }>;
