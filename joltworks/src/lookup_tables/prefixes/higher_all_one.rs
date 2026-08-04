use super::{PrefixCheckpoint, SparseDensePrefix};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    lookup_tables::prefixes::{PrefixCheckpoints, Prefixes},
    utils::lookup_bits::LookupBits,
};

/// Prefix that evaluates to 1 iff all bits with significance >= 2^BOUND, sign bit included,
/// are one. Mirrors [`super::higher_all_zero::HigherAllZeroPrefix`], checking for 1 instead of 0.
pub enum HigherAllOnePrefix<const XLEN: usize, const BOUND: usize, const CP_INDEX: usize> {}

impl<const XLEN: usize, const BOUND: usize, const CP_INDEX: usize, F: JoltField>
    SparseDensePrefix<F> for HigherAllOnePrefix<XLEN, BOUND, CP_INDEX>
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
        // See `HigherAllZeroPrefix::prefix_mle` for why this guard is needed: this prefix
        // type is shared across every table's checkpoint bookkeeping, so it must tolerate
        // being instantiated at an XLEN smaller than its own BOUND without panicking.
        if BOUND >= XLEN {
            return F::one();
        }

        let bound_index = XLEN - BOUND - 1;
        let mut result = checkpoints[CP_INDEX].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            if j > 0 {
                let prev_index = j - 1;
                if prev_index <= bound_index {
                    result = result * r_x;
                }
            }
            if j <= bound_index {
                result *= F::from_u32(c);
            }
        } else if j <= bound_index {
            result *= F::from_u32(c);
        }

        let b_len = b.len();
        let b_u64: u64 = b.into();
        for pos in 0..b_len {
            let global_index = j + 1 + pos;
            if global_index <= bound_index {
                let bit = ((b_u64 >> (b_len - 1 - pos)) & 1) as u8;
                result *= F::from_u8(bit);
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
        // See the guard in `prefix_mle` above for why this is needed.
        if BOUND >= XLEN {
            return Some(F::one()).into();
        }

        let bound_index = XLEN - BOUND - 1;
        let mut result = checkpoints[CP_INDEX].unwrap_or(F::one());

        if j > 0 {
            let prev_index = j - 1;
            if prev_index <= bound_index {
                result = result * r_x;
            }
        }
        if j <= bound_index {
            result = result * r_y;
        }

        Some(result).into()
    }
}

use crate::lookup_tables::clamp::{CLAMP_BOUND, SIGN_BIT_I32};
use common::consts::ACTIVATION_BOUND;

pub type ClampHigherAllOnePrefix<const XLEN: usize> =
    HigherAllOnePrefix<XLEN, CLAMP_BOUND, { Prefixes::ClampHigherAllOne as usize }>;

pub type SatClampHigherAllOnePrefix<const XLEN: usize> =
    HigherAllOnePrefix<XLEN, SIGN_BIT_I32, { Prefixes::SatClampHigherAllOne as usize }>;

pub type ActivationHigherAllOnePrefix<const XLEN: usize> =
    HigherAllOnePrefix<XLEN, ACTIVATION_BOUND, { Prefixes::ActivationHigherAllOne as usize }>;
