use super::{PrefixCheckpoint, SparseDensePrefix};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    lookup_tables::prefixes::{PrefixCheckpoints, Prefixes},
    utils::lookup_bits::LookupBits,
};

/// Mirror image of [`super::lower_word::LowerWordPrefix`]: instead of the value of the low
/// `BOUND` bits, computes the value of the high `XLEN - D` bits, right-shifted by `D` (i.e.
/// `x >> D` for an unsigned `XLEN`-bit `x`). Used by [`crate::lookup_tables::right_shift`].
pub enum RightShiftPrefix<const XLEN: usize, const D: usize, const CP_INDEX: usize> {}

impl<const XLEN: usize, const D: usize, const CP_INDEX: usize, F: JoltField> SparseDensePrefix<F>
    for RightShiftPrefix<XLEN, D, CP_INDEX>
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
        // See `LowerWordPrefix::prefix_mle` for why these guards are needed: this prefix is
        // shared checkpoint bookkeeping across every table, so it must tolerate being
        // instantiated at an XLEN smaller than D without panicking.
        if j + b.len() >= XLEN || D >= XLEN {
            return F::zero();
        }

        // Largest global bit index (0 = MSB) that still contributes after shifting off the
        // low `D` bits.
        let ubound_index = XLEN - D - 1;
        let mut result = checkpoints[CP_INDEX].unwrap_or(F::zero());

        if let Some(r_x) = r_x {
            if j > 0 {
                let prev_index = j - 1;
                if prev_index <= ubound_index {
                    let exponent = ubound_index - prev_index;
                    result += F::from_u64(1 << exponent) * r_x;
                }
            }
            if j <= ubound_index {
                let exponent = ubound_index - j;
                result += F::from_u64(1 << exponent) * F::from_u32(c);
            }
        } else if j <= ubound_index {
            let exponent = ubound_index - j;
            result += F::from_u64(1 << exponent) * F::from_u32(c);
        }

        // `b` doesn't necessarily reach the true LSB (it's phase-local), so first shift
        // it into its true global position, then right-shift by `D`; the guard above
        // guarantees `left_shift >= 0`.
        let b_len = b.len();
        let b_u64: u64 = b.into();
        let left_shift = XLEN - 1 - j - b_len;
        result += F::from_u64((b_u64 << left_shift) >> D);

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
        if j + suffix_len >= XLEN || D >= XLEN {
            return None.into();
        }

        let ubound_index = XLEN - D - 1;
        let mut result = checkpoints[CP_INDEX].unwrap_or(F::zero());

        if j > 0 {
            let prev_index = j - 1;
            if prev_index <= ubound_index {
                let exponent = ubound_index - prev_index;
                result += F::from_u64(1 << exponent) * r_x;
            }
        }
        if j <= ubound_index {
            let exponent = ubound_index - j;
            result += F::from_u64(1 << exponent) * r_y;
        }

        Some(result).into()
    }
}

use common::consts::TRIG_DOWNSCALE_BITS;

/// Value of the high `XLEN - TRIG_DOWNSCALE_BITS` bits of an unsigned `XLEN`-bit input,
/// right-shifted by [`TRIG_DOWNSCALE_BITS`], used by [`crate::lookup_tables::right_shift::RightShiftTable`].
pub type TrigRightShiftPrefix<const XLEN: usize> =
    RightShiftPrefix<XLEN, { TRIG_DOWNSCALE_BITS as usize }, { Prefixes::TrigRightShift as usize }>;
