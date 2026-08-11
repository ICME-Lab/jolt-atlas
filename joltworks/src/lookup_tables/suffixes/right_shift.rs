use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Evaluates to the value of `bits` right-shifted by `D` (i.e. the suffix-local
/// contribution of `x >> D`). Used by [`crate::lookup_tables::right_shift`].
///
/// `bits` is always the tail window of the full address (ending at the true LSB), so
/// dropping the low `D` bits is exactly `bits >> D` — no re-weighting needed.
pub enum RightShiftSuffix<const XLEN: usize, const D: usize> {}

impl<const XLEN: usize, const D: usize> SparseDenseSuffix for RightShiftSuffix<XLEN, D> {
    fn suffix_mle(bits: LookupBits) -> u32 {
        let bits_u64: u64 = bits.into();
        (bits_u64 >> D) as u32
    }
}

use common::consts::TRIG_DOWNSCALE_BITS;

/// Value of the bits with significance `>= TRIG_DOWNSCALE_BITS`, right-shifted by
/// [`TRIG_DOWNSCALE_BITS`], used by [`crate::lookup_tables::right_shift::RightShiftTable`].
pub type TrigRightShiftSuffix<const XLEN: usize> =
    RightShiftSuffix<XLEN, { TRIG_DOWNSCALE_BITS as usize }>;
