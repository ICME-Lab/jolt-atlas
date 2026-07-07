use super::{PrefixCheckpoint, PrefixCheckpoints, Prefixes, SparseDensePrefix};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

/// Prefix component for lower word without MSB lookup table decomposition.
pub enum WordNoMsbPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for WordNoMsbPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &PrefixCheckpoints<F>,
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let mut word = checkpoints[Prefixes::WordNoMsb].unwrap_or(F::zero());
        if j >= XLEN {
            return word; // HACK: For binary ps-shout, `j` may exceed `XLEN - 1` because we go over 2*XLEN variables
        }
        let suffix_len = XLEN - j - b.len() - 1;
        match (r_x, j) {
            (None, 0) => {
                // sign bit is in c
            }
            (None, _) => {
                let x_shift = XLEN - j - 1;
                let y_shift = x_shift - 1;
                word += F::from_u64(1 << x_shift) * F::from_u32(c);
                word += F::from_u64(1 << y_shift) * F::from_u8(b.pop_msb());
            }
            (Some(r_x), _) => {
                let x_shift = XLEN - j;
                let y_shift = x_shift - 1;
                let r_x = if j == 1 { F::zero() } else { r_x.into() };
                word += F::from_u64(1 << x_shift) * r_x;
                word += F::from_u64(1 << y_shift) * F::from_u32(c);
            }
        }

        word += F::from_u64((<LookupBits as Into<u64>>::into(b)) << suffix_len);
        word
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
        let mut word = checkpoints[Prefixes::WordNoMsb].unwrap_or(F::zero());
        let x_shift = XLEN.saturating_sub(j); // HACK: For binary ps-shout, `j` may exceed `XLEN - 1` because we go over 2*XLEN variables
        let y_shift = x_shift.saturating_sub(1);
        let r_x = if j == 1 { F::zero() } else { r_x.into() };
        word += F::from_u64(1 << x_shift) * r_x + F::from_u64(1 << y_shift) * r_y;
        Some(word).into()
    }
}
