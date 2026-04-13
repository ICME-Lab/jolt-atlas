use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

/// Prefix component for two's complement negation: `(!lower_word) + 1`.
///
/// Accumulates the bitwise-NOT of the magnitude bits (lower word, excluding the sign bit)
/// plus one, used in the `neg_relu` prefix-suffix decomposition.
pub enum NotLowerWordPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for NotLowerWordPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        // ignore high order variables and sign bit
        if j < XLEN {
            return F::zero();
        }
        let jj = j - XLEN;
        let mut word = checkpoints[Prefixes::NotLowerWord].unwrap_or(F::one());
        let suffix_len = XLEN * 2 - j - b.len() - 1;
        match (r_x, j) {
            (None, jjj) if jjj == XLEN => {
                // sign bit is in c
            }
            (None, _) => {
                let x_shift = XLEN - jj - 1;
                let y_shift = x_shift - 1;
                word += F::from_u64(1 << x_shift) * (F::one() - F::from_u32(c));
                word += F::from_u64(1 << y_shift) * (F::one() - F::from_u8(b.pop_msb()));
            }
            (Some(r_x), _) => {
                let x_shift = XLEN - jj;
                let y_shift = x_shift - 1;
                let not_r_x = if j == (XLEN + 1) {
                    F::zero()
                } else {
                    F::one() - r_x.into()
                };
                word += F::from_u64(1 << x_shift) * not_r_x;
                word += F::from_u64(1 << y_shift) * (F::one() - F::from_u32(c));
            }
        }

        word += F::from_u64((<LookupBits as Into<u64>>::into(!b)) << suffix_len);
        word
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        // ignore high order variables
        if j < XLEN {
            return None.into();
        }

        let mut word = checkpoints[Prefixes::NotLowerWord].unwrap_or(F::one());
        let jj = j - XLEN;
        let x_shift = XLEN - jj;
        let y_shift = x_shift - 1;
        let not_r_x = if j == (XLEN + 1) {
            F::zero()
        } else {
            F::one() - r_x.into()
        };
        word += F::from_u64(1 << x_shift) * not_r_x + F::from_u64(1 << y_shift) * (F::one() - r_y);
        Some(word).into()
    }
}
