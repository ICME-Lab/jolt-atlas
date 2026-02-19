use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use joltworks::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

/// Prefix component for not-MSB (most significant bit) lookup table decomposition.
pub enum NotMsbPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for NotMsbPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut _b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        match j {
            // suffix will handle
            j if j < XLEN => F::one(),
            // sign bit in c
            j if j == XLEN => F::one() - F::from_u32(c),
            // sign bit in r_x
            j if j == XLEN + 1 => F::one() - r_x.unwrap(),
            // sign bit processed; use checkpoint.
            _ => checkpoints[Prefixes::NotMsb].unwrap(),
        }
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        _r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        match j {
            j if j < XLEN => None.into(),
            // sign bit will be in r_x when j == XLEN + 1
            j if j == XLEN + 1 => Some(F::one() - r_x).into(),
            _ => checkpoints[Prefixes::NotMsb].into(),
        }
    }
}
