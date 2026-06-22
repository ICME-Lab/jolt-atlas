use super::{PrefixCheckpoint, PrefixCheckpoints, Prefixes, SparseDensePrefix};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

/// Prefix component for not-MSB (most significant bit) lookup table decomposition.
pub enum NotMsbPrefix2<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for NotMsbPrefix2<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &PrefixCheckpoints<F>,
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
            // sign bit in c
            0 => F::one() - F::from_u32(c),
            // sign bit in r_x
            1 => F::one() - r_x.unwrap(),
            // sign bit processed; use checkpoint.
            _ => checkpoints[Prefixes::NotMsb2].unwrap(),
        }
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &PrefixCheckpoints<F>,
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
            0 => None.into(),
            1 => Some(F::one() - r_x).into(),
            _ => checkpoints[Prefixes::NotMsb2].into(),
        }
    }
}
