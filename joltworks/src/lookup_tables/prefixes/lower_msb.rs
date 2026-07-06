use super::{PrefixCheckpoint, PrefixCheckpoints, Prefixes, SparseDensePrefix};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

/// Prefix component for the sign bit of the lower 32-bit word (the i32 sign bit, at
/// position XLEN/2 in the full 64-bit layout). Used in `SatClampTable` decomposition.
pub enum LowerMsbPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for LowerMsbPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &PrefixCheckpoints<F>,
        r_x: Option<C>,
        c: u32,
        _b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        match j {
            // suffix will handle
            j if j < 32 => F::one(),
            // sign bit in c
            32 => F::from_u32(c),
            // sign bit in r_x
            33 => r_x.unwrap().into(),
            // sign bit processed; use checkpoint.
            _ => checkpoints[Prefixes::LowerMsb].unwrap(),
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
            j if j <= 32 => None.into(),
            // sign bit will be in r_x when j == 33
            33 => Some(r_x.into()).into(),
            _ => checkpoints[Prefixes::LowerMsb].into(),
        }
    }
}
