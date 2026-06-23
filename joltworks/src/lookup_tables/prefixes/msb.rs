use super::{PrefixCheckpoint, PrefixCheckpoints, Prefixes, SparseDensePrefix};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

/// Prefix for the MSB (sign bit) in the XLEN-bit layout where sign is at position 0.
///
/// Sign bit is the first variable (index 0) rather than the XLEN-th variable.
pub enum MsbPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for MsbPrefix<XLEN> {
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
            0 => F::from_u32(c),
            1 => r_x.unwrap().into(),
            _ => checkpoints[Prefixes::Msb].unwrap(),
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
            1 => Some(r_x.into()).into(),
            _ => checkpoints[Prefixes::Msb].into(),
        }
    }
}
