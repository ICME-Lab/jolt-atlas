use super::{PrefixCheckpoint, PrefixCheckpoints, Prefixes, SparseDensePrefix};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

/// Prefix component for `1 - sign_bit` of the lower 32-bit word (complement of
/// `LowerMsbPrefix`). Used in `SatClampTable` decomposition.
pub enum NotLowerMsbPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for NotLowerMsbPrefix<XLEN> {
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
            // suffix will handle; return 1 to avoid affecting the product
            j if j < 32 => F::one(),
            // sign bit in c
            32 => F::one() - F::from_u32(c),
            // sign bit in r_x
            33 => F::one() - r_x.unwrap(),
            // sign bit processed; use checkpoint.
            _ => checkpoints[Prefixes::NotLowerMsb].unwrap(),
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
            33 => Some(F::one() - r_x).into(),
            _ => checkpoints[Prefixes::NotLowerMsb].into(),
        }
    }
}
