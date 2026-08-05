use super::{PrefixCheckpoint, SparseDensePrefix};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

/// The constant 0. Used to structurally disable a branch of a generic decomposition (e.g. the
/// "negative, in-range" branch of a floor-at-0 clamp table) regardless of round/challenge/bit
/// values, not just numerically at some points.
pub enum ZeroPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for ZeroPrefix<XLEN> {
    fn prefix_mle<C>(
        _checkpoints: &super::PrefixCheckpoints<F>,
        _r_x: Option<C>,
        _c: u32,
        _b: LookupBits,
        _j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        F::zero()
    }

    fn update_prefix_checkpoint<C>(
        _checkpoints: &super::PrefixCheckpoints<F>,
        _r_x: C,
        _r_y: C,
        _j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        Some(F::zero()).into()
    }
}
