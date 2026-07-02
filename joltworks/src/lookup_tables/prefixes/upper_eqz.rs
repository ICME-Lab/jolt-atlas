use super::{PrefixCheckpoint, PrefixCheckpoints, Prefixes, SparseDensePrefix};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

/// Prefix component that accumulates whether the upper half-word bits (positions 0..31)
/// are all zero (equal-to-zero / eqz check). Used in `SatClampTable` decomposition.
pub enum UpperEqzPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for UpperEqzPrefix<XLEN> {
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
        let mut result = checkpoints[Prefixes::UpperEqz].unwrap_or(F::one());
        if j >= 32 {
            return result;
        }
        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            result *= (F::one() - r_x) * (F::one() - y);
        } else {
            let x = F::from_u8(c as u8);
            let y = F::from_u8(b.pop_msb());
            result *= (F::one() - x) * (F::one() - y);
        }
        result * F::from_bool(b.eqz())
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
        if j >= 32 {
            return checkpoints[Prefixes::UpperEqz].into();
        }

        let updated = checkpoints[Prefixes::UpperEqz].unwrap_or(F::one())
            * (F::one() - r_x)
            * (F::one() - r_y);
        Some(updated).into()
    }
}
