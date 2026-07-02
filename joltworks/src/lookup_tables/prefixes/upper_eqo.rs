use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    lookup_tables::prefixes::PrefixCheckpoints,
    utils::lookup_bits::LookupBits,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// Prefix component that accumulates whether the upper half-word bits (positions 0..31)
/// are all one (equal-to-ones / eqo check). Used in `SatClampTable` decomposition.
pub enum UpperEqoPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for UpperEqoPrefix<XLEN> {
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
        let mut result = checkpoints[Prefixes::UpperEqo].unwrap_or(F::one());
        if j >= 32 {
            return result;
        }

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            result *= r_x * y;
        } else {
            let x = F::from_u8(c as u8);
            let y_msb = F::from_u8(b.pop_msb());
            result *= x * y_msb;
        }

        let eqo = b.eqo() as u64;
        result *= F::from_u64(eqo);
        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &PrefixCheckpoints<F>,
        r_x: C,
        r_y: C,
        j: usize,
        _: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let cp = checkpoints[Prefixes::UpperEqo].unwrap_or(F::one());

        if j >= 32 {
            return Some(cp).into();
        }

        let updated = cp * (r_x * r_y);
        Some(updated).into()
    }
}
