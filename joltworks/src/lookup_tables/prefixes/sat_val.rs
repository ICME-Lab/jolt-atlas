use super::{PrefixCheckpoint, PrefixCheckpoints, Prefixes, SparseDensePrefix};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

/// Prefix component that computes the saturated boundary value `m*MIN + (1-m)*MAX`,
/// where `m` is the global sign bit (r[0]). Selects `MIN` for negative overflows and
/// `MAX` for positive overflows. Used in `SatClampTable` decomposition.
pub enum SatValPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for SatValPrefix<XLEN> {
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
            // r[0] is the current variable: evaluate at c
            0 => sat_val::<F, XLEN>(F::from_u32(c)),
            // r[0] has just been bound to r_x from the previous round
            1 => sat_val::<F, XLEN>(r_x.unwrap().into()),
            // r[0] fully processed: use checkpoint
            _ => checkpoints[Prefixes::SatVal].unwrap(),
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
            1 => Some(sat_val::<F, XLEN>(r_x.into())).into(),
            _ => checkpoints[Prefixes::SatVal].into(),
        }
    }
}

fn sat_val<F: JoltField, const XLEN: usize>(sign_eval: F) -> F {
    // NOTE: XLEN=32 (clamp i32→i16) is listed here because SatValPrefix is part of the global
    // Prefixes enum and update_checkpoints calls it for every table regardless of XLEN.
    // SatClampTable<32> itself is not fully implemented: materialize_entry and evaluate_mle
    // both panic for XLEN=32. All three sites must be completed together.
    let (min, max) = match XLEN {
        16 => (i8::MIN as i32, i8::MAX as i32),
        32 => (i16::MIN as i32, i16::MAX as i32),
        64 => (i32::MIN, i32::MAX),
        _ => unimplemented!(),
    };
    sign_eval * F::from_i32(min) + (F::one() - sign_eval) * F::from_i32(max)
}
