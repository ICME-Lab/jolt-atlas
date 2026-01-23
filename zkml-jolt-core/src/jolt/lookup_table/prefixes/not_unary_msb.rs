use crate::jolt::lookup_table::prefixes::Prefixes;

use jolt_core::field::JoltField;
use jolt_core::utils::lookup_bits::LookupBits;

use super::{PrefixCheckpoint, SparseDensePrefix};

pub enum NotUnaryMsbPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F> for NotUnaryMsbPrefix<WORD_SIZE> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F::Challenge>,
        c: u32,
        _b: LookupBits,
        j: usize,
    ) -> F {
        match j {
            // sign bit is c
            j if j == WORD_SIZE => F::one() - F::from_u32(c),
            // sign bit is r_x
            j if j == WORD_SIZE + 1 => F::one() - F::from(r_x.unwrap().into()),
            // sign bit has been processed, use checkpoint
            _ => checkpoints[Prefixes::NotUnaryMsb].unwrap_or(F::one()),
        }
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F::Challenge,
        _r_y: F::Challenge,
        j: usize,
        _phase_length: usize,
    ) -> PrefixCheckpoint<F> {
        let r_x_f: F = r_x.into();
        match j {
            j if j == WORD_SIZE + 1 => Some(F::one() - r_x_f).into(),
            _ => checkpoints[Prefixes::NotUnaryMsb].into(),
        }
    }
}
