use crate::jolt::lookup_table::prefixes::Prefixes;
use jolt_core::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, SparseDensePrefix};

pub enum UnaryMsbPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F> for UnaryMsbPrefix<WORD_SIZE> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        _b: LookupBits,
        j: usize,
    ) -> F {
        match j {
            // sign bit is c
            j if j == WORD_SIZE => F::from_u32(c),
            // sign bit is r_x
            j if j == WORD_SIZE + 1 => r_x.unwrap(),
            // sign bit has been processed, use checkpoint
            _ => checkpoints[Prefixes::UnaryMsb].unwrap_or(F::one()),
        }
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        _r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        match j {
            j if j == WORD_SIZE + 1 => Some(r_x).into(),
            _ => checkpoints[Prefixes::UnaryMsb].into(),
        }
    }
}
