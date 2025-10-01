use jolt_core::field::JoltField;
use jolt_core::utils::lookup_bits::LookupBits;

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum ReluPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F> for ReluPrefix<WORD_SIZE> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        // Ignore high-order variables
        if j < WORD_SIZE {
            return F::zero();
        }
        let nsign_bit =
            *Prefixes::NotUnaryMsb.prefix_mle::<WORD_SIZE, F>(checkpoints, r_x, c, b, j);
        let word = *Prefixes::LowerWordNoMsb.prefix_mle::<WORD_SIZE, F>(checkpoints, r_x, c, b, j);
        nsign_bit * word
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let two = 2 * WORD_SIZE;
        match j {
            // suffix handles relu
            j if j < WORD_SIZE => None.into(),
            j if j == WORD_SIZE + 1 => {
                // Sign bit is in r_x
                let sign_bit = r_x;
                let y_shift = two - j - 1;
                let updated = checkpoints[Prefixes::Relu].unwrap_or(F::zero())
                    + F::from_u64(1 << y_shift) * r_y * (F::one() - sign_bit);
                Some(updated).into()
            }
            _ => {
                let x_shift = two - j;
                let y_shift = x_shift - 1;
                let updated = checkpoints[Prefixes::Relu].unwrap_or(F::zero())
                    + F::from_u64(1 << x_shift) * r_x * checkpoints[Prefixes::NotUnaryMsb].unwrap()
                    + F::from_u64(1 << y_shift) * r_y * checkpoints[Prefixes::NotUnaryMsb].unwrap();
                Some(updated).into()
            }
        }
    }
}
