use jolt_core::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum AbsPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F> for AbsPrefix<WORD_SIZE> {
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
        let sign_bit = *Prefixes::UnaryMsb.prefix_mle::<WORD_SIZE, F>(checkpoints, r_x, c, b, j);
        let word = *Prefixes::LowerWordNoMsb.prefix_mle::<WORD_SIZE, F>(checkpoints, r_x, c, b, j);
        let negated_word =
            *Prefixes::NOTLowerNoMsb.prefix_mle::<WORD_SIZE, F>(checkpoints, r_x, c, b, j);

        // (1 - sign_bit) * word + sign_bit * negated_word
        sign_bit * (negated_word - word) + word
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let two = 2 * WORD_SIZE;
        match j {
            // suffix handles abs
            j if j < WORD_SIZE => None.into(),
            j if j == WORD_SIZE + 1 => {
                // Sign bit is in r_x
                let sign_bit = r_x;
                let y_shift = two - j - 1;
                let updated = checkpoints[Prefixes::Abs].unwrap_or(F::zero())
                    + F::from_u64(1 << y_shift)
                        * ((F::one() - sign_bit) * r_y + sign_bit * (F::one() - r_y)); // if positive then r_y, else !r_y
                Some(updated).into()
            }
            _ => {
                let x_shift = two - j;
                let y_shift = x_shift - 1;
                let sign_bit = checkpoints[Prefixes::UnaryMsb].unwrap();
                let mut updated = checkpoints[Prefixes::Abs].unwrap_or(F::zero())
                    + F::from_u64(1 << x_shift)
                        * ((F::one() - sign_bit) * r_x + sign_bit * (F::one() - r_x)) // if positive then r_x, else !r_x
                    + F::from_u64(1 << y_shift)
                        * ((F::one() - sign_bit) * r_y + sign_bit * (F::one() - r_y)); // if positive then r_y, else !r_y
                if j == two - 1 {
                    // Add the +1 for the negative case (from -x = (!x) + 1)
                    updated += sign_bit;
                }
                Some(updated).into()
            }
        }
    }
}
