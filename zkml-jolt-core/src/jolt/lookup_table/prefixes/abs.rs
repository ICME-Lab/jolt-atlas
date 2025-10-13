use jolt_core::{field::JoltField, utils::lookup_bits::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// This prefix either returns x if x > 0, or !x if x < 0.
/// A +1 is needed to recover abs(x) when x < 0, this is handled by a suffix.
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

        // TODO(AntoineF4C5): Should be able to get rid of UnaryMsb and NOTLowerNoMsb prefixes by refactoring
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
                let updated = checkpoints[Prefixes::Abs].unwrap_or(F::zero())
                    + F::from_u64(1 << x_shift)
                        * ((F::one() - sign_bit) * r_x + sign_bit * (F::one() - r_x)) // if positive then r_x, else !r_x
                    + F::from_u64(1 << y_shift)
                        * ((F::one() - sign_bit) * r_y + sign_bit * (F::one() - r_y)); // if positive then r_y, else !r_y
                Some(updated).into()
            }
        }
    }
}
