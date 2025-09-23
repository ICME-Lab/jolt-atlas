use jolt_core::field::JoltField;
use jolt_core::utils::lookup_bits::LookupBits;

use crate::jolt::executor::read_raf_checking::current_suffix_len;

use super::{AtlasPrefixCheckpoint, AtlasSparseDensePrefix, Prefixes};

pub enum LowerWordNoMsbPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> AtlasSparseDensePrefix<F>
    for LowerWordNoMsbPrefix<WORD_SIZE>
{
    fn prefix_mle(
        checkpoints: &[AtlasPrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        // Ignore high-order variables
        if j < WORD_SIZE {
            return F::zero();
        }

        let two = 2 * WORD_SIZE;
        let c_f = F::from_u8(c as u8);
        let mut result = checkpoints[Prefixes::LowerWordNoMsb].unwrap_or(F::zero());
        let mut add = |shift: usize, val: F| {
            result += F::from_u64(1u64 << shift) * val;
        };
        match (r_x, j) {
            // MSB is in c
            (None, jj) if jj == WORD_SIZE => {
                let y_msb = b.pop_msb();
                let y_shift = two - j - 2;
                add(y_shift, F::from_u8(y_msb));
            }

            // MSB is in r_x
            (Some(_), jj) if jj == WORD_SIZE + 1 => {
                let y_shift = two - j - 1;
                add(y_shift, c_f);
            }

            // General cases
            (None, _) => {
                let y_msb = b.pop_msb();
                let x_shift = two - j - 1;
                let y_shift = x_shift - 1;
                add(x_shift, c_f);
                add(y_shift, F::from_u8(y_msb));
            }
            (Some(rx), _) => {
                let x_shift = two - j;
                let y_shift = x_shift - 1;
                add(x_shift, rx);
                add(y_shift, c_f);
            }
        }
        let suffix_len = current_suffix_len(two, j);
        result += F::from_u64(u64::from(b) << suffix_len);
        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[AtlasPrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> AtlasPrefixCheckpoint<F> {
        let two = 2 * WORD_SIZE;
        match j {
            j if j < WORD_SIZE => None.into(),
            j if j == WORD_SIZE + 1 => {
                let y_shift = two - j - 1;
                let updated = checkpoints[Prefixes::LowerWordNoMsb].unwrap_or(F::zero())
                    + F::from_u64(1 << y_shift) * r_y;
                Some(updated).into()
            }
            _ => {
                let x_shift = two - j;
                let y_shift = x_shift - 1;
                let updated = checkpoints[Prefixes::LowerWordNoMsb].unwrap_or(F::zero())
                    + F::from_u64(1 << x_shift) * r_x
                    + F::from_u64(1 << y_shift) * r_y;
                Some(updated).into()
            }
        }
    }
}
