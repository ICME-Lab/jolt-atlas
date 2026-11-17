//! The ring-switching prover.
#![allow(clippy::indexing_slicing)]

use super::{draw_batching, transpose, PACKING};
use crate::bitz::gf128::Gf128;
use crate::bitz::multilinear::eq_table;
use crate::bitz::pack::LOG_PACKING;
use atlas_transcript::Transcript;
use rayon::prelude::*;

const BYTES: usize = PACKING / 8;

/// For an `F_2`-linear map given by its images of the 128 basis elements, the image of every
/// value of each byte: `tables[k][b]` is the image of the element whose byte `k` is `b`.
fn byte_tables(images: &[Gf128]) -> Vec<[Gf128; 256]> {
    (0..BYTES)
        .map(|k| {
            let mut table = [Gf128::ZERO; 256];
            for b in 1usize..256 {
                let low = b.trailing_zeros() as usize;
                table[b] = table[b & (b - 1)] + images[8 * k + low];
            }
            table
        })
        .collect()
}

/// Reduces the claim `f(point) = a` on the bit polynomial packed as `packed`: the columns to
/// send, the weight `A` on the packed polynomial, and the claim `Σ_w A(w)·t'(w)`.
pub fn reduce<T: Transcript>(
    packed: &[Gf128],
    point: &[Gf128],
    target_bits: u32,
    transcript: &mut T,
) -> (Vec<Gf128>, Vec<Gf128>, Gf128) {
    let high = &point[LOG_PACKING..];
    assert_eq!(packed.len(), 1 << high.len(), "packed polynomial size");
    let eq_high = eq_table(high);

    // Column `v` is the sum of `eq(r_high, w)` over the elements `w` whose bit `v` is set.
    // Accumulate per byte value, then combine: sixteen additions per element.
    let sums = packed
        .par_iter()
        .zip(eq_high.par_iter())
        .fold(
            || vec![[Gf128::ZERO; 256]; BYTES],
            |mut acc, (element, &weight)| {
                for (k, table) in acc.iter_mut().enumerate() {
                    table[(element.0 >> (8 * k)) as u8 as usize] += weight;
                }
                acc
            },
        )
        .reduce(
            || vec![[Gf128::ZERO; 256]; BYTES],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    for (p, q) in x.iter_mut().zip(y) {
                        *p += *q;
                    }
                }
                a
            },
        );
    let columns: Vec<Gf128> = (0..PACKING)
        .map(|v| {
            let (k, bit) = (v / 8, v % 8);
            (0..256)
                .filter(|b| (b >> bit) & 1 == 1)
                .fold(Gf128::ZERO, |acc, b| acc + sums[k][b])
        })
        .collect();

    #[allow(clippy::expect_used)]
    let batching = draw_batching(&columns, target_bits, transcript)
        .expect("the prover's proof of work cannot fail");
    let reduced = batching
        .iter()
        .zip(&transpose(&columns))
        .fold(Gf128::ZERO, |acc, (&w, &row)| acc + w * row);

    // A(w) = T(eq(r_high, w)) for the linear map T sending β_u to eq(u, r'').
    let tables = byte_tables(&batching);
    let weights = eq_high
        .par_iter()
        .map(|value| {
            tables
                .iter()
                .enumerate()
                .fold(Gf128::ZERO, |acc, (k, table)| {
                    acc + table[(value.0 >> (8 * k)) as u8 as usize]
                })
        })
        .collect();
    (columns, weights, reduced)
}
