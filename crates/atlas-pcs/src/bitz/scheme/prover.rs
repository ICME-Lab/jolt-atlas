//! The prover's side of the BitZ commitment scheme: bit planes from polynomials, the
//! commitment, and the joint opening.
#![allow(clippy::indexing_slicing, clippy::expect_used, clippy::panic)]

use super::{
    opening_seed, opening_weights, BitzBatchCommitment, BitzOpening, BitzSlot, Layout, Shape,
    OPENING_LABEL,
};
use crate::bitz::proof::Commitment;
use crate::bitz::prover::{self as bitz, Committed};
use crate::bitz::Params;
use crate::commitment_scheme::BatchCommitOutput;
use atlas_algebra::field::fp128::Fp128;
use atlas_algebra::poly::multilinear::MultilinearPolynomial;
use atlas_transcript::ProvingTranscript;
use rayon::prelude::*;
use std::sync::Arc;

/// What the prover keeps of a batch commitment.
struct Batch {
    commitment: Commitment,
    committed: Committed,
    layout: Layout,
    params: Params,
    shapes: Vec<Shape>,
}

/// A polynomial's handle on its batch.
#[derive(Clone)]
pub struct BitzHint {
    batch: Arc<Batch>,
    slot: u32,
}

impl std::fmt::Debug for BitzHint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitzHint")
            .field("slot", &self.slot)
            .finish()
    }
}

impl PartialEq for BitzHint {
    fn eq(&self, other: &Self) -> bool {
        self.slot == other.slot && Arc::ptr_eq(&self.batch, &other.batch)
    }
}

/// The coefficients of a polynomial as nonnegative integers below `2^64`, the signed kinds
/// shifted by half their range.
enum Coefficients<'a> {
    Bool(&'a [bool]),
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    U64(&'a [u64]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    /// One nonzero coefficient per cycle `t`, at `address · cycles + t`.
    OneHot {
        addresses: usize,
        indices: &'a [Option<u16>],
    },
}

impl<'a> Coefficients<'a> {
    fn of(poly: &'a MultilinearPolynomial<Fp128>) -> Self {
        match poly {
            MultilinearPolynomial::BoolScalars(p) => Self::Bool(&p.coeffs),
            MultilinearPolynomial::U8Scalars(p) => Self::U8(&p.coeffs),
            MultilinearPolynomial::U16Scalars(p) => Self::U16(&p.coeffs),
            MultilinearPolynomial::U32Scalars(p) => Self::U32(&p.coeffs),
            MultilinearPolynomial::U64Scalars(p) => Self::U64(&p.coeffs),
            MultilinearPolynomial::I32Scalars(p) => Self::I32(&p.coeffs),
            MultilinearPolynomial::I64Scalars(p) => Self::I64(&p.coeffs),
            MultilinearPolynomial::OneHot(p) => Self::OneHot {
                addresses: p.K,
                indices: &p.nonzero_indices,
            },
            MultilinearPolynomial::LargeScalars(_)
            | MultilinearPolynomial::U128Scalars(_)
            | MultilinearPolynomial::I128Scalars(_)
            | MultilinearPolynomial::S128Scalars(_) => {
                panic!("BitZ commits coefficients of at most 64 bits")
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Bool(c) => c.len(),
            Self::U8(c) => c.len(),
            Self::U16(c) => c.len(),
            Self::U32(c) => c.len(),
            Self::U64(c) => c.len(),
            Self::I32(c) => c.len(),
            Self::I64(c) => c.len(),
            Self::OneHot { addresses, indices } => addresses * indices.len(),
        }
    }

    /// The bits a coefficient needs, and whether the kind is shifted.
    fn bits(&self) -> (u8, bool) {
        fn width<T: Copy + Sync>(values: &[T], to_u64: impl Fn(T) -> u64 + Sync) -> u8 {
            let max = values.par_iter().map(|&v| to_u64(v)).max().unwrap_or(0);
            (u64::BITS - max.leading_zeros()).max(1) as u8
        }
        match self {
            Self::Bool(_) | Self::OneHot { .. } => (1, false),
            Self::U8(c) => (width(c, u64::from), false),
            Self::U16(c) => (width(c, u64::from), false),
            Self::U32(c) => (width(c, u64::from), false),
            Self::U64(c) => (width(c, |v| v), false),
            Self::I32(_) => (32, true),
            Self::I64(_) => (64, true),
        }
    }

    /// Writes plane `bit` into `words`, which start zeroed: bit `j` of the plane at position
    /// `j mod 64` of word `j / 64`.
    fn write_plane(&self, bit: usize, words: &mut [u64]) {
        fn pack<T: Copy>(values: &[T], bit: usize, words: &mut [u64], to_u64: impl Fn(T) -> u64) {
            for (word, chunk) in words.iter_mut().zip(values.chunks(64)) {
                *word = chunk
                    .iter()
                    .enumerate()
                    .fold(0, |acc, (i, &v)| acc | ((to_u64(v) >> bit) & 1) << i);
            }
        }
        match self {
            Self::Bool(c) => pack(c, bit, words, u64::from),
            Self::U8(c) => pack(c, bit, words, u64::from),
            Self::U16(c) => pack(c, bit, words, u64::from),
            Self::U32(c) => pack(c, bit, words, u64::from),
            Self::U64(c) => pack(c, bit, words, |v| v),
            Self::I32(c) => pack(c, bit, words, |v| (v as i64 + (1 << 31)) as u64),
            Self::I64(c) => pack(c, bit, words, |v| (v as i128 + (1 << 63)) as u64),
            Self::OneHot { indices, .. } => {
                let cycles = indices.len();
                for (cycle, index) in indices.iter().enumerate() {
                    if let Some(address) = index {
                        let position = *address as usize * cycles + cycle;
                        words[position / 64] |= 1 << (position % 64);
                    }
                }
            }
        }
    }
}

/// Commits to every polynomial of a proof.
pub(super) fn commit<U>(polys: &[U]) -> BatchCommitOutput<super::BitzScheme>
where
    U: std::borrow::Borrow<MultilinearPolynomial<Fp128>> + Sync,
{
    if polys.is_empty() {
        return (BitzBatchCommitment::default(), Vec::new());
    }
    let coefficients: Vec<Coefficients<'_>> = polys
        .iter()
        .map(|poly| Coefficients::of(poly.borrow()))
        .collect();
    let shapes: Vec<Shape> = coefficients
        .par_iter()
        .map(|c| {
            let len = c.len();
            assert!(
                len.is_power_of_two(),
                "a polynomial has a power of two coefficients"
            );
            let (bits, signed) = c.bits();
            Shape {
                num_vars: len.trailing_zeros() as u8,
                bits,
                signed,
            }
        })
        .collect();
    let layout = Layout::new(&shapes).expect("the batch fits a BitZ commitment");
    let params = layout.params().expect("the layout has parameters");

    // The bit matrix, column by column: each plane's words are a run of whole columns, and
    // the runs are disjoint.
    let words_per_column = (1usize << layout.log_rows) / 64;
    let mut words = vec![0u64; words_per_column << layout.log_columns];
    let mut planes: Vec<(usize, usize, usize, usize)> = layout
        .planes
        .iter()
        .enumerate()
        .flat_map(|(slot, planes)| {
            planes
                .iter()
                .enumerate()
                .map(move |(bit, p)| (p.first_column, p.log_columns, slot, bit))
        })
        .collect();
    planes.sort();
    let mut runs = Vec::with_capacity(planes.len());
    let mut rest = words.as_mut_slice();
    let mut taken = 0;
    for &(first_column, log_columns, slot, bit) in &planes {
        let start = first_column * words_per_column - taken;
        let len = words_per_column << log_columns;
        let (run, after) = rest[start..].split_at_mut(len);
        runs.push((run, slot, bit));
        rest = after;
        taken = first_column * words_per_column + len;
    }
    runs.into_par_iter()
        .for_each(|(run, slot, bit)| coefficients[slot].write_plane(bit, run));

    let (commitment, committed) = bitz::commit(words, &params);
    let batch = Arc::new(Batch {
        commitment,
        committed,
        layout,
        params,
        shapes: shapes.clone(),
    });
    let per_polynomial = (0..polys.len() as u32)
        .map(|slot| {
            (
                BitzSlot(slot),
                BitzHint {
                    batch: batch.clone(),
                    slot,
                },
            )
        })
        .collect();
    (
        BitzBatchCommitment {
            root: commitment.root,
            shapes,
        },
        per_polynomial,
    )
}

/// Opens `Σ_i coeffs_i · f_i` at `point` for the polynomials of `hints`, which are of one
/// batch.
pub(super) fn open<T: ProvingTranscript>(
    coeffs: &[Fp128],
    hints: &[BitzHint],
    point: &[Fp128],
    transcript: &mut T,
) -> BitzOpening {
    let batch = &hints.first().expect("an opening opens a polynomial").batch;
    assert!(
        hints.iter().all(|hint| Arc::ptr_eq(&hint.batch, batch)),
        "the opened polynomials are of one batch"
    );
    let slots: Vec<u32> = hints.iter().map(|hint| hint.slot).collect();
    let weights = opening_weights(&batch.layout, &batch.shapes, &slots, coeffs, point)
        .expect("the opened polynomials are the batch's, at a point that covers them");

    let seed = opening_seed(transcript);
    let mut inner = T::new(OPENING_LABEL);
    inner.append_bytes(&seed);
    let (_, proof) = bitz::prove(
        &batch.commitment,
        &batch.committed,
        &weights.row_point,
        &weights.columns,
        &batch.params,
        &mut inner,
    );
    BitzOpening {
        proof,
        narg: inner.into_narg(),
    }
}
