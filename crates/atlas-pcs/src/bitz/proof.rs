//! The BitZ commitment and opening proof, and what prover and verifier compute alike.
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::unimplemented,
    clippy::todo,
    clippy::indexing_slicing
)]

use super::gf128::Gf128;
use super::integers::Wide;
use super::merkle::Digest;
use super::params::Params;
use super::{gkr, ligerito};
use atlas_algebra::field::fp128::Fp128;
use atlas_algebra::serialize::{Canonical, Error};
use atlas_transcript::Transcript;
use common::impl_canonical_struct;
use std::io::{Read, Write};

/// A commitment to a bit vector: the Merkle root of its packed, encoded form.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Commitment {
    pub root: Digest,
}

impl_canonical_struct!([] Commitment { root });

impl Canonical for Wide {
    fn encode<W: Write>(&self, writer: W) -> Result<(), Error> {
        self.0.encode(writer)
    }

    fn encoded_len(&self) -> usize {
        self.0.encoded_len()
    }

    fn decode<R: Read>(reader: R) -> Result<Self, Error> {
        Ok(Self(<[u64; 5]>::decode(reader)?))
    }
}

/// A proof that `Σ_{row, column} eq(row point, row)·weight(column)·bit(row, column)` is a
/// claimed value in Fp128, bit `(row, column)` being bit `row + 2^log_rows · column`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    /// The claim over the integers: the inner product of the bits with the tensor of the
    /// canonical representatives of the row `eq` table and of the column weights.
    pub lifted_claim: Wide,
    /// The proof-of-work nonce before the projection prime is drawn.
    pub projection_nonce: u64,
    /// For each column, the inner product over the integers of its bits with the row table
    /// reduced modulo the projection prime.
    pub column_sums: Vec<u128>,
    pub grand_product: gkr::Proof,
    /// The ring-switching columns of the claim the grand product leaves on the bits.
    pub ring_columns: Vec<Gf128>,
    pub opening: ligerito::Proof,
}

impl_canonical_struct!([] Proof {
    lifted_claim,
    projection_nonce,
    column_sums,
    grand_product,
    ring_columns,
    opening
});

/// `eq(point, b)` over Fp128 for every `b` in the cube, variable `j` being bit `j` of `b`.
pub fn eq_table(point: &[Fp128]) -> Vec<Fp128> {
    let mut table = Vec::with_capacity(1 << point.len());
    table.push(Fp128::from(1u128));
    for &coordinate in point {
        let len = table.len();
        for i in 0..len {
            if let Some(&low) = table.get(i) {
                let high = low * coordinate;
                if let Some(slot) = table.get_mut(i) {
                    *slot = low - high;
                }
                table.push(high);
            }
        }
    }
    table
}

/// Binds the statement: the commitment, the shape, the weights and the claimed value.
pub(super) fn bind_statement<T: Transcript>(
    commitment: &Commitment,
    row_point: &[Fp128],
    column_weights: &[Fp128],
    claim: &Fp128,
    params: &Params,
    transcript: &mut T,
) {
    transcript.append_message(b"bitz-opening");
    transcript.append_bytes(&commitment.root);
    transcript.append_u64(params.log_rows as u64);
    transcript.append_u64(params.log_columns as u64);
    transcript.append_u64(params.target_bits as u64);
    transcript.append_scalars::<Fp128>(row_point);
    transcript.append_scalars::<Fp128>(column_weights);
    transcript.append_scalar(claim);
}
