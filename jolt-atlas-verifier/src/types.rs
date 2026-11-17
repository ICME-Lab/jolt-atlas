//! Proof identifiers, proof classification, and the opening-claim wrapper.

use atlas_algebra::field::JoltField;
use atlas_algebra::serialize::{Canonical, Error};
use atlas_sumcheck::opening_proof::Openings;
use std::io::{Read, Write};

// ---------------------------------------------------------------------------
// ProofId / ProofType
// ---------------------------------------------------------------------------

/// Unique identifier for a sumcheck proof instance.
///
/// Combines the node index with the proof type to uniquely identify each proof.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProofId(pub usize, pub ProofType);

/// The node index (as a `u64`), then the proof type's tag.
impl Canonical for ProofId {
    fn encode<W: Write>(&self, mut writer: W) -> Result<(), Error> {
        self.0.encode(&mut writer)?;
        self.1.encode(writer)
    }

    fn encoded_len(&self) -> usize {
        self.0.encoded_len() + self.1.encoded_len()
    }

    fn decode<R: Read>(mut reader: R) -> Result<Self, Error> {
        Ok(Self(
            usize::decode(&mut reader)?,
            ProofType::decode(reader)?,
        ))
    }
}

/// Type of sumcheck proof for different operations in the neural network.
///
/// The `u8` discriminants are part of the serialization format: a variant keeps its number, and
/// the numbers of deleted variants (1 to 8, 11, 13 to 15, 17 to 26) stay
/// free; a proof carrying one does not deserialize. Every variant is emitted by the prover for
/// some checked-in model (`proof_types_all_exercised`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProofType {
    /// Execution sumcheck for basic operations.
    Execution = 0,
    /// Axis-reduction sumcheck for `Sum` (proves the output equals the sum over the
    /// reduced axis; the output's range claim is in the packed range buckets, see
    /// `range_bucket`).
    SumReduction = 9,
    /// Einsum contraction (matmul) sumcheck proving
    /// `Σ_k L·R = out·2^S + R` (the range claims of `out` and `R` are in the packed
    /// range buckets; analogous to `SumReduction`).
    EinsumMatmul = 10,
    /// Arithmetic sumcheck for a fused `Mul`/`Square`/`Cube` or a `MeanOfSquares`,
    /// proving `acc(r) = out·D + R` for the rescale divisor `D` (the range claims of `out`
    /// and `R` are in the packed range buckets; the einsum analogue is `EinsumMatmul`).
    RescaleArith = 12,
    /// All range buckets' value sumchecks (opening each bucket's packed value), batched.
    DeferredRemainderRC = 16,
    /// All range buckets' digit range checks, batched.
    DeferredRemainderDigitRange = 27,
    /// Every deferred table read (`DeferredBatch::Lookups`), as one batched sumcheck.
    DeferredLookups = 28,
    /// Every one-hot check of a read address (`DeferredBatch::OneHot`), as one batched
    /// sumcheck; absent when the model has no lookup.
    DeferredOneHot = 29,
    /// A softmax node's reduction of its output claim and its two row sums to its table reads.
    SoftmaxSums = 30,
}

impl ProofType {
    /// Every proof type, in tag order.
    pub fn all() -> impl Iterator<Item = Self> {
        (0..=u8::MAX).filter_map(|tag| Self::try_from(tag).ok())
    }
}

impl TryFrom<u8> for ProofType {
    type Error = Error;

    fn try_from(tag: u8) -> Result<Self, Self::Error> {
        match tag {
            0 => Ok(Self::Execution),
            9 => Ok(Self::SumReduction),
            10 => Ok(Self::EinsumMatmul),
            12 => Ok(Self::RescaleArith),
            16 => Ok(Self::DeferredRemainderRC),
            27 => Ok(Self::DeferredRemainderDigitRange),
            28 => Ok(Self::DeferredLookups),
            29 => Ok(Self::DeferredOneHot),
            30 => Ok(Self::SoftmaxSums),
            _ => Err(Error::InvalidData),
        }
    }
}

impl Canonical for ProofType {
    fn encode<W: Write>(&self, writer: W) -> Result<(), Error> {
        (*self as u8).encode(writer)
    }

    fn encoded_len(&self) -> usize {
        0u8.encoded_len()
    }

    fn decode<R: Read>(reader: R) -> Result<Self, Error> {
        let tag = u8::decode(reader)?;
        ProofType::try_from(tag)
    }
}

// ---------------------------------------------------------------------------
// Claims
// ---------------------------------------------------------------------------

/// Wrapper for polynomial opening claims.
#[derive(Debug, Clone)]
pub struct Claims<F: JoltField>(Openings<F>);

impl<F: JoltField> Claims<F> {
    /// Wrap the prover's final openings.
    pub fn from_parts(openings: Openings<F>) -> Self {
        Self(openings)
    }

    /// The claims, keyed by opening id.
    pub fn openings(&self) -> &Openings<F> {
        &self.0
    }

    /// Mutable claims, for the soundness harness (`test-util`).
    #[cfg(feature = "test-util")]
    pub fn openings_mut(&mut self) -> &mut Openings<F> {
        &mut self.0
    }
}
