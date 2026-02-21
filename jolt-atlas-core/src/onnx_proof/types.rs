//! Internal types used throughout the ONNX proof system.
//!
//! Contains proof identifiers, proof classification types, claim wrappers,
//! and debug structures. Re‑exported from the parent [`super`] module so
//! external code can continue to use `crate::onnx_proof::{ProofId, …}`.

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
use joltworks::{
    field::JoltField,
    poly::opening_proof::{Openings, ProverOpeningAccumulator},
    transcripts::Transcript,
};

// ---------------------------------------------------------------------------
// ProofId / ProofType
// ---------------------------------------------------------------------------

/// Unique identifier for a sumcheck proof instance.
///
/// Combines the node index with the proof type to uniquely identify each proof.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, CanonicalSerialize, CanonicalDeserialize,
)]
pub struct ProofId(pub usize, pub ProofType);

/// Type of sumcheck proof for different operations in the neural network.
///
/// The `u8` discriminants are part of the serialization format — do not reorder or remove variants
/// without a migration, as it would break proof deserialization.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProofType {
    /// Execution sumcheck for basic operations.
    Execution = 0,
    /// Neural teleportation for tanh approximation.
    NeuralTeleport = 1,
    /// Read-address one-hot encoding checks.
    RaOneHotChecks = 2,
    /// Hamming weight check for read-addresses.
    RaHammingWeight = 3,
    /// Softmax division by sum of max.
    SoftmaxDivSumMax = 4,
    /// Softmax exponentiation read-raf checking.
    SoftmaxExponentiationReadRaf = 5,
    /// Softmax exponentiation read-address one-hot encoding checks.
    SoftmaxExponentiationRaOneHot = 6,
    /// Range-checking for remainders.
    RangeCheck = 7,
}

impl TryFrom<u8> for ProofType {
    type Error = SerializationError;

    fn try_from(tag: u8) -> Result<Self, Self::Error> {
        match tag {
            0 => Ok(Self::Execution),
            1 => Ok(Self::NeuralTeleport),
            2 => Ok(Self::RaOneHotChecks),
            3 => Ok(Self::RaHammingWeight),
            4 => Ok(Self::SoftmaxDivSumMax),
            5 => Ok(Self::SoftmaxExponentiationReadRaf),
            6 => Ok(Self::SoftmaxExponentiationRaOneHot),
            7 => Ok(Self::RangeCheck),
            _ => Err(SerializationError::InvalidData),
        }
    }
}

impl CanonicalSerialize for ProofType {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (*self as u8).serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        0u8.serialized_size(compress)
    }
}

impl Valid for ProofType {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for ProofType {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(reader, compress, validate)?;
        ProofType::try_from(tag)
    }
}

// ---------------------------------------------------------------------------
// Claims
// ---------------------------------------------------------------------------

/// Wrapper for polynomial opening claims.
#[derive(Debug, Clone)]
pub struct Claims<F: JoltField>(pub Openings<F>);

// ---------------------------------------------------------------------------
// ProverDebugInfo
// ---------------------------------------------------------------------------

/// Debug information from the prover for testing and verification.
///
/// Contains the transcript and opening accumulator state for comparing
/// prover and verifier execution in tests.
#[allow(dead_code)]
pub struct ProverDebugInfo<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    pub(crate) transcript: ProofTranscript,
    pub(crate) opening_accumulator: ProverOpeningAccumulator<F>,
}
