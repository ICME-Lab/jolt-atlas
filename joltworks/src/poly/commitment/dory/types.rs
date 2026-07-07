//! Wrapper types bridging `dory-pcs` into joltworks' [`CommitmentScheme`].
//!
//! [`CommitmentScheme`]: super::super::commitment_scheme::CommitmentScheme
//!
//! dory-pcs exposes its own `DorySerialize`/`DoryDeserialize` traits and
//! `Compress`/`Validate` enums rather than arkworks' `CanonicalSerialize`.
//! The group and proof types (`ArkGT`/`ArkG1`/`ArkDoryProof`) already implement
//! arkworks' traits, so those wrappers delegate via `#[derive(...)]`; the setup
//! types only implement dory's traits, so those wrappers bridge through a small
//! enum mapping.

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
use dory::backends::arkworks::{ArkDoryProof, ArkFr, ArkG1, ArkGT, BN254};
use dory::primitives::arithmetic::Group;
use dory::primitives::serialization::{
    Compress as DoryCompress, DoryDeserialize, DorySerialize,
    SerializationError as DorySerializationError, Validate as DoryValidate,
};
use dory::{ProverSetup, VerifierSetup};

use crate::transcripts::{AppendToTranscript, Transcript};

// -- enum bridges between arkworks' and dory's serialization vocabularies -----

#[inline]
fn map_compress(c: Compress) -> DoryCompress {
    match c {
        Compress::Yes => DoryCompress::Yes,
        Compress::No => DoryCompress::No,
    }
}

#[inline]
fn map_validate(v: Validate) -> DoryValidate {
    match v {
        Validate::Yes => DoryValidate::Yes,
        Validate::No => DoryValidate::No,
    }
}

#[inline]
fn map_err(_e: DorySerializationError) -> SerializationError {
    SerializationError::InvalidData
}

// -- Commitment (a single GT element) ----------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryCommitment(pub ArkGT);

impl Default for DoryCommitment {
    #[inline]
    fn default() -> Self {
        Self(<ArkGT as Group>::identity())
    }
}

impl AppendToTranscript for DoryCommitment {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        let mut buf = Vec::new();
        CanonicalSerialize::serialize_compressed(&self.0, &mut buf)
            .expect("GT commitment serialization is infallible");
        transcript.append_bytes(&buf);
    }
}

// -- Evaluation proof --------------------------------------------------------

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryProof(pub ArkDoryProof);

// -- Opening hint: the row (tier-1) commitments + blind from `commit` --------

#[derive(Clone, Debug, PartialEq)]
pub struct DoryHint {
    pub(crate) row_commitments: Vec<ArkG1>,
    pub(crate) commit_blind: ArkFr,
}

// -- Setups ------------------------------------------------------------------

/// Prover setup. Carries the derived verifier setup alongside the prover SRS so
/// [`setup_verifier`] can hand it out without recomputing (or risking a
/// mismatched re-sample of) the reference string.
///
/// [`setup_verifier`]: super::super::commitment_scheme::CommitmentScheme::setup_verifier
#[derive(Clone, Debug)]
pub struct DoryProverSetup {
    pub(crate) prover: ProverSetup<BN254>,
    pub(crate) verifier: VerifierSetup<BN254>,
}

#[derive(Clone, Debug)]
pub struct DoryVerifierSetup(pub VerifierSetup<BN254>);

impl CanonicalSerialize for DoryVerifierSetup {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        DorySerialize::serialize_with_mode(&self.0, writer, map_compress(compress)).map_err(map_err)
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        DorySerialize::serialized_size(&self.0, map_compress(compress))
    }
}
impl Valid for DoryVerifierSetup {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}
impl CanonicalDeserialize for DoryVerifierSetup {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        DoryDeserialize::deserialize_with_mode(
            reader,
            map_compress(compress),
            map_validate(validate),
        )
        .map(Self)
        .map_err(map_err)
    }
}

impl CanonicalSerialize for DoryProverSetup {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        DorySerialize::serialize_with_mode(&self.prover, &mut writer, map_compress(compress))
            .map_err(map_err)?;
        DorySerialize::serialize_with_mode(&self.verifier, &mut writer, map_compress(compress))
            .map_err(map_err)
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        DorySerialize::serialized_size(&self.prover, map_compress(compress))
            + DorySerialize::serialized_size(&self.verifier, map_compress(compress))
    }
}
impl Valid for DoryProverSetup {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}
impl CanonicalDeserialize for DoryProverSetup {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let prover = DoryDeserialize::deserialize_with_mode(
            &mut reader,
            map_compress(compress),
            map_validate(validate),
        )
        .map_err(map_err)?;
        let verifier = DoryDeserialize::deserialize_with_mode(
            &mut reader,
            map_compress(compress),
            map_validate(validate),
        )
        .map_err(map_err)?;
        Ok(Self { prover, verifier })
    }
}
