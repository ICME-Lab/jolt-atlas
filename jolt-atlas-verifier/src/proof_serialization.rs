//! Serialization and deserialization for [`ONNXProof`] and related types.
//!
//! Uses [`Canonical`], the canonical encoding the proof format is defined by.

use atlas_algebra::serialize::{encode_slice, slice_encoded_len, Canonical, Error};
use std::io::{Read, Write};

use atlas_algebra::field::JoltField;
use atlas_pcs::commitment_scheme::CommitmentScheme;

use crate::{ONNXProof, ReducedOpeningProof};

// ---------------------------------------------------------------------------
// ReducedOpeningProof<F, PCS>
// ---------------------------------------------------------------------------

impl<F, PCS> Canonical for ReducedOpeningProof<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn encode<W: Write>(&self, writer: W) -> Result<(), Error> {
        self.joint_opening_proof().encode(writer)
    }

    fn encoded_len(&self) -> usize {
        self.joint_opening_proof().encoded_len()
    }

    fn decode<R: Read>(reader: R) -> Result<Self, Error> {
        Ok(Self::from_parts(PCS::Proof::decode(reader)?))
    }
}

// ---------------------------------------------------------------------------
// ONNXProof<F, PCS>
// ---------------------------------------------------------------------------

impl<F, PCS> Canonical for ONNXProof<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn encode<W: Write>(&self, mut writer: W) -> Result<(), Error> {
        // 1. Commitments: the batch commitment, then one per polynomial
        self.batch_commitment().encode(&mut writer)?;
        encode_slice(self.commitments(), &mut writer)?;

        // 2. Reduced opening proof (an `Option`: a `bool` tag, then the value)
        let reduced = self.reduced_opening_proof();
        reduced.is_some().encode(&mut writer)?;
        if let Some(r) = reduced {
            r.encode(&mut writer)?;
        }

        // 3. Transcript NARG string
        encode_slice(self.narg(), &mut writer)?;

        Ok(())
    }

    fn encoded_len(&self) -> usize {
        self.batch_commitment().encoded_len()
            + slice_encoded_len(self.commitments())
            + true.encoded_len()
            + self.reduced_opening_proof().map_or(0, |r| r.encoded_len())
            + slice_encoded_len(self.narg())
    }

    fn decode<R: Read>(mut reader: R) -> Result<Self, Error> {
        let batch_commitment = PCS::BatchCommitment::decode(&mut reader)?;
        let commitments = Vec::decode(&mut reader)?;
        let reduced_opening_proof = Option::decode(&mut reader)?;
        let narg = Vec::<u8>::decode(&mut reader)?;

        Ok(Self::from_parts(
            batch_commitment,
            commitments,
            reduced_opening_proof,
            narg,
        ))
    }
}

// ---------------------------------------------------------------------------
// Convenience helpers
// ---------------------------------------------------------------------------

/// Serialize a proof to bytes.
pub fn serialize_proof<F, PCS>(proof: &ONNXProof<F, PCS>) -> Result<Vec<u8>, Error>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut buf = Vec::with_capacity(proof.encoded_len());
    proof.encode(&mut buf)?;
    Ok(buf)
}

/// Deserialize a proof from bytes. The bytes must hold exactly one proof: trailing
/// bytes are rejected, so a proof has a single byte encoding.
pub fn deserialize_proof<F, PCS>(bytes: &[u8]) -> Result<ONNXProof<F, PCS>, Error>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut reader = bytes;
    let proof = ONNXProof::decode(&mut reader)?;
    if !reader.is_empty() {
        return Err(Error::InvalidData);
    }
    Ok(proof)
}
