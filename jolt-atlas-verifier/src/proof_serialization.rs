//! Serialization and deserialization for [`ONNXProof`] and related types.
//!
//! Uses [`Canonical`], the canonical encoding the proof format is defined by.
//! BTreeMap fields are serialized as sorted `(key, value)` pairs prefixed by their length.

use std::collections::BTreeMap;

use atlas_algebra::serialize::{encode_slice, slice_encoded_len, Canonical, Error};
use std::io::{Read, Write};

use atlas_algebra::field::JoltField;
use atlas_pcs::commitment_scheme::CommitmentScheme;
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;

use crate::{ONNXProof, ReducedOpeningProof};

// ---------------------------------------------------------------------------
// BTreeMap helpers
// ---------------------------------------------------------------------------

fn serialize_btreemap<W, K, V>(map: &BTreeMap<K, V>, writer: &mut W) -> Result<(), Error>
where
    W: Write,
    K: Canonical,
    V: Canonical,
{
    (map.len() as u64).encode(&mut *writer)?;
    for (k, v) in map.iter() {
        k.encode(&mut *writer)?;
        v.encode(&mut *writer)?;
    }
    Ok(())
}

fn serialized_size_btreemap<K, V>(map: &BTreeMap<K, V>) -> usize
where
    K: Canonical,
    V: Canonical,
{
    let mut size = 0u64.encoded_len();
    for (k, v) in map.iter() {
        size += k.encoded_len();
        size += v.encoded_len();
    }
    size
}

fn deserialize_btreemap<R, K, V>(reader: &mut R) -> Result<BTreeMap<K, V>, Error>
where
    R: Read,
    K: Canonical + Ord,
    V: Canonical,
{
    let len = u64::decode(&mut *reader)? as usize;
    let mut map = BTreeMap::new();
    for _ in 0..len {
        let k = K::decode(&mut *reader)?;
        let v = V::decode(&mut *reader)?;
        insert_ascending(&mut map, k, v)?;
    }
    Ok(map)
}

/// Insert a decoded entry, rejecting a key that is not strictly greater than every key so far.
/// Maps serialize in key order, so this makes the encoding of each map unique.
fn insert_ascending<K: Ord, V>(map: &mut BTreeMap<K, V>, k: K, v: V) -> Result<(), Error> {
    if map.last_key_value().is_some_and(|(last, _)| *last >= k) {
        return Err(Error::InvalidData);
    }
    map.insert(k, v);
    Ok(())
}

// ---------------------------------------------------------------------------
// ReducedOpeningProof<F, PCS>
// ---------------------------------------------------------------------------

impl<F, PCS> Canonical for ReducedOpeningProof<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn encode<W: Write>(&self, mut writer: W) -> Result<(), Error> {
        self.sumcheck_proof().encode(&mut writer)?;
        self.joint_opening_proof().encode(&mut writer)?;
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        self.sumcheck_proof().encoded_len() + self.joint_opening_proof().encoded_len()
    }

    fn decode<R: Read>(mut reader: R) -> Result<Self, Error> {
        let sumcheck_proof = SumcheckInstanceProof::decode(&mut reader)?;
        let joint_opening_proof = PCS::Proof::decode(&mut reader)?;
        Ok(Self::from_parts(sumcheck_proof, joint_opening_proof))
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
        // 1. Proofs (BTreeMap<ProofId, SumcheckInstanceProof>)
        serialize_btreemap(self.proofs(), &mut writer)?;

        // 2. Commitments: the batch commitment, then one per polynomial
        self.batch_commitment().encode(&mut writer)?;
        encode_slice(self.commitments(), &mut writer)?;

        // 3. In-IOP evaluation reduction artifacts
        serialize_btreemap(self.eval_reduction_proofs(), &mut writer)?;

        // 4. Reduced opening proof (an `Option`: a `bool` tag, then the value)
        let reduced = self.reduced_opening_proof();
        reduced.is_some().encode(&mut writer)?;
        if let Some(r) = reduced {
            r.encode(&mut writer)?;
        }

        // 5. Transcript NARG string
        encode_slice(self.narg(), &mut writer)?;

        Ok(())
    }

    fn encoded_len(&self) -> usize {
        serialized_size_btreemap(self.proofs())
            + self.batch_commitment().encoded_len()
            + slice_encoded_len(self.commitments())
            + serialized_size_btreemap(self.eval_reduction_proofs())
            + true.encoded_len()
            + self.reduced_opening_proof().map_or(0, |r| r.encoded_len())
            + slice_encoded_len(self.narg())
    }

    fn decode<R: Read>(mut reader: R) -> Result<Self, Error> {
        let proofs = deserialize_btreemap(&mut reader)?;
        let batch_commitment = PCS::BatchCommitment::decode(&mut reader)?;
        let commitments = Vec::decode(&mut reader)?;
        let eval_reduction_proofs = deserialize_btreemap(&mut reader)?;
        let reduced_opening_proof = Option::decode(&mut reader)?;
        let narg = Vec::<u8>::decode(&mut reader)?;

        Ok(Self::from_parts(
            proofs,
            batch_commitment,
            commitments,
            eval_reduction_proofs,
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn encode(entries: &[(u64, u64)]) -> Vec<u8> {
        let mut buf = Vec::new();
        (entries.len() as u64).encode(&mut buf).unwrap();
        for (k, v) in entries {
            k.encode(&mut buf).unwrap();
            v.encode(&mut buf).unwrap();
        }
        buf
    }

    fn decode(bytes: &[u8]) -> Result<BTreeMap<u64, u64>, Error> {
        deserialize_btreemap(&mut &bytes[..])
    }

    #[test]
    fn forged_lengths_are_rejected_without_allocating() {
        // A sumcheck proof claiming 2⁶⁴ − 1 rounds, with none behind the length.
        let bytes = u64::MAX.to_le_bytes();
        assert!(
            SumcheckInstanceProof::<atlas_algebra::field::fp128::Fp128>::decode(&bytes[..])
                .is_err()
        );
    }

    #[test]
    fn map_decoding_requires_strictly_ascending_keys() {
        let map = decode(&encode(&[(1, 10), (2, 20), (5, 50)])).unwrap();
        assert_eq!(map.len(), 3);
        assert!(
            decode(&encode(&[(2, 20), (1, 10)])).is_err(),
            "unsorted keys"
        );
        assert!(
            decode(&encode(&[(1, 10), (1, 11)])).is_err(),
            "duplicate key"
        );
    }
}
