//! Serialization and deserialization for [`ONNXProof`] and related types.
//!
//! Uses `ark_serialize`'s `CanonicalSerialize` / `CanonicalDeserialize` traits.
//! BTreeMap fields are serialized as sorted `(key, value)` pairs prefixed by their length.

use std::collections::BTreeMap;

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};

use joltworks::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        opening_proof::{OpeningId, OpeningPoint},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
};

use crate::{Claims, ONNXProof, ReducedOpeningProof};

// ---------------------------------------------------------------------------
// BTreeMap helpers
// ---------------------------------------------------------------------------

fn serialize_btreemap<W, K, V>(
    map: &BTreeMap<K, V>,
    writer: &mut W,
    compress: Compress,
) -> Result<(), SerializationError>
where
    W: Write,
    K: CanonicalSerialize,
    V: CanonicalSerialize,
{
    (map.len() as u64).serialize_with_mode(&mut *writer, compress)?;
    for (k, v) in map.iter() {
        k.serialize_with_mode(&mut *writer, compress)?;
        v.serialize_with_mode(&mut *writer, compress)?;
    }
    Ok(())
}

fn serialized_size_btreemap<K, V>(map: &BTreeMap<K, V>, compress: Compress) -> usize
where
    K: CanonicalSerialize,
    V: CanonicalSerialize,
{
    let mut size = 0u64.serialized_size(compress);
    for (k, v) in map.iter() {
        size += k.serialized_size(compress);
        size += v.serialized_size(compress);
    }
    size
}

fn deserialize_btreemap<R, K, V>(
    reader: &mut R,
    compress: Compress,
    validate: Validate,
) -> Result<BTreeMap<K, V>, SerializationError>
where
    R: Read,
    K: CanonicalDeserialize + Ord,
    V: CanonicalDeserialize,
{
    let len = u64::deserialize_with_mode(&mut *reader, compress, validate)? as usize;
    let mut map = BTreeMap::new();
    for _ in 0..len {
        let k = K::deserialize_with_mode(&mut *reader, compress, validate)?;
        let v = V::deserialize_with_mode(&mut *reader, compress, validate)?;
        insert_ascending(&mut map, k, v)?;
    }
    Ok(map)
}

/// Insert a decoded entry, rejecting a key that is not strictly greater than every key so far.
/// Maps serialize in key order, so this makes the encoding of each map unique.
fn insert_ascending<K: Ord, V>(
    map: &mut BTreeMap<K, V>,
    k: K,
    v: V,
) -> Result<(), SerializationError> {
    if map.last_key_value().is_some_and(|(last, _)| *last >= k) {
        return Err(SerializationError::InvalidData);
    }
    map.insert(k, v);
    Ok(())
}

// ---------------------------------------------------------------------------
// Claims<F>
// ---------------------------------------------------------------------------

impl<F: JoltField> CanonicalSerialize for Claims<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (self.openings().len() as u64).serialize_with_mode(&mut writer, compress)?;
        for (key, (_point, claim)) in self.openings().iter() {
            key.serialize_with_mode(&mut writer, compress)?;
            claim.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let mut size = 0u64.serialized_size(compress);
        for (key, (_point, claim)) in self.openings().iter() {
            size += key.serialized_size(compress);
            size += claim.serialized_size(compress);
        }
        size
    }
}

impl<F: JoltField> Valid for Claims<F> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: JoltField> CanonicalDeserialize for Claims<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let len = u64::deserialize_with_mode(&mut reader, compress, validate)? as usize;
        let mut claims = BTreeMap::new();
        for _ in 0..len {
            let key = OpeningId::deserialize_with_mode(&mut reader, compress, validate)?;
            let claim = F::deserialize_with_mode(&mut reader, compress, validate)?;
            insert_ascending(&mut claims, key, (OpeningPoint::default(), claim))?;
        }
        Ok(Claims::from_parts(claims))
    }
}

// ---------------------------------------------------------------------------
// ReducedOpeningProof<F, PCS>
// ---------------------------------------------------------------------------

impl<F, PCS> CanonicalSerialize for ReducedOpeningProof<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.sumcheck_proof()
            .serialize_with_mode(&mut writer, compress)?;
        self.sumcheck_claims()
            .serialize_with_mode(&mut writer, compress)?;
        self.joint_opening_proof()
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.sumcheck_proof().serialized_size(compress)
            + self.sumcheck_claims().serialized_size(compress)
            + self.joint_opening_proof().serialized_size(compress)
    }
}

impl<F, PCS> Valid for ReducedOpeningProof<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F, PCS> CanonicalDeserialize for ReducedOpeningProof<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let sumcheck_proof =
            SumcheckInstanceProof::deserialize_with_mode(&mut reader, compress, validate)?;
        let sumcheck_claims = Vec::deserialize_with_mode(&mut reader, compress, validate)?;
        let joint_opening_proof =
            PCS::Proof::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(Self::from_parts(
            sumcheck_proof,
            sumcheck_claims,
            joint_opening_proof,
        ))
    }
}

// ---------------------------------------------------------------------------
// ONNXProof<F, PCS>
// ---------------------------------------------------------------------------

impl<F, PCS> CanonicalSerialize for ONNXProof<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        // 1. Opening claims
        self.opening_claims()
            .serialize_with_mode(&mut writer, compress)?;

        // 2. Proofs (BTreeMap<ProofId, SumcheckInstanceProof>)
        serialize_btreemap(self.proofs(), &mut writer, compress)?;

        // 3. Commitments
        self.commitments()
            .serialize_with_mode(&mut writer, compress)?;

        // 4. In-IOP evaluation reduction artifacts
        serialize_btreemap(self.eval_reduction_proofs(), &mut writer, compress)?;

        // 5. Reduced opening proof (Option, encoded as ark-serialize encodes `Option`)
        let reduced = self.reduced_opening_proof();
        reduced
            .is_some()
            .serialize_with_mode(&mut writer, compress)?;
        if let Some(r) = reduced {
            r.serialize_with_mode(&mut writer, compress)?;
        }

        // 6. Transcript NARG string
        self.narg().serialize_with_mode(&mut writer, compress)?;

        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.opening_claims().serialized_size(compress)
            + serialized_size_btreemap(self.proofs(), compress)
            + self.commitments().serialized_size(compress)
            + serialized_size_btreemap(self.eval_reduction_proofs(), compress)
            + true.serialized_size(compress)
            + self
                .reduced_opening_proof()
                .map_or(0, |r| r.serialized_size(compress))
            + self.narg().serialized_size(compress)
    }
}

impl<F, PCS> Valid for ONNXProof<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F, PCS> CanonicalDeserialize for ONNXProof<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let opening_claims = Claims::deserialize_with_mode(&mut reader, compress, validate)?;
        let proofs = deserialize_btreemap(&mut reader, compress, validate)?;
        let commitments = Vec::deserialize_with_mode(&mut reader, compress, validate)?;
        let eval_reduction_proofs = deserialize_btreemap(&mut reader, compress, validate)?;
        let reduced_opening_proof = Option::deserialize_with_mode(&mut reader, compress, validate)?;
        let narg = Vec::<u8>::deserialize_with_mode(&mut reader, compress, validate)?;

        Ok(Self::from_parts(
            opening_claims,
            proofs,
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

/// Serialize a proof to bytes (compressed).
pub fn serialize_proof<F, PCS>(proof: &ONNXProof<F, PCS>) -> Result<Vec<u8>, SerializationError>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut buf = Vec::with_capacity(proof.serialized_size(Compress::Yes));
    proof.serialize_compressed(&mut buf)?;
    Ok(buf)
}

/// Deserialize a proof from bytes (compressed). The bytes must hold exactly one proof: trailing
/// bytes are rejected, so a proof has a single byte encoding.
pub fn deserialize_proof<F, PCS>(bytes: &[u8]) -> Result<ONNXProof<F, PCS>, SerializationError>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut reader = bytes;
    let proof = ONNXProof::deserialize_compressed(&mut reader)?;
    if !reader.is_empty() {
        return Err(SerializationError::InvalidData);
    }
    Ok(proof)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(entries: &[(u64, u64)]) -> Vec<u8> {
        let mut buf = Vec::new();
        (entries.len() as u64)
            .serialize_compressed(&mut buf)
            .unwrap();
        for (k, v) in entries {
            k.serialize_compressed(&mut buf).unwrap();
            v.serialize_compressed(&mut buf).unwrap();
        }
        buf
    }

    fn decode(bytes: &[u8]) -> Result<BTreeMap<u64, u64>, SerializationError> {
        deserialize_btreemap(&mut &bytes[..], Compress::Yes, Validate::Yes)
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
