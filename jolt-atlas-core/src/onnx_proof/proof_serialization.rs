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
        opening_proof::{OpeningId, OpeningPoint, VirtualOperandClaims},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
};

use super::{Claims, ONNXProof, ReducedOpeningProof};

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
        map.insert(k, v);
    }
    Ok(map)
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
        (self.0.len() as u64).serialize_with_mode(&mut writer, compress)?;
        for (key, (_point, claim)) in self.0.iter() {
            key.serialize_with_mode(&mut writer, compress)?;
            claim.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let mut size = 0u64.serialized_size(compress);
        for (key, (_point, claim)) in self.0.iter() {
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
            claims.insert(key, (OpeningPoint::default(), claim));
        }
        Ok(Claims(claims))
    }
}

// ---------------------------------------------------------------------------
// VirtualOperandClaims<F> helper (type alias for BTreeMap<usize, Vec<F>>)
// ---------------------------------------------------------------------------

fn serialize_virtual_operand_claims<W: Write, F: JoltField>(
    claims: &VirtualOperandClaims<F>,
    writer: &mut W,
    compress: Compress,
) -> Result<(), SerializationError> {
    serialize_btreemap(claims, writer, compress)
}

fn serialized_size_virtual_operand_claims<F: JoltField>(
    claims: &VirtualOperandClaims<F>,
    compress: Compress,
) -> usize {
    serialized_size_btreemap(claims, compress)
}

fn deserialize_virtual_operand_claims<R: Read, F: JoltField>(
    reader: &mut R,
    compress: Compress,
    validate: Validate,
) -> Result<VirtualOperandClaims<F>, SerializationError> {
    deserialize_btreemap(reader, compress, validate)
}

// ---------------------------------------------------------------------------
// ReducedOpeningProof<F, T, PCS>
// ---------------------------------------------------------------------------

impl<F, T, PCS> CanonicalSerialize for ReducedOpeningProof<F, T, PCS>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.sumcheck_claims
            .serialize_with_mode(&mut writer, compress)?;
        self.joint_opening_proof
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.sumcheck_proof.serialized_size(compress)
            + self.sumcheck_claims.serialized_size(compress)
            + self.joint_opening_proof.serialized_size(compress)
    }
}

impl<F, T, PCS> Valid for ReducedOpeningProof<F, T, PCS>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F, T, PCS> CanonicalDeserialize for ReducedOpeningProof<F, T, PCS>
where
    F: JoltField,
    T: Transcript,
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
        Ok(Self {
            sumcheck_proof,
            sumcheck_claims,
            joint_opening_proof,
        })
    }
}

// ---------------------------------------------------------------------------
// ONNXProof<F, T, PCS>
// ---------------------------------------------------------------------------

impl<F, T, PCS> CanonicalSerialize for ONNXProof<F, T, PCS>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        // 1. Opening claims
        self.opening_claims
            .serialize_with_mode(&mut writer, compress)?;

        // 2. Proofs (BTreeMap<ProofId, SumcheckInstanceProof>)
        serialize_btreemap(&self.proofs, &mut writer, compress)?;

        // 3. Virtual operand claims
        serialize_virtual_operand_claims(&self.virtual_operand_claims, &mut writer, compress)?;

        // 4. Commitments
        self.commitments
            .serialize_with_mode(&mut writer, compress)?;

        // 5. Reduced opening proof (Option)
        self.reduced_opening_proof
            .serialize_with_mode(&mut writer, compress)?;

        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.opening_claims.serialized_size(compress)
            + serialized_size_btreemap(&self.proofs, compress)
            + serialized_size_virtual_operand_claims(&self.virtual_operand_claims, compress)
            + self.commitments.serialized_size(compress)
            + self.reduced_opening_proof.serialized_size(compress)
    }
}

impl<F, T, PCS> Valid for ONNXProof<F, T, PCS>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F, T, PCS> CanonicalDeserialize for ONNXProof<F, T, PCS>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let opening_claims = Claims::deserialize_with_mode(&mut reader, compress, validate)?;
        let proofs = deserialize_btreemap(&mut reader, compress, validate)?;
        let virtual_operand_claims =
            deserialize_virtual_operand_claims(&mut reader, compress, validate)?;
        let commitments = Vec::deserialize_with_mode(&mut reader, compress, validate)?;
        let reduced_opening_proof = Option::deserialize_with_mode(&mut reader, compress, validate)?;

        Ok(Self {
            opening_claims,
            proofs,
            virtual_operand_claims,
            commitments,
            reduced_opening_proof,
        })
    }
}

// ---------------------------------------------------------------------------
// Convenience helpers
// ---------------------------------------------------------------------------

/// Serialize a proof to bytes (compressed).
pub fn serialize_proof<F, T, PCS>(
    proof: &ONNXProof<F, T, PCS>,
) -> Result<Vec<u8>, SerializationError>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    let mut buf = Vec::with_capacity(proof.serialized_size(Compress::Yes));
    proof.serialize_compressed(&mut buf)?;
    Ok(buf)
}

/// Deserialize a proof from bytes (compressed).
pub fn deserialize_proof<F, T, PCS>(
    bytes: &[u8],
) -> Result<ONNXProof<F, T, PCS>, SerializationError>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    ONNXProof::deserialize_compressed(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_proof::{
        AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    };
    use ark_bn254::{Bn254, Fr};
    use atlas_onnx_tracer::{
        model::{Model, RunArgs},
        tensor::Tensor,
    };
    use joltworks::{poly::commitment::hyperkzg::HyperKZG, transcripts::Blake2bTranscript};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    type TestProof = ONNXProof<Fr, Blake2bTranscript, HyperKZG<Bn254>>;

    #[test]
    fn test_proof_serialization_roundtrip_transformer() {
        // --- Setup ---
        let working_dir = "../atlas-onnx-tracer/models/transformer/";
        let model = Model::load(&format!("{working_dir}network.onnx"), &RunArgs::default());

        let pp = AtlasSharedPreprocessing::preprocess(model);
        let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);

        // Generate random input matching transformer model expectations
        let mut rng = StdRng::seed_from_u64(0x1096);
        let input_data: Vec<i32> = (0..64 * 64)
            .map(|_| (1 << 7) + rng.gen_range(-50..=50))
            .collect();
        let input = Tensor::new(Some(&input_data), &[1, 64, 64]).unwrap();

        // --- Prove ---
        let (proof, io, _debug_info) = TestProof::prove(&prover_pp, &[input]);

        // --- Serialize ---
        let bytes = serialize_proof(&proof).expect("serialization should succeed");
        println!(
            "Serialized proof size: {:.1} kB ({} bytes)",
            bytes.len() as f64 / 1024.0,
            bytes.len()
        );
        assert!(!bytes.is_empty(), "serialized proof should not be empty");

        // --- Deserialize ---
        let deserialized: TestProof =
            deserialize_proof(&bytes).expect("deserialization should succeed");

        // --- Verify structural equality ---
        // Check opening claims match
        assert_eq!(
            proof.opening_claims.0.len(),
            deserialized.opening_claims.0.len(),
            "opening claims count mismatch"
        );
        for (key, (_point, claim)) in proof.opening_claims.0.iter() {
            let (_, deserialized_claim) = deserialized
                .opening_claims
                .0
                .get(key)
                .expect("deserialized proof missing opening claim key");
            assert_eq!(
                claim, deserialized_claim,
                "claim value mismatch for {key:?}"
            );
        }

        // Check proofs map matches
        assert_eq!(
            proof.proofs.len(),
            deserialized.proofs.len(),
            "proof count mismatch"
        );
        for (proof_id, original_proof) in proof.proofs.iter() {
            let deser_proof = deserialized
                .proofs
                .get(proof_id)
                .expect("deserialized proof missing proof ID");
            assert_eq!(
                original_proof.compressed_polys.len(),
                deser_proof.compressed_polys.len(),
                "sumcheck proof round count mismatch for {proof_id:?}"
            );
        }

        // Check virtual operand claims match
        assert_eq!(
            proof.virtual_operand_claims.len(),
            deserialized.virtual_operand_claims.len(),
            "virtual operand claims count mismatch"
        );

        // Check commitments match
        assert_eq!(
            proof.commitments.len(),
            deserialized.commitments.len(),
            "commitments count mismatch"
        );
        assert_eq!(
            proof.commitments, deserialized.commitments,
            "commitments mismatch"
        );

        // --- Verify the deserialized proof still verifies ---
        let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);
        deserialized
            .verify(&verifier_pp, &io, None)
            .expect("deserialized proof should verify successfully");

        // --- Double round-trip ---
        let bytes2 = serialize_proof(&deserialized).expect("re-serialization should succeed");
        assert_eq!(
            bytes, bytes2,
            "double round-trip should produce identical bytes"
        );
    }
}
