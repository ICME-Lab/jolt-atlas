//! An opt-in wire format for BN254/HyperKZG proofs. Softmax's integer advice is stored as
//! contiguous u32 vectors, with one identifier per vector. Decoding constructs
//! the ordinary proof directly; it does not expand an intermediate byte buffer.
//!
//! This is serialization, not a different proof protocol. Every claim and
//! transcript value is preserved. The existing canonical format is unchanged.
//! This format does not apply to native ZK proofs or hide auxiliary vectors.
//! See `compact-format.md` beside this module for the byte layout and usage.

use super::{
    proof_serialization::serialize_btreemap, Claims, HyperKZG, ONNXProof, ReducedOpeningProof,
};
use ark_bn254::{Bn254, Fr};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Validate,
};
use common::VirtualPoly;
use joltworks::{
    field::JoltField,
    poly::{
        commitment::hyperkzg::HyperKZGProof,
        opening_proof::{OpeningId, OpeningPoint, SumcheckId},
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::{evaluation_reduction::EvalReductionProof, sumcheck::SumcheckInstanceProof},
    transcripts::Transcript,
};

type Result<T> = core::result::Result<T, SerializationError>;
const MAGIC: &[u8; 8] = b"ATLSCP01";

fn invalid<T>() -> Result<T> {
    Err(SerializationError::InvalidData)
}

fn read<T: CanonicalDeserialize>(input: &mut &[u8], compress: Compress) -> Result<T> {
    T::deserialize_with_mode(input, compress, Validate::Yes)
}

fn index(input: &mut &[u8]) -> Result<usize> {
    usize::try_from(read::<u64>(input, Compress::No)?).map_err(|_| SerializationError::InvalidData)
}

fn advice(id: OpeningId, claim: Fr) -> Option<(u8, usize, usize, u32)> {
    let (tag, node, index) = match id.virtual_poly()? {
        VirtualPoly::SoftmaxSumOutput(n, k) => (1, n, k),
        VirtualPoly::SoftmaxMaxOutput(n, k) => (2, n, k),
        VirtualPoly::SoftmaxMaxIndex(n, k) => (3, n, k),
        _ => return None,
    };
    if id.sumcheck != SumcheckId::NodeExecution(node) {
        return None;
    }
    let value = u32::try_from(claim.to_u64()?).ok()?;
    Some((tag, node, index, value))
}

fn key(tag: u8, node: usize, index: usize) -> Result<OpeningId> {
    let poly = match tag {
        1 => VirtualPoly::SoftmaxSumOutput(node, index),
        2 => VirtualPoly::SoftmaxMaxOutput(node, index),
        3 => VirtualPoly::SoftmaxMaxIndex(node, index),
        _ => return invalid(),
    };
    Ok(OpeningId::new(poly, SumcheckId::NodeExecution(node)))
}

impl Claims<Fr> {
    fn write_compact(&self, output: &mut Vec<u8>, compress: Compress) -> Result<()> {
        (self.0.len() as u64).serialize_uncompressed(&mut *output)?;
        let mut entries = self.0.iter().peekable();
        while let Some((id, (_, value))) = entries.next() {
            if let Some((tag, node, first, value)) = advice(*id, *value) {
                output.push(tag);
                (node as u64).serialize_uncompressed(&mut *output)?;
                (first as u64).serialize_uncompressed(&mut *output)?;
                let count_offset = output.len();
                0u64.serialize_uncompressed(&mut *output)?;
                value.serialize_uncompressed(&mut *output)?;
                let mut count = 1usize;
                while let Some((id, (_, value))) = entries.peek() {
                    let Some((next_tag, next_node, next_index, value)) = advice(**id, *value)
                    else {
                        break;
                    };
                    if next_tag != tag
                        || next_node != node
                        || first.checked_add(count) != Some(next_index)
                    {
                        break;
                    }
                    value.serialize_uncompressed(&mut *output)?;
                    count += 1;
                    entries.next();
                }
                output[count_offset..count_offset + 8]
                    .copy_from_slice(&(count as u64).to_le_bytes());
            } else {
                output.push(0);
                id.serialize_with_mode(&mut *output, compress)?;
                value.serialize_with_mode(&mut *output, compress)?;
            }
        }
        Ok(())
    }

    fn read_compact(
        input: &mut &[u8],
        compress: Compress,
        max_claims: usize,
        mut canonical: impl ark_serialize::Write,
    ) -> Result<Self> {
        let count = index(input)?;
        // Each entry consumes at least four bytes. Do not reserve from an
        // untrusted count, even when the application supplies a generous limit.
        if count > max_claims || count > input.len() / 4 {
            return invalid();
        }
        (count as u64).serialize_uncompressed(&mut canonical)?;
        let mut entries = Vec::new();
        let mut previous_run = None;
        while entries.len() < count {
            let tag = read::<u8>(input, Compress::No)?;
            if tag == 0 {
                // Round-trip the identifier to reject narrowed indices on 32-bit
                // targets. Field decoding below validates the canonical scalar.
                let before = *input;
                let id: OpeningId = read(input, compress)?;
                let mut equal = EqualBytes(&before[..before.len() - input.len()]);
                id.serialize_with_mode(&mut equal, compress)?;
                if !equal.0.is_empty() {
                    return invalid();
                }
                let value: Fr = read(input, compress)?;
                if advice(id, value).is_some() || entries.last().is_some_and(|(p, _)| *p >= id) {
                    return invalid();
                }
                canonical.write_all(&before[..before.len() - input.len()])?;
                entries.push((id, (OpeningPoint::default(), value)));
                previous_run = None;
            } else {
                let node = index(input)?;
                let first = index(input)?;
                let len = index(input)?;
                if len == 0 || len > count - entries.len() || len > input.len() / 4 {
                    return invalid();
                }
                let last = first
                    .checked_add(len - 1)
                    .ok_or(SerializationError::InvalidData)?;
                let first_key = key(tag, node, first)?;
                if previous_run == Some((tag, node, first))
                    || entries.last().is_some_and(|(p, _)| *p >= first_key)
                {
                    return invalid();
                }
                // A softmax identifier is 27 bytes: two tags, node/index u64s,
                // and the NodeExecution tag/node. Its BN254 scalar is 32 bytes.
                // Only the index and low scalar word vary within this run.
                let mut entry = [0u8; 59];
                first_key.serialize_uncompressed(&mut entry[..27])?;
                for k in first..=last {
                    let value: u32 = read(input, Compress::No)?;
                    entry[10..18].copy_from_slice(&(k as u64).to_le_bytes());
                    entry[27..31].copy_from_slice(&value.to_le_bytes());
                    canonical.write_all(&entry)?;
                    entries.push((
                        key(tag, node, k)?,
                        (OpeningPoint::default(), Fr::from_u32(value)),
                    ));
                }
                previous_run = last.checked_add(1).map(|next| (tag, node, next));
            }
        }
        Ok(Self(entries.into_iter().collect()))
    }
}

// Compare small canonical encodings without allocating a second byte buffer.
// This rejects narrowed indices and normalized encodings of curve points.
struct EqualBytes<'a>(&'a [u8]);
impl ark_serialize::Write for EqualBytes<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if !self.0.starts_with(bytes) {
            return Err(std::io::ErrorKind::InvalidData.into());
        }
        self.0 = &self.0[bytes.len()..];
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn read_canonical<T: CanonicalDeserialize + CanonicalSerialize>(
    input: &mut &[u8],
    compress: Compress,
) -> Result<T> {
    let before = *input;
    let value: T = read(input, compress)?;
    let mut equal = EqualBytes(&before[..before.len() - input.len()]);
    value.serialize_with_mode(&mut equal, compress)?;
    if !equal.0.is_empty() {
        return invalid();
    }
    Ok(value)
}

// Every variable-length collection is bounded by its remaining encoded bytes
// before allocating. Generic arkworks Vec decoding reserves an untrusted count.
fn read_vec<V>(
    input: &mut &[u8],
    minimum_size: usize,
    mut value: impl FnMut(&mut &[u8]) -> Result<V>,
) -> Result<Vec<V>> {
    let len = index(input)?;
    if len > input.len() / minimum_size {
        return invalid();
    }
    let mut entries = Vec::with_capacity(len);
    for _ in 0..len {
        entries.push(value(input)?);
    }
    Ok(entries)
}

// Only used for sumcheck and evaluation-reduction values, which are bounded
// vectors of canonically decoded BN254 scalars. Their encoding is unique.
fn read_map<K: CanonicalDeserialize + CanonicalSerialize + Ord, V>(
    input: &mut &[u8],
    compress: Compress,
    mut value: impl FnMut(&mut &[u8]) -> Result<V>,
) -> Result<std::collections::BTreeMap<K, V>> {
    let len = index(input)?;
    if len > input.len() {
        return invalid();
    }
    let mut entries = Vec::new();
    for _ in 0..len {
        let key: K = read_canonical(input, compress)?;
        if entries.last().is_some_and(|(previous, _)| previous >= &key) {
            return invalid();
        }
        entries.push((key, value(input)?));
    }
    Ok(entries.into_iter().collect())
}

fn scalars(input: &mut &[u8], compress: Compress) -> Result<Vec<Fr>> {
    read_vec(input, 32, |input| read(input, compress))
}

fn sumcheck<T: Transcript>(
    input: &mut &[u8],
    compress: Compress,
) -> Result<SumcheckInstanceProof<Fr, T>> {
    Ok(SumcheckInstanceProof::new(read_vec(input, 8, |input| {
        Ok(CompressedUniPoly {
            coeffs_except_linear_term: scalars(input, compress)?,
        })
    })?))
}

fn point_size(compress: Compress) -> usize {
    match compress {
        Compress::Yes => 32,
        Compress::No => 64,
    }
}

impl<T: Transcript> ONNXProof<Fr, T, HyperKZG<Bn254>> {
    /// Encode a versioned compact proof. `compress` controls the existing curve
    /// encodings in the tail. Softmax advice uses exact u32 values in both modes,
    /// including the prover's two's-complement encoding of negative maxima.
    /// Other claims are preserved as full scalars; no value is truncated.
    pub fn serialize_compact(&self, compress: Compress) -> Result<Vec<u8>> {
        let mut output = MAGIC.to_vec();
        output.push(match compress {
            Compress::Yes => 1,
            Compress::No => 0,
        });
        self.opening_claims.write_compact(&mut output, compress)?;
        self.write_compact_tail(&mut output, compress)?;
        Ok(output)
    }

    /// Decode a complete compact proof, validating scalar and group encodings.
    /// The caller supplies a claim limit appropriate for its registered model
    /// and must also bound the input size. Trailing bytes are rejected. As with
    /// canonical deserialization, callers still need to verify the proof.
    pub fn deserialize_compact(bytes: &[u8], max_claims: usize) -> Result<Self> {
        Self::deserialize_compact_with_canonical(bytes, max_claims, std::io::sink())
    }

    /// Decode while streaming the equivalent legacy canonical encoding to
    /// `canonical`, in the compact header's compression mode. This supports
    /// hashing an existing proof-byte transcript without materializing the
    /// expanded bytes or converting every scalar back out of Montgomery form.
    /// Output may be partial on error; discard it unless decoding succeeds.
    pub fn deserialize_compact_with_canonical(
        mut bytes: &[u8],
        max_claims: usize,
        mut canonical: impl ark_serialize::Write,
    ) -> Result<Self> {
        if !bytes.starts_with(MAGIC) {
            return invalid();
        }
        bytes = &bytes[MAGIC.len()..];
        let compress = match read::<u8>(&mut bytes, Compress::No)? {
            0 => Compress::No,
            1 => Compress::Yes,
            _ => return invalid(),
        };
        let opening_claims =
            Claims::read_compact(&mut bytes, compress, max_claims, &mut canonical)?;
        let tail = bytes;
        let proof = Self {
            opening_claims,
            proofs: read_map(&mut bytes, compress, |input| sumcheck(input, compress))?,
            commitments: read_vec(&mut bytes, point_size(compress), |input| {
                read_canonical(input, compress)
            })?,
            eval_reduction_proofs: read_map(&mut bytes, compress, |input| {
                Ok(EvalReductionProof {
                    h: UniPoly {
                        coeffs: scalars(input, compress)?,
                    },
                })
            })?,
            reduced_opening_proof: match read::<u8>(&mut bytes, Compress::No)? {
                0 => None,
                1 => Some(ReducedOpeningProof {
                    sumcheck_proof: sumcheck(&mut bytes, compress)?,
                    sumcheck_claims: scalars(&mut bytes, compress)?,
                    joint_opening_proof: HyperKZGProof {
                        com: read_vec(&mut bytes, point_size(compress), |input| {
                            read_canonical(input, compress)
                        })?,
                        w: read_vec(&mut bytes, point_size(compress), |input| {
                            read_canonical(input, compress)
                        })?,
                        v: read_vec(&mut bytes, 8, |input| scalars(input, compress))?,
                    },
                }),
                _ => return invalid(),
            },
        };
        if !bytes.is_empty() {
            return invalid();
        }
        canonical.write_all(tail)?;
        Ok(proof)
    }

    fn write_compact_tail(
        &self,
        mut output: impl ark_serialize::Write,
        compress: Compress,
    ) -> Result<()> {
        serialize_btreemap(&self.proofs, &mut output, compress)?;
        self.commitments
            .serialize_with_mode(&mut output, compress)?;
        serialize_btreemap(&self.eval_reduction_proofs, &mut output, compress)?;
        self.reduced_opening_proof
            .serialize_with_mode(output, compress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_proof::{
        AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    };
    use ark_bn254::Bn254;
    use atlas_onnx_tracer::{model::test::ModelBuilder, tensor::Tensor};
    use joltworks::{poly::commitment::hyperkzg::HyperKZG, transcripts::Blake2bTranscript};
    use std::collections::BTreeMap;

    type Proof = ONNXProof<Fr, Blake2bTranscript, HyperKZG<Bn254>>;

    fn fixture() -> Proof {
        let mut claims = BTreeMap::new();
        for tag in 1..=3 {
            for (k, v) in [0u32, 1, i32::MAX as u32, u32::MAX].into_iter().enumerate() {
                claims.insert(
                    key(tag, 17, k).unwrap(),
                    (OpeningPoint::default(), Fr::from(v)),
                );
            }
            // Gaps and arbitrary field values must survive exactly.
            claims.insert(
                key(tag, 17, 6).unwrap(),
                (OpeningPoint::default(), Fr::from(u64::MAX)),
            );
            claims.insert(
                key(tag, 17, 9).unwrap(),
                (OpeningPoint::default(), -Fr::from(1u32)),
            );
        }
        claims.insert(
            OpeningId::new(VirtualPoly::SoftmaxSumOutput(17, 0), SumcheckId::Raf),
            (OpeningPoint::default(), Fr::from(7u32)),
        );
        claims.insert(
            OpeningId::new(VirtualPoly::NodeOutput(2), SumcheckId::NodeExecution(3)),
            (OpeningPoint::default(), Fr::from(0u32)),
        );
        Proof {
            opening_claims: Claims(claims),
            proofs: BTreeMap::new(),
            commitments: Vec::new(),
            eval_reduction_proofs: BTreeMap::new(),
            reduced_opening_proof: None,
        }
    }

    fn canonical(proof: &Proof, compress: Compress) -> Vec<u8> {
        let mut out = Vec::new();
        proof.serialize_with_mode(&mut out, compress).unwrap();
        out
    }

    #[test]
    fn exact_legacy_roundtrip_both_modes_and_claim_limits() {
        let proof = fixture();
        for compress in [Compress::Yes, Compress::No] {
            let encoded = proof.serialize_compact(compress).unwrap();
            let decoded =
                Proof::deserialize_compact(&encoded, proof.opening_claims.0.len()).unwrap();
            assert_eq!(canonical(&proof, compress), canonical(&decoded, compress));
            assert_eq!(encoded, decoded.serialize_compact(compress).unwrap());
            let mut streamed = Vec::new();
            Proof::deserialize_compact_with_canonical(&encoded, 100, &mut streamed).unwrap();
            assert_eq!(streamed, canonical(&proof, compress));
            assert!(encoded.len() < canonical(&proof, compress).len());
            assert!(
                Proof::deserialize_compact(&encoded, proof.opening_claims.0.len() - 1).is_err()
            );
        }
        let mut empty = fixture();
        empty.opening_claims.0.clear();
        let bytes = empty.serialize_compact(Compress::No).unwrap();
        assert!(Proof::deserialize_compact(&bytes, 0).is_ok());
    }

    #[test]
    fn rejects_bad_framing_and_truncation_without_panicking() {
        let proof = fixture();
        for compress in [Compress::Yes, Compress::No] {
            let bytes = proof.serialize_compact(compress).unwrap();
            for end in 0..bytes.len() {
                assert!(Proof::deserialize_compact(&bytes[..end], 100).is_err());
            }
            let mut extra = bytes.clone();
            extra.push(0);
            assert!(Proof::deserialize_compact(&extra, 100).is_err());
            // Mutations can encode a different proof, but cannot alias this one.
            for pos in 0..bytes.len() {
                let mut changed = bytes.clone();
                changed[pos] ^= 0x80;
                if let Ok(decoded) = Proof::deserialize_compact(&changed, 100) {
                    assert_ne!(canonical(&proof, compress), canonical(&decoded, compress));
                    assert_eq!(changed, decoded.serialize_compact(compress).unwrap());
                }
            }
            let mut huge = bytes.clone();
            huge[9..17].copy_from_slice(&u64::MAX.to_le_bytes());
            assert!(Proof::deserialize_compact(&huge, usize::MAX).is_err());
        }
    }

    fn run(out: &mut Vec<u8>, tag: u8, node: u64, first: u64, count: u64) {
        out.push(tag);
        for value in [node, first, count] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for _ in 0..count.min(4) {
            out.extend_from_slice(&0u32.to_le_bytes());
        }
    }

    #[test]
    fn rejects_noncanonical_and_overflowing_runs() {
        let decode = |body: &[u8], count: u64| {
            let mut bytes = count.to_le_bytes().to_vec();
            bytes.extend_from_slice(body);
            Claims::read_compact(&mut bytes.as_slice(), Compress::No, 20, std::io::sink())
        };
        let mut good = Vec::new();
        run(&mut good, 1, 17, 0, 2);
        assert!(decode(&good, 2).is_ok());
        for (tag, first, len) in [
            (4, 0, 2),
            (1, 0, 0),
            (1, 0, 3),
            (1, u64::MAX, 2),
            (1, 0, u64::MAX),
        ] {
            let mut bad = Vec::new();
            run(&mut bad, tag, 17, first, len);
            assert!(decode(&bad, 2).is_err());
        }
        for second in [0, 1] {
            let mut split = Vec::new();
            run(&mut split, 1, 17, 0, 1);
            run(&mut split, 1, 17, second, 1);
            assert!(decode(&split, 2).is_err());
        }
        let mut raw = vec![0];
        key(1, 17, 0)
            .unwrap()
            .serialize_uncompressed(&mut raw)
            .unwrap();
        Fr::from(0u32).serialize_uncompressed(&mut raw).unwrap();
        assert!(decode(&raw, 1).is_err());
        // A canonical full scalar is required, even for non-advice entries.
        let mut bad_scalar = vec![0];
        OpeningId::new(VirtualPoly::NodeOutput(0), SumcheckId::Raf)
            .serialize_uncompressed(&mut bad_scalar)
            .unwrap();
        bad_scalar.extend_from_slice(&[255; 32]);
        assert!(decode(&bad_scalar, 1).is_err());
    }

    #[test]
    fn rejects_duplicate_or_reordered_tail_maps_and_writer_failure() {
        use crate::onnx_proof::{ProofId, ProofType};
        let proof = fixture();
        let bytes = proof.serialize_compact(Compress::No).unwrap();
        assert!(Proof::deserialize_compact_with_canonical(&bytes, 100, EqualBytes(&[])).is_err());
        let mut remaining = &bytes[9..];
        Claims::read_compact(&mut remaining, Compress::No, 100, std::io::sink()).unwrap();
        let offset = bytes.len() - remaining.len();
        for nodes in [[2usize, 2], [3, 2]] {
            let mut changed = bytes[..offset].to_vec();
            2u64.serialize_uncompressed(&mut changed).unwrap();
            for node in nodes {
                ProofId(node, ProofType::Execution)
                    .serialize_uncompressed(&mut changed)
                    .unwrap();
                // SumcheckInstanceProof with no messages.
                0u64.serialize_uncompressed(&mut changed).unwrap();
            }
            changed.extend_from_slice(&bytes[offset + 8..]);
            assert!(Proof::deserialize_compact(&changed, 100).is_err());
        }
    }

    #[test]
    fn complete_softmax_proof_verifies_and_tampering_rejects() {
        let mut builder = ModelBuilder::with_scale(common::consts::MODEL_SCALE as u32);
        let input_node = builder.input(vec![2, 8]);
        let output = builder.softmax_last_axis(input_node);
        builder.mark_output(output);
        let pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(
            AtlasSharedPreprocessing::preprocess(builder.build()),
        );
        let input = Tensor::new(
            Some(&[
                -80, -70, -60, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 60, 70, 80,
            ]),
            &[2, 8],
        )
        .unwrap();
        let (proof, io, _) = Proof::prove(&pp, &[input]);
        let vp = AtlasVerifierPreprocessing::from(&pp);
        for compress in [Compress::Yes, Compress::No] {
            let bytes = proof.serialize_compact(compress).unwrap();
            let mut decoded = Proof::deserialize_compact(&bytes, 100_000).unwrap();
            assert_eq!(canonical(&decoded, compress), canonical(&proof, compress));
            let mut streamed = Vec::new();
            Proof::deserialize_compact_with_canonical(&bytes, 100_000, &mut streamed).unwrap();
            assert_eq!(streamed, canonical(&proof, compress));
            decoded.verify(&vp, &io, None).unwrap();
            let id = *decoded
                .opening_claims
                .0
                .keys()
                .find(|id| matches!(id.virtual_poly(), Some(VirtualPoly::SoftmaxSumOutput(..))))
                .unwrap();
            decoded.opening_claims.0.get_mut(&id).unwrap().1 += Fr::from(1u32);
            let changed = decoded.serialize_compact(compress).unwrap();
            assert!(Proof::deserialize_compact(&changed, 100_000)
                .unwrap()
                .verify(&vp, &io, None)
                .is_err());
        }
    }
}
