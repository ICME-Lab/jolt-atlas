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
use std::sync::Arc;

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
    pub(crate) row_commitments: Arc<Vec<ArkG1>>,
    pub(crate) commit_blind: ArkFr,
}

impl DoryHint {
    pub(crate) fn into_parts(self) -> (Vec<ArkG1>, ArkFr) {
        (
            Arc::unwrap_or_clone(self.row_commitments),
            self.commit_blind,
        )
    }
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
    /// Affine copy of `prover.g1_vec`, so sparse (one-hot) commits can use
    /// mixed projective+affine additions. Derived, never serialized.
    pub(crate) g1_affine: Vec<ark_bn254::G1Affine>,
    /// Prepared (Miller-loop-ready) copy of `prover.g2_vec`, so tier-2 pairings
    /// over a subset of rows need no per-call G2 preparation. Derived.
    pub(crate) g2_prepared: Vec<<ark_bn254::Bn254 as ark_ec::pairing::Pairing>::G2Prepared>,
}

impl DoryProverSetup {
    pub(crate) fn new(prover: ProverSetup<BN254>, verifier: VerifierSetup<BN254>) -> Self {
        let projective: Vec<ark_bn254::G1Projective> = prover.g1_vec.iter().map(|g| g.0).collect();
        let g1_affine = ark_ec::CurveGroup::normalize_batch(&projective);
        let g2_prepared = {
            use rayon::prelude::*;
            prover
                .g2_vec
                .par_iter()
                .map(|g| {
                    let affine: ark_bn254::G2Affine = ark_ec::CurveGroup::into_affine(g.0);
                    affine.into()
                })
                .collect()
        };
        Self {
            prover,
            verifier,
            g1_affine,
            g2_prepared,
        }
    }
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
        Ok(Self::new(prover, verifier))
    }
}

#[cfg(test)]
mod shared_hint_tests {
    use super::*;
    use ark_bn254::{Fr, G1Projective};
    use ark_ec::PrimeGroup;

    #[test]
    fn native_shared_hints_preserve_owned_and_borrowed_rows() {
        let rows = vec![ArkG1(G1Projective::generator()); 16];
        let hint = DoryHint {
            row_commitments: rows.clone().into(),
            commit_blind: ArkFr(Fr::from(7u64)),
        };
        let copied = hint.clone();
        assert!(Arc::ptr_eq(&hint.row_commitments, &copied.row_commitments));
        let (mut owned, blind) = copied.into_parts();
        assert_eq!(owned, rows);
        assert_eq!(blind, hint.commit_blind);
        owned[0] = ArkG1(G1Projective::generator()).scale(&ArkFr(Fr::from(2u64)));
        assert_eq!(hint.row_commitments.as_ref(), &rows);
        let address = hint.row_commitments.as_ptr();
        let (unique, _) = hint.into_parts();
        assert_eq!(address, unique.as_ptr());
        assert_eq!(unique, rows);
    }

    #[test]
    #[ignore = "Isolated hint clone storage and time, not complete proof memory"]
    fn native_shared_hints_benchmark() {
        let mode = std::env::var("NATIVE_SHARED_HINT_MODE").unwrap();
        assert!(mode == "copy" || mode == "share");
        let hint = DoryHint {
            row_commitments: vec![ArkG1(G1Projective::generator()); 8192].into(),
            commit_blind: ArkFr(Fr::from(7u64)),
        };
        let start = std::time::Instant::now();
        let copies = (0..512)
            .map(|_| {
                if mode == "copy" {
                    DoryHint {
                        row_commitments: hint.row_commitments.as_ref().clone().into(),
                        commit_blind: hint.commit_blind,
                    }
                } else {
                    hint.clone()
                }
            })
            .collect::<Vec<_>>();
        let seconds = start.elapsed().as_secs_f64();
        let mut unique = std::collections::BTreeSet::new();
        unique.insert(hint.row_commitments.as_ptr() as usize);
        for copy in &copies {
            unique.insert(copy.row_commitments.as_ptr() as usize);
        }
        println!("SHARED_HINT_BENCH {{\"mode\":\"{}\",\"seconds\":{},\"rows\":8192,\"clones\":512,\"unique_allocations\":{},\"allocated_row_bytes\":{},\"complete_proof\":false}}",mode,seconds,unique.len(),unique.len()*8192*std::mem::size_of::<ArkG1>());
        std::hint::black_box(copies);
    }
}
