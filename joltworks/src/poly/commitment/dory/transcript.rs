//! Bridges joltworks' [`Transcript`] into dory-pcs's transcript trait so a Dory
//! opening shares the surrounding proof's Fiat-Shamir transcript.
//!
//! Prover/verifier parity is by construction: both sides traverse this adapter
//! in the same order, so the exact (label-free) byte layout only needs to be
//! self-consistent. Byte/group/serde appends are length-prefixed to keep the
//! absorb injective.

use ark_bn254::Fr;
use dory::backends::arkworks::{ArkFr, BN254};
use dory::primitives::arithmetic::Group as DoryGroup;
use dory::primitives::transcript::Transcript as DoryTranscript;
use dory::primitives::DorySerialize;

use crate::transcripts::Transcript;

pub struct LocalToDoryTranscript<'a, T: Transcript> {
    transcript: &'a mut T,
}

impl<'a, T: Transcript> LocalToDoryTranscript<'a, T> {
    pub fn new(transcript: &'a mut T) -> Self {
        Self { transcript }
    }
}

impl<T: Transcript> DoryTranscript for LocalToDoryTranscript<'_, T> {
    type Curve = BN254;

    fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
        self.transcript.append_u64(bytes.len() as u64);
        self.transcript.append_bytes(bytes);
    }

    fn append_field(&mut self, _label: &[u8], x: &ArkFr) {
        self.transcript.append_scalar::<Fr>(&x.0);
    }

    fn append_group<G: DoryGroup>(&mut self, _label: &[u8], g: &G) {
        let mut buf = Vec::new();
        g.serialize_compressed(&mut buf)
            .expect("group serialization is infallible");
        self.transcript.append_u64(buf.len() as u64);
        self.transcript.append_bytes(&buf);
    }

    fn append_serde<S: DorySerialize>(&mut self, _label: &[u8], s: &S) {
        let mut buf = Vec::new();
        s.serialize_compressed(&mut buf)
            .expect("DorySerialize serialization is infallible");
        self.transcript.append_u64(buf.len() as u64);
        self.transcript.append_bytes(&buf);
    }

    fn challenge_scalar(&mut self, _label: &[u8]) -> ArkFr {
        ArkFr(self.transcript.challenge_scalar::<Fr>())
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        unreachable!("dory-pcs does not invoke Transcript::reset on the prove/verify path")
    }
}
