//! A verifier transcript that logs what it reads.
//!
//! [`Recording`] wraps a verifier transcript and logs, for each read, its byte range in the
//! proof and its kind. The proof-mutation sweep verifies an honest proof under it once and
//! takes its sites from the log, so every value the verifier reads is a site by construction:
//! a read added to the protocol shows up in the log.

use std::borrow::Borrow;
use std::sync::{Arc, Mutex};

use atlas_algebra::{field::JoltField, serialize::Canonical, utils::errors::ProofVerifyError};
use atlas_sumcheck::{
    engine::SUMCHECK_ROUND_TRANSCRIPT_LABEL, evaluation_reduction::H_BEGIN_TRANSCRIPT_LABEL,
};
use atlas_transcript::{Transcript, VerifyingTranscript};

/// What a read is, by what the protocol reads there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum ReadKind {
    /// The batch commitment or a polynomial's commitment: a framed value read before the
    /// first challenge.
    Commitment,
    /// An opening claim, or an opening group's reduced claim: a single field element.
    Claim,
    /// One coefficient of a sumcheck round message.
    RoundCoeff,
    /// One coefficient of an evaluation reduction's polynomial.
    ReductionCoeff,
    /// The joint opening proof: a framed value read after the first challenge.
    OpeningProof,
    /// A proof-of-work nonce.
    PowNonce,
}

impl ReadKind {
    /// Every kind of read.
    pub(crate) const ALL: [Self; 6] = [
        Self::Commitment,
        Self::Claim,
        Self::RoundCoeff,
        Self::ReductionCoeff,
        Self::OpeningProof,
        Self::PowNonce,
    ];

    /// Whether the read is a framed value: a byte length, then that many bytes.
    pub(crate) fn is_framed(self) -> bool {
        matches!(self, Self::Commitment | Self::OpeningProof)
    }
}

/// One read: its byte range in the proof and its kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Read {
    pub offset: usize,
    pub len: usize,
    pub kind: ReadKind,
}

/// The reads of one verification, in order. Shared, so the harness keeps a handle while the
/// verifier owns the transcript.
pub(crate) type ReadLog = Arc<Mutex<Vec<Read>>>;

/// A verifier transcript that logs each read of the transcript it wraps.
///
/// The wrapped transcript does every operation; the wrapper follows the read position in its
/// own copy of the proof to size the reads whose length the proof states (a framed value's
/// length prefix, a nonce's LEB128 bytes). A batch of field elements is a round message or an
/// evaluation reduction's polynomial, told apart by the label absorbed before it.
pub(crate) struct Recording<T: VerifyingTranscript> {
    inner: T,
    narg: Vec<u8>,
    offset: usize,
    /// Whether a challenge was drawn: a framed value read before is a commitment.
    challenged: bool,
    /// What the next batch of field elements is, from the last label absorbed.
    batch_kind: Option<ReadKind>,
    log: ReadLog,
}

impl<T: VerifyingTranscript> Recording<T> {
    /// A handle on the log.
    pub(crate) fn log(&self) -> ReadLog {
        self.log.clone()
    }

    fn record(&mut self, len: usize, kind: ReadKind) {
        self.log.lock().expect("read log").push(Read {
            offset: self.offset,
            len,
            kind,
        });
        self.offset += len;
    }
}

impl<T: VerifyingTranscript> Transcript for Recording<T> {
    fn append_message(&mut self, msg: &'static [u8]) {
        if msg == H_BEGIN_TRANSCRIPT_LABEL {
            self.batch_kind = Some(ReadKind::ReductionCoeff);
        }
        self.inner.append_message(msg);
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        if bytes.starts_with(SUMCHECK_ROUND_TRANSCRIPT_LABEL) {
            self.batch_kind = Some(ReadKind::RoundCoeff);
        }
        self.inner.append_bytes(bytes);
    }

    fn append_u64(&mut self, x: u64) {
        self.inner.append_u64(x);
    }

    fn append_scalar<F: JoltField>(&mut self, scalar: &F) {
        self.inner.append_scalar(scalar);
    }

    fn append_serializable<F: Canonical>(&mut self, scalar: &F) {
        self.inner.append_serializable(scalar);
    }

    fn append_scalars<F: JoltField>(&mut self, scalars: &[impl Borrow<F>]) {
        self.inner.append_scalars(scalars);
    }

    fn challenge_u128(&mut self) -> u128 {
        self.challenged = true;
        self.inner.challenge_u128()
    }

    fn challenge_scalar<F: JoltField>(&mut self) -> F {
        self.challenged = true;
        self.inner.challenge_scalar()
    }

    fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        self.challenged = true;
        self.inner.challenge_vector(len)
    }

    fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        self.challenged = true;
        self.inner.challenge_scalar_powers(len)
    }

    fn challenge_scalar_optimized<F: JoltField>(&mut self) -> F::Challenge {
        self.challenged = true;
        self.inner.challenge_scalar_optimized::<F>()
    }

    fn challenge_vector_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F::Challenge> {
        self.challenged = true;
        self.inner.challenge_vector_optimized::<F>(len)
    }

    fn pow(&mut self, bits: u32) -> Result<(), ProofVerifyError> {
        self.inner.pow(bits)?;
        if bits > 0 {
            // The nonce is LEB128: every byte but its last has the high bit set.
            let len = self.narg[self.offset..]
                .iter()
                .position(|byte| byte & 0x80 == 0)
                .expect("a checked nonce ends")
                + 1;
            self.record(len, ReadKind::PowNonce);
        }
        Ok(())
    }

    fn state_history(&self) -> &[[u8; 32]] {
        self.inner.state_history()
    }

    fn compare_to(&mut self, history: Vec<[u8; 32]>) {
        self.inner.compare_to(history);
    }
}

impl<T: VerifyingTranscript> VerifyingTranscript for Recording<T> {
    fn new(label: &'static [u8], narg: Vec<u8>) -> Self {
        Self {
            inner: T::new(label, narg.clone()),
            narg,
            offset: 0,
            challenged: false,
            batch_kind: None,
            log: ReadLog::default(),
        }
    }

    fn read_scalar<F: JoltField>(&mut self) -> Result<F, ProofVerifyError> {
        let scalar: F = self.inner.read_scalar()?;
        self.record(scalar.encoded_len(), ReadKind::Claim);
        Ok(scalar)
    }

    fn read_scalars<F: JoltField>(&mut self, n: usize) -> Result<Vec<F>, ProofVerifyError> {
        let scalars: Vec<F> = self.inner.read_scalars(n)?;
        let kind = self
            .batch_kind
            .take()
            .expect("a batch of field elements follows the label that says what it is");
        for scalar in &scalars {
            self.record(scalar.encoded_len(), kind);
        }
        Ok(scalars)
    }

    fn read_serializable<C: Canonical>(&mut self) -> Result<C, ProofVerifyError> {
        let value: C = self.inner.read_serializable()?;
        let prefix: [u8; 8] = self.narg[self.offset..self.offset + 8]
            .try_into()
            .expect("eight bytes");
        let kind = if self.challenged {
            ReadKind::OpeningProof
        } else {
            ReadKind::Commitment
        };
        self.record(8 + u64::from_le_bytes(prefix) as usize, kind);
        Ok(value)
    }

    fn finish(self) -> Result<(), ProofVerifyError> {
        self.inner.finish()
    }
}
