//! Transcripts on the spongefish-native surface: a spongefish [`ProverState`] for
//! the prover and a [`VerifierState`] for the verifier, over one duplex sponge.
//!
//! A value both sides compute is absorbed as a public message (the `append_*` operations). A
//! value the prover sends is a prover message: the prover absorbs it and writes it to the NARG
//! string (`send_*`), and the verifier's read of it absorbs it (`read_*`), so no read returns a
//! value the sponge has not absorbed. Proof-of-work nonces ([`Transcript::pow`]) are prover
//! messages too. The NARG string is the prover messages in the order they were sent, with no
//! tag, count or key: a field element is its canonical encoding, a `Canonical` value its
//! encoding behind a little-endian `u64` byte length.
//!
//! Full-width challenges decode through the [`ChallengeCodec`](crate::ChallengeCodec), which
//! bounds each challenge's bias by the codec budget; 125-bit challenges are uniform on their
//! set.

#[cfg(feature = "prover")]
use super::transcript::ProvingTranscript;
use super::transcript::{Transcript, VerifyingTranscript};
#[cfg(feature = "prover")]
use crate::ProverTranscript as _;
use crate::{BytesMsg, UniformBytes, VerifierTranscript as _};
use atlas_algebra::field::JoltField;
use atlas_algebra::serialize::Canonical;
use atlas_algebra::utils::errors::ProofVerifyError;
#[cfg(feature = "prover")]
use spongefish::ProverState;
use spongefish::{DuplexSpongeInit, VerificationError, VerifierState};
use std::borrow::Borrow;
use std::mem::ManuallyDrop;

/// The instance digest bound at construction. Statements are absorbed explicitly afterwards
/// (model inputs, commitments), so the construction-time instance is fixed.
const INSTANCE: [u8; 32] = [0; 32];

// One tag byte per operation keeps values of different kinds from colliding.
const TAG_MESSAGE: u8 = 1;
const TAG_BYTES: u8 = 2;
const TAG_U64: u8 = 3;
const TAG_SCALAR: u8 = 4;
const TAG_SERIALIZABLE: u8 = 5;
const TAG_SCALARS: u8 = 6;
// Prover messages: what the sponge absorbs for a sent value.
const TAG_SENT_SCALAR: u8 = 7;
const TAG_SENT_SCALARS: u8 = 8;
const TAG_SENT_SERIALIZABLE: u8 = 9;

/// A sponge usable for both transcript roles.
pub trait TranscriptSponge:
    DuplexSpongeInit<U = u8> + UniformBytes + Send + Sync + 'static
{
}
impl<H: DuplexSpongeInit<U = u8> + UniformBytes + Send + Sync + 'static> TranscriptSponge for H {}

fn tagged(tag: u8, body: &[u8]) -> BytesMsg {
    let mut v = Vec::with_capacity(1 + body.len());
    v.push(tag);
    v.extend_from_slice(body);
    BytesMsg(v)
}

fn compressed<T: Canonical>(x: &T) -> Vec<u8> {
    let mut buf = Vec::with_capacity(x.encoded_len());
    Canonical::encode(x, &mut buf).expect("in-memory serialization cannot fail");
    buf
}

/// What the sponge absorbs for a prover message: `tag ‖ body` behind its length, the encoding
/// of [`tagged`]`(tag, body)`.
fn sent(tag: u8, body: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(8 + 1 + body.len());
    v.extend_from_slice(&(1 + body.len() as u64).to_le_bytes());
    v.push(tag);
    v.extend_from_slice(body);
    v
}

/// The body absorbed for scalars sent as one message: their count, then their encodings.
fn scalars_body<F: JoltField>(scalars: &[F]) -> Vec<u8> {
    let mut body = (scalars.len() as u64).to_le_bytes().to_vec();
    for s in scalars {
        Canonical::encode(s, &mut body).expect("in-memory serialization cannot fail");
    }
    body
}

/// The byte length of a field element's canonical encoding.
fn scalar_len<F: JoltField>() -> usize {
    F::zero().encoded_len()
}

/// The field element `bytes` is the canonical encoding of; `Err` for any other byte string, so
/// one field element has one encoding in a proof.
fn decode_scalar<F: JoltField>(bytes: &[u8]) -> Result<F, VerificationError> {
    let scalar = <F as Canonical>::decode(bytes).map_err(|_| VerificationError)?;
    if compressed(&scalar) == bytes {
        Ok(scalar)
    } else {
        Err(VerificationError)
    }
}

fn read_failed(what: &str) -> ProofVerifyError {
    ProofVerifyError::Malformed(format!("proof stream: reading {what} failed"))
}

fn powers<F: JoltField>(q: F, len: usize) -> Vec<F> {
    let mut out = vec![F::one(); len];
    for i in 1..len {
        out[i] = out[i - 1] * q;
    }
    out
}

/// Challenge operations shared by both roles, written once over the role's spongefish state.
macro_rules! impl_transcript_ops {
    () => {
        fn append_message(&mut self, msg: &'static [u8]) {
            self.absorb(tagged(TAG_MESSAGE, msg));
        }

        fn append_bytes(&mut self, bytes: &[u8]) {
            self.absorb(tagged(TAG_BYTES, bytes));
        }

        fn append_u64(&mut self, x: u64) {
            self.absorb(tagged(TAG_U64, &x.to_le_bytes()));
        }

        fn append_scalar<F: JoltField>(&mut self, scalar: &F) {
            self.absorb(tagged(TAG_SCALAR, &compressed(scalar)));
        }

        fn append_serializable<F: Canonical>(&mut self, scalar: &F) {
            self.absorb(tagged(TAG_SERIALIZABLE, &compressed(scalar)));
        }

        fn append_scalars<F: JoltField>(&mut self, scalars: &[impl Borrow<F>]) {
            let mut body = (scalars.len() as u64).to_le_bytes().to_vec();
            for s in scalars {
                Canonical::encode(s.borrow(), &mut body)
                    .expect("in-memory serialization cannot fail");
            }
            self.absorb(tagged(TAG_SCALARS, &body));
        }

        fn challenge_u128(&mut self) -> u128 {
            let c = self.state.verifier_message::<u128>();
            self.record();
            c
        }

        fn challenge_scalar<F: JoltField>(&mut self) -> F {
            self.challenge_vector::<F>(1)[0]
        }

        fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F> {
            let codec = self
                .state
                .field_challenges::<F::ChallengeCodecField>(len)
                .expect("challenge length is within the codec bound");
            self.record();
            codec.into_iter().map(F::from_challenge_codec).collect()
        }

        fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F> {
            let q: F = self.challenge_scalar();
            powers(q, len)
        }

        fn challenge_scalar_optimized<F: JoltField>(&mut self) -> F::Challenge {
            F::Challenge::from(self.challenge_u128())
        }

        fn challenge_vector_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F::Challenge> {
            (0..len)
                .map(|_| self.challenge_scalar_optimized::<F>())
                .collect()
        }

        #[cfg(any(test, feature = "test-feature"))]
        fn state_history(&self) -> &[[u8; 32]] {
            &self.history.states
        }

        #[cfg(any(test, feature = "test-feature"))]
        fn compare_to(&mut self, history: Vec<[u8; 32]>) {
            self.history.expected = Some(history);
        }
    };
}

/// State history for prover/verifier lockstep comparison in tests.
#[cfg(any(test, feature = "test-feature"))]
#[derive(Default)]
struct History {
    states: Vec<[u8; 32]>,
    expected: Option<Vec<[u8; 32]>>,
}

#[cfg(any(test, feature = "test-feature"))]
impl History {
    fn push<H: spongefish::DuplexSpongeInterface<U = u8>>(&mut self, sponge: &H) {
        let mut buf = [0u8; 32];
        sponge.clone().squeeze(&mut buf);
        if let Some(expected) = &self.expected {
            assert!(
                expected.get(self.states.len()) == Some(&buf),
                "Fiat-Shamir transcript mismatch at operation {}",
                self.states.len()
            );
        }
        self.states.push(buf);
    }
}

#[cfg(feature = "prover")]
/// The prover's transcript over sponge `H`.
pub struct SpongeProver<H: TranscriptSponge> {
    state: ProverState<H>,
    #[cfg(any(test, feature = "test-feature"))]
    history: History,
}

#[cfg(feature = "prover")]
impl<H: TranscriptSponge> SpongeProver<H> {
    fn absorb(&mut self, msg: BytesMsg) {
        self.state.public_message(&msg);
        self.record();
    }

    #[inline(always)]
    fn record(&mut self) {
        #[cfg(any(test, feature = "test-feature"))]
        self.history.push(&self.state.duplex_sponge_state);
    }
}

#[cfg(feature = "prover")]
impl<H: TranscriptSponge> Transcript for SpongeProver<H> {
    impl_transcript_ops!();

    fn pow(&mut self, bits: u32) -> Result<(), ProofVerifyError> {
        let res = self
            .state
            .grind(bits)
            .map_err(|_| ProofVerifyError::ProofOfWork);
        self.record();
        res
    }
}

#[cfg(feature = "prover")]
impl<H: TranscriptSponge> ProvingTranscript for SpongeProver<H> {
    fn new(label: &'static [u8]) -> Self {
        let mut t = Self {
            state: crate::prover_transcript::<H>(label, INSTANCE),
            #[cfg(any(test, feature = "test-feature"))]
            history: History::default(),
        };
        t.record();
        t
    }

    fn send_scalar<F: JoltField>(&mut self, scalar: &F) {
        let bytes = compressed(scalar);
        self.state.prover_message_with(
            &bytes,
            |bytes| sent(TAG_SENT_SCALAR, bytes),
            |bytes, narg| narg.extend_from_slice(bytes),
        );
        self.record();
    }

    fn send_scalars<F: JoltField>(&mut self, scalars: &[F]) {
        self.state.prover_message_with(
            scalars,
            |scalars| sent(TAG_SENT_SCALARS, &scalars_body(scalars)),
            |scalars, narg| {
                for s in scalars {
                    Canonical::encode(s, &mut *narg).expect("in-memory serialization cannot fail");
                }
            },
        );
        self.record();
    }

    fn send_serializable<C: Canonical>(&mut self, value: &C) {
        let bytes = compressed(value);
        self.state.prover_message_with(
            &bytes,
            |bytes| sent(TAG_SENT_SERIALIZABLE, bytes),
            |bytes, narg| {
                narg.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                narg.extend_from_slice(bytes);
            },
        );
        self.record();
    }

    fn narg(&self) -> &[u8] {
        self.state.narg_string()
    }

    fn into_narg(self) -> Vec<u8> {
        self.state.into_narg_string()
    }
}

/// The verifier's transcript over sponge `H`, reading a NARG string it owns.
pub struct SpongeVerifier<H: TranscriptSponge> {
    /// Borrows `narg` for `'static`; it is dropped before the buffer is freed (see `Drop`).
    state: ManuallyDrop<VerifierState<'static, H>>,
    /// Owned NARG buffer from `Box::into_raw`, immutable and never moved while `state` lives.
    narg: *mut [u8],
    #[cfg(any(test, feature = "test-feature"))]
    history: History,
}

// SAFETY: `narg` is an owned, immutable heap buffer that only `state` reads; sending or sharing
// the transcript sends or shares exactly what `VerifierState<'_, H>` would.
unsafe impl<H: TranscriptSponge> Send for SpongeVerifier<H> {}
// SAFETY: as above.
unsafe impl<H: TranscriptSponge> Sync for SpongeVerifier<H> {}

impl<H: TranscriptSponge> Drop for SpongeVerifier<H> {
    fn drop(&mut self) {
        // SAFETY: `state` is dropped exactly once, here, before the buffer it borrows; `narg`
        // came from `Box::into_raw` and is freed exactly once.
        unsafe {
            ManuallyDrop::drop(&mut self.state);
            drop(Box::from_raw(self.narg));
        }
    }
}

impl<H: TranscriptSponge> SpongeVerifier<H> {
    fn absorb(&mut self, msg: BytesMsg) {
        self.state.public_message(&msg);
        self.record();
    }

    #[inline(always)]
    fn record(&mut self) {
        #[cfg(any(test, feature = "test-feature"))]
        self.history.push(&self.state.duplex_sponge_state);
    }
}

impl<H: TranscriptSponge> Transcript for SpongeVerifier<H> {
    impl_transcript_ops!();

    fn pow(&mut self, bits: u32) -> Result<(), ProofVerifyError> {
        let res = self
            .state
            .check_grind(bits)
            .map_err(|_| ProofVerifyError::ProofOfWork);
        self.record();
        res
    }
}

impl<H: TranscriptSponge> VerifyingTranscript for SpongeVerifier<H> {
    fn new(label: &'static [u8], narg: Vec<u8>) -> Self {
        let narg: *mut [u8] = Box::into_raw(narg.into_boxed_slice());
        // SAFETY: the buffer is owned by the returned transcript, never mutated, moved or freed
        // while `state` is alive, and `state` never escapes with its `'static` lifetime.
        let slice: &'static [u8] = unsafe { &*narg };
        let mut t = Self {
            state: ManuallyDrop::new(crate::verifier_transcript::<H>(label, INSTANCE, slice)),
            narg,
            #[cfg(any(test, feature = "test-feature"))]
            history: History::default(),
        };
        t.record();
        t
    }

    fn read_scalar<F: JoltField>(&mut self) -> Result<F, ProofVerifyError> {
        let scalar = self
            .state
            .prover_message_with(
                |reader| decode_scalar::<F>(reader.take(scalar_len::<F>())?),
                |scalar| sent(TAG_SENT_SCALAR, &compressed(scalar)),
            )
            .map_err(|_| read_failed("a field element"))?;
        self.record();
        Ok(scalar)
    }

    fn read_scalars<F: JoltField>(&mut self, n: usize) -> Result<Vec<F>, ProofVerifyError> {
        let scalars = self
            .state
            .prover_message_with(
                |reader| {
                    // Grows with the bytes read, never with `n` alone.
                    let mut scalars = Vec::new();
                    for _ in 0..n {
                        scalars.push(decode_scalar::<F>(reader.take(scalar_len::<F>())?)?);
                    }
                    Ok(scalars)
                },
                |scalars| sent(TAG_SENT_SCALARS, &scalars_body(scalars)),
            )
            .map_err(|_| read_failed("field elements"))?;
        self.record();
        Ok(scalars)
    }

    fn read_serializable<C: Canonical>(&mut self) -> Result<C, ProofVerifyError> {
        let (value, _) = self
            .state
            .prover_message_with(
                |reader| {
                    let len = usize::try_from(u64::from_le_bytes(reader.take_array::<8>()?))
                        .map_err(|_| VerificationError)?;
                    // `take` fails on a length past the bytes left, before the copy below.
                    let frame = reader.take(len)?;
                    let mut rest = frame;
                    let value = C::decode(&mut rest).map_err(|_| VerificationError)?;
                    if !rest.is_empty() {
                        return Err(VerificationError);
                    }
                    Ok((value, frame.to_vec()))
                },
                |(_, frame)| sent(TAG_SENT_SERIALIZABLE, frame),
            )
            .map_err(|_| read_failed("a framed value"))?;
        self.record();
        Ok(value)
    }

    fn finish(self) -> Result<(), ProofVerifyError> {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: `state` is taken exactly once and consumed before the buffer is freed; `this`
        // is never dropped, so `Drop` does not run again.
        let state = unsafe { ManuallyDrop::take(&mut this.state) };
        let res = state
            .check_eof()
            .map_err(|_| ProofVerifyError::TrailingNarg);
        unsafe {
            drop(Box::from_raw(this.narg));
            #[cfg(any(test, feature = "test-feature"))]
            std::ptr::drop_in_place(&mut this.history);
        }
        res
    }
}

#[cfg(all(test, feature = "prover"))]
mod stream_tests {
    use crate::{
        Blake2bProver, Blake2bVerifier, ProvingTranscript, Transcript, VerifyingTranscript,
    };
    use atlas_algebra::field::{fp128::Fp128 as F, Ring};

    const LABEL: &[u8] = b"stream";

    /// A stream of one scalar, a batch of three and a framed value, a challenge after each.
    fn sent() -> (Vec<u8>, [u128; 3], Vec<[u8; 32]>) {
        let mut prover = Blake2bProver::new(LABEL);
        prover.send_scalar(&F::from_u64(7));
        let a = prover.challenge_u128();
        prover.send_scalars(&[F::from_u64(1), F::from_u64(2), F::from_u64(3)]);
        let b = prover.challenge_u128();
        prover.send_serializable(&vec![5u64, 6]);
        let c = prover.challenge_u128();
        let history = prover.state_history().to_vec();
        (prover.into_narg(), [a, b, c], history)
    }

    /// Read the stream of [`sent`] back: the challenge after each read.
    fn read(narg: Vec<u8>) -> Option<[u128; 3]> {
        let mut verifier = Blake2bVerifier::new(LABEL, narg);
        let _: F = verifier.read_scalar().ok()?;
        let a = verifier.challenge_u128();
        let _: Vec<F> = verifier.read_scalars(3).ok()?;
        let b = verifier.challenge_u128();
        let _: Vec<u64> = verifier.read_serializable().ok()?;
        let c = verifier.challenge_u128();
        verifier.finish().ok()?;
        Some([a, b, c])
    }

    #[test]
    fn a_stream_reads_back_in_lockstep_with_its_prover() {
        let (narg, challenges, history) = sent();
        // No tag, count or key: 16 bytes per scalar, the frame behind its 8-byte length.
        assert_eq!(narg.len(), 16 + 3 * 16 + 8 + 24);

        let mut verifier = Blake2bVerifier::new(LABEL, narg.clone());
        verifier.compare_to(history);
        assert_eq!(verifier.read_scalar::<F>().unwrap(), F::from_u64(7));
        let a = verifier.challenge_u128();
        assert_eq!(
            verifier.read_scalars::<F>(3).unwrap(),
            [F::from_u64(1), F::from_u64(2), F::from_u64(3)]
        );
        let b = verifier.challenge_u128();
        assert_eq!(verifier.read_serializable::<Vec<u64>>().unwrap(), [5, 6]);
        let c = verifier.challenge_u128();
        verifier.finish().unwrap();
        assert_eq!([a, b, c], challenges);
        assert_eq!(read(narg), Some(challenges));
    }

    /// Every read absorbs what it returns: a stream that differs in one byte of a read gives a
    /// different challenge after that read, and the same challenges before it.
    #[test]
    fn a_changed_read_byte_changes_the_next_challenge() {
        let (narg, honest, _) = sent();
        // The low byte of the scalar, of the batch's second scalar, and of the frame's first
        // element.
        for (byte, read_index) in [(0, 0), (16 + 16, 1), (64 + 8 + 8, 2)] {
            let mut changed = narg.clone();
            changed[byte] ^= 1;
            let challenges = read(changed).expect("the changed stream still parses");
            assert_eq!(challenges[..read_index], honest[..read_index]);
            assert_ne!(challenges[read_index], honest[read_index]);
        }
    }

    #[test]
    fn a_scalar_read_rejects_a_non_canonical_encoding() {
        let modulus = 0u128.wrapping_sub(F::C);
        for value in [modulus, modulus + 1, u128::MAX] {
            let mut verifier = Blake2bVerifier::new(LABEL, value.to_le_bytes().to_vec());
            assert!(verifier.read_scalar::<F>().is_err());
            let mut verifier = Blake2bVerifier::new(LABEL, value.to_le_bytes().to_vec());
            assert!(verifier.read_scalars::<F>(1).is_err());
        }
        let mut verifier = Blake2bVerifier::new(LABEL, (modulus - 1).to_le_bytes().to_vec());
        assert_eq!(
            verifier.read_scalar::<F>().unwrap(),
            F::from_u128(modulus - 1)
        );
    }

    #[test]
    fn a_short_read_is_rejected() {
        let (narg, _, _) = sent();
        for len in 0..narg.len() {
            assert_eq!(read(narg[..len].to_vec()), None, "truncated to {len} bytes");
        }
        // More scalars than the stream holds.
        let mut verifier = Blake2bVerifier::new(LABEL, narg);
        assert!(verifier.read_scalars::<F>(1 << 40).is_err());
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let (mut narg, _, _) = sent();
        narg.push(0);
        assert_eq!(read(narg), None);
    }

    /// A frame whose stated length exceeds the bytes left is rejected before its body is
    /// copied: were the length trusted, this would allocate a tebibyte.
    #[test]
    fn a_framed_read_is_bounded_by_the_bytes_left() {
        let mut narg = (1u64 << 40).to_le_bytes().to_vec();
        narg.extend_from_slice(&[0u8; 16]);
        let mut verifier = Blake2bVerifier::new(LABEL, narg);
        assert!(verifier.read_serializable::<Vec<u64>>().is_err());

        let mut verifier = Blake2bVerifier::new(LABEL, u64::MAX.to_le_bytes().to_vec());
        assert!(verifier.read_serializable::<Vec<u64>>().is_err());
    }

    /// A frame holds exactly one encoding: bytes after it, inside the frame, are rejected.
    #[test]
    fn a_frame_with_surplus_bytes_is_rejected() {
        let mut prover = Blake2bProver::new(LABEL);
        prover.send_serializable(&(7u64, 0u8));
        let mut verifier = Blake2bVerifier::new(LABEL, prover.into_narg());
        assert!(verifier.read_serializable::<u64>().is_err());
    }
}
