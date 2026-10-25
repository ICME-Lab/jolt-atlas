//! Transcripts on upstream `jolt-transcript`'s split surface: a spongefish [`ProverState`] for
//! the prover and a [`VerifierState`] for the verifier, over one duplex sponge.
//!
//! Every value the protocol absorbs is a public message (the verifier reads it from the
//! serialized proof and absorbs the same bytes). Full-width challenges decode through the
//! upstream [`ChallengeCodec`](jolt_transcript::ChallengeCodec), which bounds each challenge's
//! bias by the codec budget; 125-bit challenges are uniform on their set. The only prover
//! messages are proof-of-work nonces ([`Transcript::pow`]), so the NARG string holds just those.

#[cfg(feature = "prover")]
use super::transcript::ProvingTranscript;
use super::transcript::{Transcript, VerifyingTranscript};
use crate::field::JoltField;
use crate::utils::errors::ProofVerifyError;
use ark_ec::CurveGroup;
use ark_serialize::CanonicalSerialize;
#[cfg(feature = "prover")]
use jolt_transcript::ProverTranscript as _;
use jolt_transcript::{BytesMsg, UniformBytes, VerifierTranscript as _};
#[cfg(feature = "prover")]
use spongefish::ProverState;
use spongefish::{DuplexSpongeInit, VerifierState};
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
const TAG_POINT: u8 = 7;
const TAG_POINTS: u8 = 8;

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

fn compressed<T: CanonicalSerialize>(x: &T) -> Vec<u8> {
    let mut buf = Vec::with_capacity(x.compressed_size());
    x.serialize_compressed(&mut buf)
        .expect("in-memory serialization cannot fail");
    buf
}

fn point_bytes<G: CurveGroup>(point: &G) -> Vec<u8> {
    compressed(&point.into_affine())
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

        fn append_serializable<F: CanonicalSerialize>(&mut self, scalar: &F) {
            self.absorb(tagged(TAG_SERIALIZABLE, &compressed(scalar)));
        }

        fn append_scalars<F: JoltField>(&mut self, scalars: &[impl Borrow<F>]) {
            let mut body = (scalars.len() as u64).to_le_bytes().to_vec();
            for s in scalars {
                s.borrow()
                    .serialize_compressed(&mut body)
                    .expect("in-memory serialization cannot fail");
            }
            self.absorb(tagged(TAG_SCALARS, &body));
        }

        fn append_point<G: CurveGroup>(&mut self, point: &G) {
            self.absorb(tagged(TAG_POINT, &point_bytes(point)));
        }

        fn append_points<G: CurveGroup>(&mut self, points: &[G]) {
            let mut body = (points.len() as u64).to_le_bytes().to_vec();
            for p in points {
                body.extend_from_slice(&point_bytes(p));
            }
            self.absorb(tagged(TAG_POINTS, &body));
        }

        fn challenge_u128(&mut self) -> u128 {
            let c = self.state.verifier_message::<u128>();
            self.record();
            c
        }

        fn challenge_scalar<F: JoltField>(&mut self) -> F {
            self.challenge_vector::<F>(1)[0]
        }

        fn challenge_scalar_128_bits<F: JoltField>(&mut self) -> F {
            F::from_u128(self.challenge_u128())
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

        fn challenge_scalar_powers_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F> {
            // Each power is an optimized challenge-by-field product.
            let q: F::Challenge = self.challenge_scalar_optimized::<F>();
            let mut q_powers = vec![F::one(); len];
            for i in 1..len {
                q_powers[i] = q * q_powers[i - 1];
            }
            q_powers
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
            state: jolt_transcript::prover_transcript::<H>(label, INSTANCE),
            #[cfg(any(test, feature = "test-feature"))]
            history: History::default(),
        };
        t.record();
        t
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
            state: ManuallyDrop::new(jolt_transcript::verifier_transcript::<H>(
                label, INSTANCE, slice,
            )),
            narg,
            #[cfg(any(test, feature = "test-feature"))]
            history: History::default(),
        };
        t.record();
        t
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
