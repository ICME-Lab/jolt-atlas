use ark_serialize::CanonicalSerialize;
use atlas_algebra::field::JoltField;
use atlas_algebra::utils::errors::ProofVerifyError;
use std::borrow::Borrow;

/// The Fiat-Shamir operations prover and verifier share. A protocol is written once, generic
/// over `Transcript`; the prover runs it on a [`ProvingTranscript`] and the verifier on a
/// [`VerifyingTranscript`], so the two sides cannot swap roles.
pub trait Transcript: Send + Sync + 'static {
    fn append_message(&mut self, msg: &'static [u8]);
    fn append_bytes(&mut self, bytes: &[u8]);
    fn append_u64(&mut self, x: u64);
    fn append_scalar<F: JoltField>(&mut self, scalar: &F);
    fn append_serializable<F: CanonicalSerialize>(&mut self, scalar: &F);
    fn append_scalars<F: JoltField>(&mut self, scalars: &[impl Borrow<F>]);
    fn challenge_u128(&mut self) -> u128;
    /// A full-width field challenge, decoded through the transcript's challenge codec.
    fn challenge_scalar<F: JoltField>(&mut self) -> F;
    fn challenge_scalar_128_bits<F: JoltField>(&mut self) -> F;
    fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F>;
    // Compute powers of scalar q : (1, q, q^2, ..., q^(len-1))
    fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F>;
    // New methods that return F::Challenge
    fn challenge_scalar_optimized<F: JoltField>(&mut self) -> F::Challenge;
    fn challenge_vector_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F::Challenge>;
    fn challenge_scalar_powers_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F>;

    /// Proof of work of difficulty `bits` before the next challenge: the prover grinds a nonce
    /// into the proof's NARG string, the verifier reads and checks it. `0` is a no-op.
    fn pow(&mut self, bits: u32) -> Result<(), ProofVerifyError>;

    /// The sponge state after every operation so far, starting from construction.
    #[cfg(any(test, feature = "test-feature"))]
    fn state_history(&self) -> &[[u8; 32]];

    /// Panic at the first operation whose resulting state differs from `history` (typically
    /// the prover's), so a prover/verifier divergence is reported where it happens.
    #[cfg(any(test, feature = "test-feature"))]
    fn compare_to(&mut self, history: Vec<[u8; 32]>);
}

/// The prover's side: a fresh transcript whose proof of work is written to a NARG string that
/// travels with the proof.
pub trait ProvingTranscript: Transcript {
    fn new(label: &'static [u8]) -> Self;

    /// The NARG string written so far.
    fn narg(&self) -> &[u8];

    fn into_narg(self) -> Vec<u8>;
}

/// The verifier's side: a transcript that reads the prover's NARG string.
pub trait VerifyingTranscript: Transcript {
    fn new(label: &'static [u8], narg: Vec<u8>) -> Self;

    /// Reject unless the whole NARG string was consumed.
    fn finish(self) -> Result<(), ProofVerifyError>;
}

pub trait AppendToTranscript {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript);
}

/// The empty batch commitment of a homomorphic scheme: absorbs nothing.
impl AppendToTranscript for () {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, _transcript: &mut ProofTranscript) {
    }
}
