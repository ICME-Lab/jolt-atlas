mod sponge;
mod transcript;

#[cfg(feature = "prover")]
pub use sponge::SpongeProver;
pub use sponge::{SpongeVerifier, TranscriptSponge};
pub use transcript::{AppendToTranscript, ProvingTranscript, Transcript, VerifyingTranscript};

/// Blake2b-512 duplex-sponge transcripts (the default).
#[cfg(feature = "prover")]
pub type Blake2bProver = SpongeProver<jolt_transcript::Blake2b512>;
pub type Blake2bVerifier = SpongeVerifier<jolt_transcript::Blake2b512>;

/// Keccak-f1600 duplex-sponge transcripts.
#[cfg(feature = "prover")]
pub type KeccakProver = SpongeProver<spongefish::instantiations::Keccak>;
pub type KeccakVerifier = SpongeVerifier<spongefish::instantiations::Keccak>;
