#![allow(
    clippy::assertions_on_result_states,
    clippy::from_over_into,
    clippy::len_without_is_empty,
    clippy::needless_range_loop,
    clippy::new_without_default,
    clippy::too_long_first_doc_paragraph,
    long_running_const_eval,
    non_snake_case,
    type_alias_bounds
)]
//! The atlas Fiat-Shamir transcript.
//!
//! [`Transcript`] is the one transcript trait every protocol is written against; the prover runs
//! it on a [`SpongeProver`] and the verifier on a [`SpongeVerifier`], spongefish duplex-sponge
//! states (Blake2b-512 by default). Under them are the spongefish-native surfaces taken from
//! upstream `jolt-transcript` ([`ProverTranscript`], [`VerifierTranscript`], the session
//! [`setup`](prover_transcript), the proof-of-work step), and [`words`], the deployed proof
//! format's word encodings.

mod challenge;
mod codec;
mod pow;
mod prover;
mod setup;
mod sponge;
mod transcript;
mod verifier;
pub mod words;

pub use atlas_algebra::field::challenge::{
    grind_bits_for_loss, ChallengeCodec, ChallengeSetSize, GrindBitsError, MAX_CHALLENGE_BYTES,
    MAX_POW_BITS,
};
pub use challenge::UniformBytes;
pub use codec::BytesMsg;
#[cfg(feature = "prover")]
pub use pow::{MessageSearch, Probe};
pub use prover::ProverTranscript;
pub use setup::{prover_transcript, session_id, verifier_transcript, PROTOCOL_ID};
#[cfg(feature = "prover")]
pub use sponge::SpongeProver;
pub use sponge::{SpongeVerifier, TranscriptSponge};
pub use transcript::{AppendToTranscript, ProvingTranscript, Transcript, VerifyingTranscript};
pub use verifier::VerifierTranscript;

/// Spongefish's fixed-output hash bridge instantiated with Blake2b-512.
pub type Blake2b512 = spongefish::instantiations::Hash<blake2::Blake2b512>;

/// Blake2b-512 duplex-sponge transcripts (the default).
#[cfg(feature = "prover")]
pub type Blake2bProver = SpongeProver<Blake2b512>;
pub type Blake2bVerifier = SpongeVerifier<Blake2b512>;

#[cfg(test)]
mod fp128_tests {
    // `Into<Self>` is exercised deliberately: it is `JoltField::Challenge` trait surface.
    #![allow(clippy::useless_conversion)]
    use crate::{Blake2bProver, ProvingTranscript, Transcript};
    use atlas_algebra::field::Ring;
    use atlas_algebra::field::{fp128::Fp128 as F, JoltField};

    fn modulus() -> u128 {
        0u128.wrapping_sub(F::C)
    }

    #[test]
    fn challenge_via_transcript() {
        let mut t1 = Blake2bProver::new(b"fp128");
        let mut t2 = Blake2bProver::new(b"fp128");
        for t in [&mut t1, &mut t2] {
            t.append_u64(42);
            t.append_scalar(&F::from_u64(7));
        }

        let c1: <F as JoltField>::Challenge = t1.challenge_scalar_optimized::<F>();
        let c2 = t2.challenge_scalar_optimized::<F>();
        assert_eq!(c1, c2);

        // Challenge = Self: conversion into the field is the identity and
        // challenge arithmetic is field arithmetic.
        let f: F = c1.into();
        assert_eq!(f, c1);
        let x = F::from_u64(3);
        assert_eq!(c1 * x, f * x);
        assert_eq!(x - c1, x - f);

        let s1: F = t1.challenge_scalar();
        let s2: F = t2.challenge_scalar();
        assert_eq!(s1, s2);
        assert!(s1.to_canonical_u128() < modulus());

        assert_eq!(
            <F as JoltField>::Challenge::from(u128::MAX),
            F::from_u128(F::C - 1)
        );
    }
}

mod univariate {
    use crate::{AppendToTranscript, Transcript};
    use atlas_algebra::{field::JoltField, poly::univariate::UnivariatePoly};

    impl<F: JoltField> AppendToTranscript for UnivariatePoly<F> {
        fn append_to_transcript<ProofTranscript: Transcript>(
            &self,
            transcript: &mut ProofTranscript,
        ) {
            transcript.append_message(b"UncompressedUniPoly_begin");
            for coeff in self.coefficients() {
                transcript.append_scalar(coeff);
            }
            transcript.append_message(b"UncompressedUniPoly_end");
        }
    }
}
