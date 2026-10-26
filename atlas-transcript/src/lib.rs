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

#[cfg(test)]
mod fp128_tests {
    // `Into<Self>` is exercised deliberately: it is `JoltField::Challenge` trait surface.
    #![allow(clippy::useless_conversion)]
    use crate::{Blake2bProver, ProvingTranscript, Transcript};
    use atlas_algebra::field::{
        fp128::{Fp128 as F, Inner},
        JoltField,
    };
    use jolt_field::Ring;

    fn modulus() -> u128 {
        0u128.wrapping_sub(Inner::C)
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
        assert!(s1.0.to_canonical_u128() < modulus());

        assert_eq!(
            <F as JoltField>::Challenge::from(u128::MAX),
            F::from_u128(Inner::C - 1)
        );
    }
}

mod unipoly {
    use crate::{AppendToTranscript, Transcript};
    use atlas_algebra::{
        field::JoltField,
        poly::unipoly::{CompressedUniPoly, UniPoly},
    };

    impl<F: JoltField> AppendToTranscript for UniPoly<F> {
        fn append_to_transcript<ProofTranscript: Transcript>(
            &self,
            transcript: &mut ProofTranscript,
        ) {
            transcript.append_message(b"UncompressedUniPoly_begin");
            for i in 0..self.coeffs.len() {
                transcript.append_scalar(&self.coeffs[i]);
            }
            transcript.append_message(b"UncompressedUniPoly_end");
        }
    }

    impl<F: JoltField> AppendToTranscript for CompressedUniPoly<F> {
        fn append_to_transcript<ProofTranscript: Transcript>(
            &self,
            transcript: &mut ProofTranscript,
        ) {
            transcript.append_message(b"UniPoly_begin");
            for i in 0..self.coeffs_except_linear_term.len() {
                transcript.append_scalar(&self.coeffs_except_linear_term[i]);
            }
            transcript.append_message(b"UniPoly_end");
        }
    }
}
