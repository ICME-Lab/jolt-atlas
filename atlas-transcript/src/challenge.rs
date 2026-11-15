//! The transcript side of field challenges: the sponges whose output the challenge codec may
//! decode ([`UniformBytes`]) and the per-message byte budget. The codec itself
//! ([`ChallengeCodec`]) lives with the field, in `atlas_algebra::field::challenge`.

#![forbid(unsafe_code)]
#![deny(
    clippy::indexing_slicing,
    clippy::get_unwrap,
    clippy::string_slice,
    clippy::fallible_impl_from,
    clippy::mem_forget,
    clippy::exit,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::host_endian_bytes,
    clippy::wildcard_enum_match_arm
)]

use atlas_algebra::field::challenge::{ChallengeCodec, MAX_CHALLENGE_BYTES};
use spongefish::{DuplexSpongeInterface, VerificationError};

use crate::Blake2b512;
use sealed::Sealed;

mod sealed {
    pub trait Sealed {}
}

/// A sponge whose squeezed bytes are uniform, up to the sponge's own
/// security. Field challenges are decoded from raw squeezed bytes, so the
/// codecs' TV bounds hold only on such a sponge.
///
/// It is public because it bounds the sponge of the public
/// `field_challenges` methods. Callers name it only as that bound and cannot
/// implement it.
pub trait UniformBytes: DuplexSpongeInterface<U = u8> + Sealed {}

impl Sealed for Blake2b512 {}
impl UniformBytes for Blake2b512 {}

pub(crate) fn message_bytes<F: ChallengeCodec>(len: usize) -> Result<usize, VerificationError> {
    F::challenge_bytes(len)
        .filter(|&bytes| bytes <= MAX_CHALLENGE_BYTES)
        .ok_or(VerificationError)
}

#[cfg(all(test, feature = "prover"))]
mod tests {
    use super::{Sealed, UniformBytes};
    use crate::{
        prover_transcript, verifier_transcript, BytesMsg, ProverTranscript, VerifierTranscript,
    };
    use jolt_field::Prime64Offset59;
    use spongefish::{DuplexSpongeInit, DuplexSpongeInterface};

    /// A sponge whose output is all ones, so every pooled candidate is
    /// rejected. It lives here because `UniformBytes` is sealed.
    #[derive(Clone)]
    struct Saturated;

    impl DuplexSpongeInterface for Saturated {
        type U = u8;

        fn absorb(&mut self, _input: &[u8]) -> &mut Self {
            self
        }

        fn squeeze(&mut self, output: &mut [u8]) -> &mut Self {
            output.fill(0xff);
            self
        }
    }

    impl DuplexSpongeInit for Saturated {
        fn init(_session_id: &[u8; 32]) -> Self {
            Self
        }
    }

    impl Sealed for Saturated {}
    impl UniformBytes for Saturated {}

    const SESSION: &[u8] = b"jolt/challenge-codecs/v1";
    const INSTANCE: [u8; 32] = [0x42; 32];

    #[test]
    fn exhaustion_fails_on_both_sides_and_poisons_the_verifier() {
        let mut prover = prover_transcript::<Saturated>(SESSION, INSTANCE);
        assert!(ProverTranscript::field_challenges::<Prime64Offset59>(&mut prover, 3).is_err());

        let mut prover = prover_transcript::<Saturated>(SESSION, INSTANCE);
        prover.prover_message(&BytesMsg(b"after".to_vec()));
        let mut verifier =
            verifier_transcript::<Saturated>(SESSION, INSTANCE, prover.narg_string());
        assert!(VerifierTranscript::field_challenges::<Prime64Offset59>(&mut verifier, 3).is_err());
        assert!(verifier.prover_message::<BytesMsg>().is_err());
        assert!(verifier.check_eof().is_err());
    }
}
