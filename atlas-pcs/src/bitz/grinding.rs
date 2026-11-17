//! Proof of work before BitZ's projection challenge.
//!
//! The projection onto a random prime is the one step whose error is far above the target, so
//! its proof of work is the largest in a proof (23 bits at `2^30` committed bits). It is a
//! search for a nonce whose keyed BLAKE3 hash, under a key squeezed from the transcript, has
//! that many leading zero bits; the nonce is then absorbed, so the challenges that follow
//! depend on it. A prover who wants another prime must change what it sent, which changes the
//! key and costs another search.
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::unimplemented,
    clippy::todo,
    clippy::indexing_slicing
)]

use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_transcript::Transcript;
use rayon::prelude::*;

/// The largest difficulty: the nonce space holds `2^24` times the expected number of trials.
pub const MAX_BITS: u32 = 40;

/// The nonces each thread scans in one block of the search.
const NONCES_PER_THREAD: u64 = 1 << 14;

/// The key of a search, squeezed from the transcript.
fn key<T: Transcript>(transcript: &mut T) -> [u8; 32] {
    let mut key = [0u8; 32];
    let (low, high) = key.split_at_mut(16);
    low.copy_from_slice(&transcript.challenge_u128().to_le_bytes());
    high.copy_from_slice(&transcript.challenge_u128().to_le_bytes());
    key
}

/// Whether `nonce` has at least `bits` leading zero bits under `key`.
fn accepts(key: &[u8; 32], nonce: u64, bits: u32) -> bool {
    let hash = blake3::keyed_hash(key, &nonce.to_le_bytes());
    let mut word = [0u8; 8];
    for (w, h) in word.iter_mut().zip(hash.as_bytes()) {
        *w = *h;
    }
    u64::from_be_bytes(word).leading_zeros() >= bits
}

/// The prover's side: finds the least accepted nonce, absorbs it and returns it. A difficulty
/// of zero searches nothing and sends the nonce zero.
pub fn prove<T: Transcript>(bits: u32, transcript: &mut T) -> u64 {
    let nonce = if bits == 0 {
        0
    } else {
        let key = key(transcript);
        let block = NONCES_PER_THREAD * rayon::current_num_threads().max(1) as u64;
        let mut start = 0u64;
        loop {
            let end = start.saturating_add(block);
            let found = (start..end)
                .into_par_iter()
                .find_first(|&nonce| accepts(&key, nonce, bits));
            if let Some(nonce) = found {
                break nonce;
            }
            start = end;
        }
    };
    transcript.append_u64(nonce);
    nonce
}

/// The verifier's side: checks `nonce` at difficulty `bits` and absorbs it.
pub fn verify<T: Transcript>(
    bits: u32,
    nonce: u64,
    transcript: &mut T,
) -> Result<(), ProofVerifyError> {
    if bits > MAX_BITS {
        return Err(ProofVerifyError::Malformed(
            "BitZ: proof-of-work difficulty".into(),
        ));
    }
    let accepted = if bits == 0 {
        nonce == 0
    } else {
        accepts(&key(transcript), nonce, bits)
    };
    if !accepted {
        return Err(ProofVerifyError::ProofOfWork);
    }
    transcript.append_u64(nonce);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use atlas_transcript::{
        Blake2bProver, Blake2bVerifier, ProvingTranscript, VerifyingTranscript,
    };

    fn transcripts(label: u64) -> (Blake2bProver, Blake2bVerifier) {
        let mut prover = Blake2bProver::new(b"grinding-test");
        prover.append_u64(label);
        let mut verifier = Blake2bVerifier::new(b"grinding-test", Vec::new());
        verifier.append_u64(label);
        (prover, verifier)
    }

    #[test]
    fn the_least_nonce_is_found_and_accepted() {
        for bits in [1, 6, 12] {
            let (mut prover, mut verifier) = transcripts(bits as u64);
            let nonce = prove(bits, &mut prover);
            // Nothing below it is accepted under the same key.
            let (mut again, _) = transcripts(bits as u64);
            let key = key(&mut again);
            assert!(accepts(&key, nonce, bits));
            assert!((0..nonce).all(|lower| !accepts(&key, lower, bits)));
            verify(bits, nonce, &mut verifier).unwrap();
            // Both sides absorbed the nonce: their next challenges agree.
            assert_eq!(prover.challenge_u128(), verifier.challenge_u128());
        }
    }

    #[test]
    fn other_nonces_and_other_transcripts_are_rejected() {
        let bits = 14;
        let (mut prover, _) = transcripts(1);
        let nonce = prove(bits, &mut prover);
        let (_, mut verifier) = transcripts(1);
        assert!(verify(bits, nonce + 1, &mut verifier).is_err());
        // The nonce of one transcript does not carry to another.
        let (_, mut other) = transcripts(2);
        assert!(verify(bits, nonce, &mut other).is_err());
        let (_, mut verifier) = transcripts(1);
        assert!(verify(MAX_BITS + 1, nonce, &mut verifier).is_err());
    }

    #[test]
    fn no_difficulty_sends_the_zero_nonce() {
        let (mut prover, mut verifier) = transcripts(3);
        assert_eq!(prove(0, &mut prover), 0);
        verify(0, 0, &mut verifier).unwrap();
        assert_eq!(prover.challenge_u128(), verifier.challenge_u128());
        let (_, mut verifier) = transcripts(3);
        assert!(verify(0, 1, &mut verifier).is_err());
    }
}
