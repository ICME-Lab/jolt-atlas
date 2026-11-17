//! Proof of work on the transcript sponge, as a standalone step.
//!
//! The prover
//! sends a nonce below `2^(bits + 7)` as canonical LEB128, one byte per prover
//! message, and both parties squeeze `ceil(bits / 8)` bytes and accept iff
//! their low `bits` bits are zero. The step protects the verifier messages
//! that follow it, up to the next prover message. The check is squeezed in one
//! call; on every sponge here that gives the same bytes as squeezing it one
//! byte at a time, because each buffers its squeeze output.
//!
//! The same layout, search primitive and probe are proposed for spongefish in
//! arkworks-rs/spongefish#301 (`PowTranscriptExt::sponge_pow`,
//! `MessageSearch`, `Probe`); none of them is in the published 0.8.0.
//!
//! The prover searches by evaluating candidate nonces on a copy of the sponge
//! state ([`MessageSearch`]). Spongefish 0.8.0 exposes that state only under
//! its `yolocrypto` feature, so the search is behind this crate's
//! `prover` feature. Cargo unifies features across a build, so a
//! binary that links a grinding prover compiles its verifier with
//! `yolocrypto` too; in 0.8.0 the feature changes visibility only, not bytes.

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

#[cfg(feature = "prover")]
use spongefish::DuplexSpongeInterface;

use atlas_algebra::field::challenge::MAX_POW_BITS;

/// Evaluates the transcript on a candidate prover message without sending it.
///
/// A probe is [`Sync`], so a search may run on several threads at once.
#[cfg(feature = "prover")]
pub trait Probe: Sync {
    /// Writes into `out` the bytes the transcript would squeeze right after
    /// absorbing `candidate`. The transcript itself is unchanged.
    fn probe(&self, candidate: &[u8], out: &mut [u8]);
}

/// A copy of the prover's public sponge, for searching over a prover message:
/// a proof-of-work nonce, or a retry counter whose following challenges must
/// satisfy a prover-side condition.
///
/// A probe absorbs the candidate one byte per absorb call, exactly as sending
/// it as single-byte prover messages does. For the Blake2b sponge this equals
/// one absorb of the whole candidate.
///
/// It holds no private randomness, writes no proof bytes and cannot restore
/// or advance the live prover state.
#[cfg(feature = "prover")]
#[derive(Clone)]
pub struct MessageSearch<H> {
    sponge: H,
}

#[cfg(feature = "prover")]
impl<H> MessageSearch<H> {
    pub(crate) const fn new(sponge: H) -> Self {
        Self { sponge }
    }
}

#[cfg(feature = "prover")]
impl<H: DuplexSpongeInterface<U = u8> + Sync> Probe for MessageSearch<H> {
    fn probe(&self, candidate: &[u8], out: &mut [u8]) {
        let mut sponge = self.sponge.clone();
        for byte in candidate {
            let _ = sponge.absorb(core::slice::from_ref(byte));
        }
        let _ = sponge.squeeze(out);
    }
}

/// The nonces `0..=SEQUENTIAL_NONCES` are searched on the calling thread.
#[cfg(feature = "prover")]
const SEQUENTIAL_NONCES: u32 = 255;

/// The nonces each thread scans in one parallel block of the search.
#[cfg(feature = "prover")]
const NONCES_PER_THREAD: u32 = 1 << 12;

/// The layout of one proof-of-work step at a difficulty of `1..=MAX_POW_BITS`.
#[derive(Clone, Copy)]
pub(crate) struct SpongePow {
    bits: u32,
}

impl SpongePow {
    /// `None` unless `1 <= bits <= MAX_POW_BITS`.
    pub(crate) const fn new(bits: u32) -> Option<Self> {
        if bits == 0 || bits > MAX_POW_BITS {
            None
        } else {
            Some(Self { bits })
        }
    }

    /// The largest nonce, `2^(bits + 7) - 1`.
    pub(crate) const fn last_nonce(self) -> u32 {
        u32::MAX >> (MAX_POW_BITS - self.bits)
    }

    /// `ceil((bits + 7) / 7)` bytes encode every nonce, at most 5.
    pub(crate) const fn max_nonce_len(self) -> usize {
        (self.bits as usize + 13) / 7
    }

    /// `ceil(bits / 8)` squeezed bytes carry the checked bits, at most 4.
    pub(crate) const fn word_len(self) -> usize {
        (self.bits as usize).div_ceil(8)
    }

    /// The low `bits` bits of the little-endian check bytes are zero.
    /// Rejects check bytes of the wrong length.
    pub(crate) fn accepts(self, check: &[u8]) -> bool {
        let mut word = [0u8; 4];
        match word.get_mut(..self.word_len()) {
            Some(prefix) if prefix.len() == check.len() => prefix.copy_from_slice(check),
            _ => return false,
        }
        u32::from_le_bytes(word) & ((1u32 << self.bits) - 1) == 0
    }

    /// Reads a nonce byte by byte through `next`. Returns `None` unless its
    /// encoding is canonical LEB128 and it is at most `last_nonce`; read
    /// errors pass through.
    pub(crate) fn read_nonce<E>(
        self,
        mut next: impl FnMut() -> Result<u8, E>,
    ) -> Result<Option<u32>, E> {
        let mut value = 0u64;
        for i in 0..self.max_nonce_len() {
            let byte = next()?;
            value |= u64::from(byte & 0x7f) << (7 * i);
            if byte & 0x80 == 0 {
                // A canonical encoding has no trailing zero byte.
                let canonical = i == 0 || byte != 0;
                return Ok(u32::try_from(value)
                    .ok()
                    .filter(|&nonce| canonical && nonce <= self.last_nonce()));
            }
        }
        Ok(None)
    }

    /// Whether sending `nonce` makes the next squeezed bytes pass the check.
    #[cfg(feature = "prover")]
    fn accepts_nonce(self, probe: &impl Probe, nonce: u32) -> bool {
        let mut check = [0u8; 4];
        let Some(out) = check.get_mut(..self.word_len()) else {
            return false;
        };
        probe.probe(Leb128::new(nonce).as_bytes(), out);
        self.accepts(out)
    }

    /// The least accepted nonce.
    ///
    /// The first [`SEQUENTIAL_NONCES`] are scanned in order, which settles a low difficulty
    /// without leaving the calling thread. The rest are scanned a block at a time, each block
    /// split over the thread pool, so every thread works near the least accepted nonce and
    /// the result does not depend on the number of threads.
    #[cfg(feature = "prover")]
    pub(crate) fn search(self, probe: &impl Probe) -> Option<u32> {
        use rayon::prelude::*;

        let last = self.last_nonce();
        let sequential_end = SEQUENTIAL_NONCES.min(last);
        if let Some(nonce) = (0..=sequential_end).find(|&nonce| self.accepts_nonce(probe, nonce)) {
            return Some(nonce);
        }
        let threads = u32::try_from(rayon::current_num_threads())
            .unwrap_or(1)
            .max(1);
        let block = NONCES_PER_THREAD.saturating_mul(threads);
        let mut end = sequential_end;
        while end < last {
            let start = end + 1;
            end = start.saturating_add(block - 1).min(last);
            let found = (start..=end)
                .into_par_iter()
                .find_first(|&nonce| self.accepts_nonce(probe, nonce));
            if found.is_some() {
                return found;
            }
        }
        None
    }
}

/// The canonical LEB128 encoding of a `u32`: seven bits per byte, least
/// significant first, the high bit set on every byte but the last.
#[cfg(feature = "prover")]
#[derive(Clone, Copy)]
pub(crate) struct Leb128 {
    bytes: [u8; 5],
    len: usize,
}

#[cfg(feature = "prover")]
impl Leb128 {
    pub(crate) fn new(mut value: u32) -> Self {
        let mut bytes = [0u8; 5];
        let mut len = 0;
        for byte in &mut bytes {
            len += 1;
            if value < 0x80 {
                *byte = value as u8;
                break;
            }
            *byte = (value as u8) | 0x80;
            value >>= 7;
        }
        Self { bytes, len }
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        self.bytes.get(..self.len).unwrap_or(&self.bytes)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::{SpongePow, MAX_POW_BITS};

    #[test]
    fn difficulty_bounds() {
        assert!(SpongePow::new(0).is_none());
        assert!(SpongePow::new(MAX_POW_BITS).is_some());
        assert!(SpongePow::new(MAX_POW_BITS + 1).is_none());
    }

    #[test]
    fn layout() {
        let layout = |bits| {
            let pow = SpongePow::new(bits).unwrap();
            (pow.last_nonce(), pow.max_nonce_len(), pow.word_len())
        };
        assert_eq!(layout(1), (255, 2, 1));
        assert_eq!(layout(8), ((1 << 15) - 1, 3, 1));
        assert_eq!(layout(10), ((1 << 17) - 1, 3, 2));
        assert_eq!(layout(25), (u32::MAX, 5, 4));
    }

    #[test]
    fn read_nonce_accepts_only_canonical_nonces_in_range() {
        let read = |bits, bytes: &[u8]| {
            let mut bytes = bytes.iter().copied();
            SpongePow::new(bits)
                .unwrap()
                .read_nonce(|| bytes.next().ok_or("short"))
        };
        assert_eq!(read(1, &[0x00]), Ok(Some(0)));
        assert_eq!(read(1, &[0x7f]), Ok(Some(127)));
        assert_eq!(read(1, &[0x80, 0x01]), Ok(Some(128)));
        assert_eq!(read(1, &[0xff, 0x01]), Ok(Some(255)));
        assert_eq!(
            read(25, &[0xff, 0xff, 0xff, 0xff, 0x0f]),
            Ok(Some(u32::MAX))
        );
        // A trailing zero byte, a nonce of 2^8 at 1 bit, a continuation past
        // the longest encoding, a nonce of 2^32 at 25 bits, and a short read.
        assert_eq!(read(1, &[0x80, 0x00]), Ok(None));
        assert_eq!(read(8, &[0x81, 0x80, 0x00]), Ok(None));
        assert_eq!(read(1, &[0x80, 0x02]), Ok(None));
        assert_eq!(read(1, &[0x80, 0x80, 0x01]), Ok(None));
        assert_eq!(read(25, &[0x80, 0x80, 0x80, 0x80, 0x10]), Ok(None));
        assert_eq!(read(8, &[0x80]), Err("short"));
    }

    #[cfg(feature = "prover")]
    mod search {
        use super::super::{Leb128, Probe, SpongePow, MAX_POW_BITS};

        #[test]
        fn leb128_is_canonical() {
            let encode = |value| Leb128::new(value).as_bytes().to_vec();
            assert_eq!(encode(0), [0]);
            assert_eq!(encode(127), [0x7f]);
            assert_eq!(encode(128), [0x80, 0x01]);
            assert_eq!(encode(300), [0xac, 0x02]);
            assert_eq!(encode(u32::MAX), [0xff, 0xff, 0xff, 0xff, 0x0f]);
            for bits in 1..=MAX_POW_BITS {
                let pow = SpongePow::new(bits).unwrap();
                let longest = Leb128::new(pow.last_nonce());
                assert_eq!(longest.as_bytes().len(), pow.max_nonce_len());
                let mut bytes = longest.as_bytes().iter().copied();
                assert_eq!(
                    pow.read_nonce(|| bytes.next().ok_or(())),
                    Ok(Some(pow.last_nonce()))
                );
            }
        }

        /// A probe that accepts exactly the nonce `target`, or none.
        struct AcceptOnly(Option<u32>);

        impl Probe for AcceptOnly {
            fn probe(&self, candidate: &[u8], out: &mut [u8]) {
                let hit = self
                    .0
                    .is_some_and(|target| candidate == Leb128::new(target).as_bytes());
                out.fill(if hit { 0 } else { 0xff });
            }
        }

        #[test]
        fn search_finds_a_lone_nonce_anywhere() {
            for bits in [1, 9] {
                let pow = SpongePow::new(bits).unwrap();
                let last = pow.last_nonce();
                for target in [None, Some(0), Some(1), Some(last / 2), Some(last)] {
                    let probe = AcceptOnly(target);
                    assert_eq!(pow.search(&probe), target);
                }
            }
        }

        /// A probe that accepts every nonce in a set.
        struct AcceptAny(Vec<u32>);

        impl Probe for AcceptAny {
            fn probe(&self, candidate: &[u8], out: &mut [u8]) {
                let hit = self
                    .0
                    .iter()
                    .any(|&target| candidate == Leb128::new(target).as_bytes());
                out.fill(if hit { 0 } else { 0xff });
            }
        }

        /// With accepted nonces spread over the space, the search returns the
        /// least one.
        #[test]
        fn search_returns_the_least_of_several_accepted_nonces() {
            for bits in [1, 9] {
                let pow = SpongePow::new(bits).unwrap();
                let last = pow.last_nonce();
                for targets in [
                    vec![last, last / 2],
                    vec![last / 2 + 1, last / 3, last],
                    vec![last - 1, 1, last],
                    vec![last - 1, last],
                    vec![last / 4, last / 4 + 1, last / 2, 3 * (last / 4)],
                ] {
                    let least = targets.iter().min().copied();
                    let probe = AcceptAny(targets);
                    assert_eq!(pow.search(&probe), least);
                }
            }
        }
    }
}
