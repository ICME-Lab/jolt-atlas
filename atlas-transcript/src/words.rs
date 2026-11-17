//! The deployed proof format's word encodings, for protocols that absorb fixed-layout words
//! rather than the transcript's tagged values: the Akita adapter's statement and setup binding
//! and the sumcheck engine's input claims and round labels.
//!
//! Every encoding is one [`Transcript::append_bytes`] call per value:
//!
//! - [`Label`]: the label zero-padded to a 32-byte word;
//! - [`LabelWithCount`]: the label zero-padded to 24 bytes, then the count as a big-endian `u64`;
//! - [`U64Word`]: 24 zero bytes, then the value as a big-endian `u64`;
//! - a field element: its canonical bytes, big-endian.
//!
//! The layouts are part of the proof format: changing one changes every proof.

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

use crate::Transcript;
use atlas_algebra::field::CanonicalBytes;

/// A value with a word encoding.
pub trait WordEncoding {
    /// Absorbs this value's encoding.
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T);
}

/// The word-encoding operations on any [`Transcript`].
pub trait WordTranscript: Transcript + Sized {
    /// Absorbs `value`'s encoding.
    fn append<A: WordEncoding + ?Sized>(&mut self, value: &A) {
        value.append_to_transcript(self);
    }

    /// Absorbs [`Label`]`(label)`, then `value`.
    fn append_labeled<A: WordEncoding + ?Sized>(&mut self, label: &'static [u8], value: &A) {
        self.append(&Label(label));
        self.append(value);
    }

    /// Absorbs [`LabelWithCount`]`(label, values.len())`, then each value in order.
    fn append_values<A: WordEncoding>(&mut self, label: &'static [u8], values: &[A]) {
        self.append(&LabelWithCount(label, values.len() as u64));
        for value in values {
            self.append(value);
        }
    }
}

impl<T: Transcript> WordTranscript for T {}

/// A field element (or any canonically encoded value): its canonical bytes, big-endian.
impl<F: CanonicalBytes> WordEncoding for F {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        let mut buf = vec![0u8; F::NUM_BYTES];
        self.to_bytes_le(&mut buf);
        buf.reverse();
        transcript.append_bytes(&buf);
    }
}

/// A label zero-padded to one 32-byte word.
pub struct Label(pub &'static [u8]);

impl WordEncoding for Label {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        assert!(
            self.0.len() <= 32,
            "label {:?} exceeds 32 bytes",
            core::str::from_utf8(self.0)
        );
        let mut padded = [0u8; 32];
        let (head, _) = padded.split_at_mut(self.0.len());
        head.copy_from_slice(self.0);
        transcript.append_bytes(&padded);
    }
}

/// A label (zero-padded to 24 bytes) and a count (big-endian `u64`) in one 32-byte word.
pub struct LabelWithCount(pub &'static [u8], pub u64);

impl WordEncoding for LabelWithCount {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        assert!(
            self.0.len() <= 24,
            "label {:?} exceeds 24 bytes",
            core::str::from_utf8(self.0)
        );
        let mut packed = [0u8; 32];
        let (head, _) = packed.split_at_mut(self.0.len());
        head.copy_from_slice(self.0);
        let (_, count) = packed.split_at_mut(24);
        count.copy_from_slice(&self.1.to_be_bytes());
        transcript.append_bytes(&packed);
    }
}

/// A `u64` left-padded to one 32-byte word: 24 zero bytes, then the value big-endian.
pub struct U64Word(pub u64);

impl WordEncoding for U64Word {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        let mut packed = [0u8; 32];
        let (_, value) = packed.split_at_mut(24);
        value.copy_from_slice(&self.0.to_be_bytes());
        transcript.append_bytes(&packed);
    }
}
