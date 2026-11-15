//! The bridge onto the `akita-*` crates' transcript.
//!
//! Akita's batch opening is generic over `jolt_transcript`'s facade `Transcript`, which the
//! external `akita-*` crates bind by name. [`FacadeTranscript`] presents one of our transcripts
//! as that facade for the duration of an Akita call.

use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_transcript::Transcript;
use std::ptr::NonNull;

/// One of our transcripts, presented as the `akita-*` crates' facade transcript while an Akita
/// call runs.
///
/// The facade trait requires `'static` and `Default`, which a borrow cannot provide, so the view
/// holds a raw pointer to the borrowed transcript and exists only inside
/// [`FacadeTranscript::scope_with`]: the caller never sees it by value, so it cannot outlive the
/// borrow. A `Default` view is detached; Akita never constructs one (it only receives `&mut`
/// views), and using one panics. `draw` turns the underlying transcript's next challenge into
/// the facade's `F`.
pub(crate) struct FacadeTranscript<T: Transcript, F: Send + Sync + 'static> {
    inner: Option<NonNull<T>>,
    draw: fn(&mut T) -> F,
    pow_bits: u32,
    pow_failed: bool,
}

// SAFETY: the view only forwards to `T`, which is `Send + Sync`; the pointer is never shared
// beyond the `&mut` borrow it was created from.
unsafe impl<T: Transcript, F: Send + Sync + 'static> Send for FacadeTranscript<T, F> {}
// SAFETY: as above.
unsafe impl<T: Transcript, F: Send + Sync + 'static> Sync for FacadeTranscript<T, F> {}

fn detached<T, F>(_: &mut T) -> F {
    panic!("a detached facade transcript cannot be used")
}

impl<T: Transcript, F: Send + Sync + 'static> Default for FacadeTranscript<T, F> {
    fn default() -> Self {
        Self {
            inner: None,
            draw: detached::<T, F>,
            pow_bits: 0,
            pow_failed: false,
        }
    }
}

impl<T: Transcript, F: Send + Sync + 'static> FacadeTranscript<T, F> {
    /// Run `body` with `transcript` presented as the facade, drawing challenges with `draw`;
    /// every challenge is preceded by a proof of work of difficulty `pow_bits`. A failed
    /// proof-of-work check (verifier side) is reported after `body` returns.
    pub(crate) fn scope_with<R>(
        transcript: &mut T,
        pow_bits: u32,
        draw: fn(&mut T) -> F,
        body: impl FnOnce(&mut Self) -> R,
    ) -> (R, Result<(), ProofVerifyError>) {
        let mut view = Self {
            inner: Some(NonNull::from(transcript)),
            draw,
            pow_bits,
            pow_failed: false,
        };
        let out = body(&mut view);
        let pow = if view.pow_failed {
            Err(ProofVerifyError::ProofOfWork)
        } else {
            Ok(())
        };
        (out, pow)
    }

    fn inner(&mut self) -> &mut T {
        let ptr = self
            .inner
            .expect("a detached facade transcript cannot be used");
        // SAFETY: `inner` is set only by `scope_with`, from a `&mut T` that outlives the view,
        // and the view is reachable only through the `&mut Self` handed to its body.
        unsafe { &mut *ptr.as_ptr() }
    }
}

impl<T: Transcript, F: Send + Sync + 'static> jolt_transcript::Transcript
    for FacadeTranscript<T, F>
{
    type Challenge = F;

    fn new(_label: &'static [u8]) -> Self {
        Self::default()
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.inner().append_bytes(bytes);
    }

    fn challenge(&mut self) -> F {
        let bits = self.pow_bits;
        if self.inner().pow(bits).is_err() {
            self.pow_failed = true;
        }
        (self.draw)(self.inner())
    }

    fn state(&self) -> [u8; 32] {
        unreachable!("Akita, driven through the facade, does not read the transcript state")
    }
}
