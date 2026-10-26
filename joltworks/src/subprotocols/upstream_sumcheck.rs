//! The seam onto upstream `jolt-sumcheck`'s engine.
//!
//! Upstream drives a batched sumcheck through [`jolt_sumcheck::prove_batch`] and verifies it with
//! [`jolt_sumcheck::SumcheckVerifier::verify_compressed`], both over the facade
//! [`jolt_transcript::Transcript`]. This module adapts the atlas side to that engine:
//!
//! - [`FacadeTranscript`] presents one of our split transcripts as the facade for the duration of
//!   an engine call. Its challenges are our 125-bit challenges (as field elements), each preceded
//!   by the round's proof of work, so stage 3's per-round grind survives the engine swap.
//! - [`InstanceMember`] runs one of our [`SumcheckInstanceProver`]s as an upstream batch member,
//!   handing it its round challenges back in the optimized [`JoltField::Challenge`] form.
//! - [`InstanceRounds`] schedules members sequentially or, for batches of independent instances,
//!   in parallel.

#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;
use crate::{
    field::JoltField, poly::unipoly::CompressedUniPoly,
    subprotocols::sumcheck_prover::SumcheckInstanceProver, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_sumcheck::{MemberFinish, MemberRound, ProveRounds, RoundScheduler, SumcheckError};
use rayon::prelude::*;
use std::{marker::PhantomData, ptr::NonNull};

/// One of our transcripts, presented as upstream's facade transcript while an engine call runs.
///
/// The facade trait requires `'static` and `Default`, which a borrow cannot provide, so the view
/// holds a raw pointer to the borrowed transcript and exists only inside [`FacadeTranscript::scope`]:
/// the caller never sees it by value, so it cannot outlive the borrow. A `Default` view is
/// detached; the engine never constructs one (it only receives `&mut` views), and using one
/// panics.
pub struct FacadeTranscript<T: Transcript, F: JoltField> {
    inner: Option<NonNull<T>>,
    pow_bits: u32,
    pow_failed: bool,
    _field: PhantomData<fn() -> F>,
}

// SAFETY: the view only forwards to `T`, which is `Send + Sync`; the pointer is never shared
// beyond the `&mut` borrow it was created from.
unsafe impl<T: Transcript, F: JoltField> Send for FacadeTranscript<T, F> {}
// SAFETY: as above.
unsafe impl<T: Transcript, F: JoltField> Sync for FacadeTranscript<T, F> {}

impl<T: Transcript, F: JoltField> Default for FacadeTranscript<T, F> {
    fn default() -> Self {
        Self {
            inner: None,
            pow_bits: 0,
            pow_failed: false,
            _field: PhantomData,
        }
    }
}

impl<T: Transcript, F: JoltField> FacadeTranscript<T, F> {
    /// Run `body` with `transcript` presented as the facade; every challenge it draws is preceded
    /// by a proof of work of difficulty `pow_bits`. A failed proof-of-work check (verifier side)
    /// is reported after `body` returns.
    pub fn scope<R>(
        transcript: &mut T,
        pow_bits: u32,
        body: impl FnOnce(&mut Self) -> R,
    ) -> (R, Result<(), ProofVerifyError>) {
        let mut view = Self {
            inner: Some(NonNull::from(transcript)),
            pow_bits,
            pow_failed: false,
            _field: PhantomData,
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
        // SAFETY: `inner` is set only by `scope`, from a `&mut T` that outlives the view, and
        // the view is reachable only through the `&mut Self` handed to `scope`'s body.
        unsafe { &mut *ptr.as_ptr() }
    }
}

impl<T: Transcript, F: JoltField> jolt_transcript::Transcript for FacadeTranscript<T, F> {
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
        self.inner().challenge_scalar_optimized::<F>().into()
    }

    fn state(&self) -> [u8; 32] {
        unreachable!("the sumcheck engine does not read the transcript state")
    }
}

/// Map an upstream engine error into the prover's panic (the engine's errors are internal
/// inconsistencies of the prover: a round polynomial that fails its own running-claim check).
pub(crate) fn expect_engine<F: JoltField, R>(res: Result<R, SumcheckError<F>>) -> R {
    res.unwrap_or_else(|e| panic!("sumcheck prover: {e:?}"))
}

/// The engine error for a round message whose `s(0) + s(1)` is `actual` instead of
/// `previous_claim`.
pub fn round_check_failed<F: JoltField>(
    round: usize,
    previous_claim: F,
) -> impl FnOnce(F) -> SumcheckError<F> {
    move |actual| SumcheckError::RoundCheckFailed {
        round,
        expected: previous_claim,
        actual,
    }
}

/// One of our sumcheck instances as an upstream batch member: its own [`ProveRounds`] when it
/// has one, otherwise its `compute_message`/`ingest_challenge`/`finalize`.
pub struct InstanceMember<'a, F: JoltField, T: Transcript> {
    instance: &'a mut dyn SumcheckInstanceProver<F, T>,
}

impl<'a, F: JoltField, T: Transcript> InstanceMember<'a, F, T> {
    pub fn new(instance: &'a mut dyn SumcheckInstanceProver<F, T>) -> Self {
        Self { instance }
    }
}

impl<F: JoltField, T: Transcript> ProveRounds<F> for InstanceMember<'_, F, T> {
    fn num_rounds(&self) -> usize {
        self.instance.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(rounds) = self.instance.rounds() {
            return rounds.prove_round(bind, round, previous_claim);
        }
        if let Some(r) = bind {
            self.instance
                .ingest_challenge(F::challenge_from_field(r), round - 1);
        }
        let message = self.instance.compute_message(round, previous_claim);
        Ok(UnivariatePoly::new(message.coeffs))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        if let Some(rounds) = self.instance.rounds() {
            return rounds.finish_rounds(bind);
        }
        let last = self.instance.num_rounds() - 1;
        self.instance
            .ingest_challenge(F::challenge_from_field(bind), last);
        self.instance.finalize();
        Ok(())
    }
}

/// Round scheduling for our instances: in declaration order, or in parallel across members for
/// batches of mutually independent instances (each member's own parallelism stays on; rayon's
/// work stealing absorbs the nesting).
///
/// Crate-private: parallel scheduling relies on every member being an [`InstanceMember`], whose
/// instance is `Send + Sync` (see [`SendRound`]), which only this crate's batched sumcheck
/// guarantees.
pub(crate) struct InstanceRounds {
    parallel: bool,
    round: usize,
}

impl InstanceRounds {
    pub(crate) fn new(parallel: bool) -> Self {
        Self { parallel, round: 0 }
    }
}

/// A round handle sent to a rayon worker.
struct SendRound<'s, T>(&'s mut T);

// SAFETY: `InstanceRounds` only schedules `InstanceMember`s, whose instances are
// `SumcheckInstanceProver: Send + Sync`; the remaining handle fields are field elements and
// indices. Each handle is visited by exactly one worker.
unsafe impl<T> Send for SendRound<'_, T> {}

impl<F: JoltField> RoundScheduler<F> for InstanceRounds {
    fn batch_prove_round(
        &mut self,
        work: &mut [MemberRound<'_, F>],
    ) -> Result<(), SumcheckError<F>> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage(format!("Sumcheck round {}", self.round).as_str());
        self.round += 1;
        if self.parallel {
            work.iter_mut()
                .map(SendRound)
                .collect::<Vec<_>>()
                .into_par_iter()
                .try_for_each(|SendRound(item)| item.run())
        } else {
            work.iter_mut().try_for_each(|item| item.run())
        }
    }

    fn batch_finish_rounds(
        &mut self,
        finishes: &mut [MemberFinish<'_, F>],
    ) -> Result<(), SumcheckError<F>> {
        if self.parallel {
            finishes
                .iter_mut()
                .map(SendRound)
                .collect::<Vec<_>>()
                .into_par_iter()
                .try_for_each(|SendRound(item)| item.run())
        } else {
            finishes.iter_mut().try_for_each(|item| item.run())
        }
    }
}

/// Our wire form of a compressed round polynomial, from upstream's.
pub fn compressed_from_upstream<F: JoltField>(poly: &CompressedPoly<F>) -> CompressedUniPoly<F> {
    CompressedUniPoly {
        coeffs_except_linear_term: poly.coeffs_except_linear_term().to_vec(),
    }
}

/// Upstream's compressed round polynomial, from our wire form.
pub fn compressed_to_upstream<F: JoltField>(poly: &CompressedUniPoly<F>) -> CompressedPoly<F> {
    CompressedPoly::new(poly.coeffs_except_linear_term.clone())
}
