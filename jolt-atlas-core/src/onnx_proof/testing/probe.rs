//! Probe mode for the soundness sweeps.
//!
//! The prover carries `#[cfg(test)]` self-checks that recompute a sumcheck
//! claim from the witness and `assert_eq!` it. Under an honest run they are
//! free invariants; under a deliberately tampered witness they would panic
//! inside the prover and hide whether the *verifier* catches the tamper. In
//! probe mode those checks record a divergence instead of panicking, so the
//! tampered proof reaches the verifier, which is the oracle under test.
//!
//! The probe state and the `probe_assert!` macros live in
//! [`jolt_atlas_verifier::probe`], so self-checks on either side of the crate
//! boundary share it; this module owns the guard. State is process-global;
//! nextest runs each test in its own process; under plain `cargo test` the
//! sweeps are `#[serial]`.

use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) use jolt_atlas_verifier::probe::take_divergences;
pub(crate) use jolt_atlas_verifier::probe_assert_eq;

static EXPECTING_PANIC: AtomicBool = AtomicBool::new(false);

/// Probe mode is on while this guard lives: prover self-checks record
/// instead of panicking, panics inside [`expecting_panics`] are not printed
/// (tampered proves are expected to panic sometimes), and the forgery target
/// is cleared on entry and exit.
pub(crate) struct ProbeGuard(());

impl ProbeGuard {
    pub(crate) fn enter() -> Self {
        assert!(
            !jolt_atlas_verifier::probe::set_active(true),
            "probe mode is already active (nested or concurrent sweep)"
        );
        take_divergences();
        atlas_sumcheck::opening_proof::forgery::set(None);
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !EXPECTING_PANIC.load(Ordering::SeqCst) {
                default_hook(info);
            }
        }));
        ProbeGuard(())
    }
}

impl Drop for ProbeGuard {
    fn drop(&mut self) {
        // `take_hook` itself panics on a panicking thread; leave the hook in
        // place then (the process is failing anyway).
        if !std::thread::panicking() {
            drop(std::panic::take_hook());
        }
        atlas_sumcheck::opening_proof::forgery::set(None);
        jolt_atlas_verifier::probe::set_active(false);
    }
}

/// Run `f` with panic messages suppressed: the caller catches them and
/// classifies them (a prover refusing a tampered witness, a verifier crash).
pub(crate) fn expecting_panics<R>(f: impl FnOnce() -> R) -> R {
    EXPECTING_PANIC.store(true, Ordering::SeqCst);
    let r = f();
    EXPECTING_PANIC.store(false, Ordering::SeqCst);
    r
}
