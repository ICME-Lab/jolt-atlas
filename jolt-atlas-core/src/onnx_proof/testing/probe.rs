//! Probe mode for the soundness sweeps.
//!
//! The prover carries `#[cfg(test)]` self-checks that recompute a sumcheck
//! claim from the witness and `assert_eq!` it. Under an honest run they are
//! free invariants; under a deliberately tampered witness they would panic
//! inside the prover and hide whether the *verifier* catches the tamper. In
//! probe mode those checks record a divergence instead of panicking, so the
//! tampered proof reaches the verifier, which is the oracle under test.
//!
//! State is process-global (the self-checks run inside rayon closures, so a
//! thread-local would miss them). nextest runs each test in its own process;
//! under plain `cargo test` the sweeps are `#[serial]`.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

static ACTIVE: AtomicBool = AtomicBool::new(false);
static EXPECTING_PANIC: AtomicBool = AtomicBool::new(false);
static DIVERGENCES: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Probe mode is on while this guard lives: prover self-checks record
/// instead of panicking, panics inside [`expecting_panics`] are not printed
/// (tampered proves are expected to panic sometimes), and the forgery target
/// is cleared on entry and exit.
pub(crate) struct ProbeGuard(());

impl ProbeGuard {
    pub(crate) fn enter() -> Self {
        assert!(
            !ACTIVE.swap(true, Ordering::SeqCst),
            "probe mode is already active (nested or concurrent sweep)"
        );
        DIVERGENCES.lock().unwrap().clear();
        joltworks::poly::opening_proof::forgery::set(None);
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
        joltworks::poly::opening_proof::forgery::set(None);
        ACTIVE.store(false, Ordering::SeqCst);
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

/// Whether a sweep is running.
pub(crate) fn active() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

/// Record a prover self-check that would have fired.
pub(crate) fn note_divergence(site: &str) {
    DIVERGENCES.lock().unwrap().push(site.to_string());
}

/// Drain the recorded divergences.
pub(crate) fn take_divergences() -> Vec<String> {
    std::mem::take(&mut *DIVERGENCES.lock().unwrap())
}

/// `assert!` outside probe mode; a recorded divergence inside it.
macro_rules! probe_assert {
    ($cond:expr $(, $($arg:tt)+)?) => {
        if $crate::onnx_proof::testing::probe::active() {
            if !($cond) {
                $crate::onnx_proof::testing::probe::note_divergence(concat!(file!(), ":", line!()));
            }
        } else {
            assert!($cond $(, $($arg)+)?);
        }
    };
}

/// `assert_eq!` outside probe mode; a recorded divergence inside it.
macro_rules! probe_assert_eq {
    ($left:expr, $right:expr $(, $($arg:tt)+)?) => {
        if $crate::onnx_proof::testing::probe::active() {
            if $left != $right {
                $crate::onnx_proof::testing::probe::note_divergence(concat!(file!(), ":", line!()));
            }
        } else {
            assert_eq!($left, $right $(, $($arg)+)?);
        }
    };
}

pub(crate) use {probe_assert, probe_assert_eq};
