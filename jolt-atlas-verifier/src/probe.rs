//! Probe-mode state for the prover self-checks (`test-util` only).
//!
//! The prover carries self-checks that recompute a claim from the witness and assert it. Under
//! an honest run they are free invariants; under the soundness sweeps' deliberately tampered
//! witness they would panic inside the prover and hide whether the *verifier* catches the
//! tamper. While probe mode is on, [`probe_assert!`](crate::probe_assert) and
//! [`probe_assert_eq!`](crate::probe_assert_eq) record a divergence instead of panicking.
//! `jolt-atlas-core`'s test suite owns the guard that switches it on.
//!
//! State is process-global: the self-checks run inside rayon closures, so a thread-local would
//! miss them.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

static ACTIVE: AtomicBool = AtomicBool::new(false);
static DIVERGENCES: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Switch probe mode; returns the previous state.
pub fn set_active(on: bool) -> bool {
    ACTIVE.swap(on, Ordering::SeqCst)
}

/// Whether probe mode is on.
pub fn active() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

/// Record a prover self-check that would have fired.
pub fn note_divergence(site: &str) {
    DIVERGENCES.lock().unwrap().push(site.to_string());
}

/// Drain the recorded divergences.
pub fn take_divergences() -> Vec<String> {
    std::mem::take(&mut *DIVERGENCES.lock().unwrap())
}

/// `assert!` outside probe mode; a recorded divergence inside it.
#[macro_export]
macro_rules! probe_assert {
    ($cond:expr $(, $($arg:tt)+)?) => {
        if $crate::probe::active() {
            if !($cond) {
                $crate::probe::note_divergence(concat!(file!(), ":", line!()));
            }
        } else {
            assert!($cond $(, $($arg)+)?);
        }
    };
}

/// `assert_eq!` outside probe mode; a recorded divergence inside it.
#[macro_export]
macro_rules! probe_assert_eq {
    ($left:expr, $right:expr $(, $($arg:tt)+)?) => {
        if $crate::probe::active() {
            if $left != $right {
                $crate::probe::note_divergence(concat!(file!(), ":", line!()));
            }
        } else {
            assert_eq!($left, $right $(, $($arg)+)?);
        }
    };
}
