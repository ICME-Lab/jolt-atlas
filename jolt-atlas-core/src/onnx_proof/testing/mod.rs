//! The crate's test suite, in one place.
//!
//! * [`harness`]: one prove → verify → verify-from-bytes path that every
//!   end-to-end test goes through, generic over the PCS.
//! * [`op_kind`] + [`op_surface`]: the exhaustive single-op table. A new
//!   `atlas_trace::Op` variant fails to compile here until a case exists.
//! * [`range_buckets`]: what the packed range buckets hold and commit for a model.
//! * [`zoo`]: the checked-in model zoo (tiny GPT-2 / Qwen2 / BERT stand-ins,
//!   op gauntlet) on every run; the real downloads behind `#[ignore]`.
//! * [`soundness`] (+ [`mutate`], [`recording`], [`probe`], [`models`]): generic
//!   attacks on the proof system: mutation of every value the verifier reads
//!   (the sites come from a recorded read log), witness tamper, opening forgery,
//!   binding tests, and pinned `known_gap_*` tests.
//! * [`digests`]: cross-stage pins of the zoo's proof bytes and final
//!   transcript state, so a coordinated prover+verifier change is visible.
//! * [`schedules`]: pins of the zoo's verification schedules, so a change to what the
//!   verifier checks for a model, or to its order, is visible.
//! * [`rounds`]: native sumcheck members driven round by round at eq points
//!   no end-to-end run reaches.
//! * [`relations`]: relation-defined ops' op binding and the claim flows
//!   they take part in.
//! * [`model_builder`]: programmatic model construction.
//!
//! This module is a descendant of `onnx_proof` so tests can reach the
//! prover and verifier helpers without widening any visibility.

#[cfg(feature = "akita")]
mod akita;
mod digests;
pub(crate) mod harness;
pub(crate) mod model_builder;
mod models;
mod mutate;
pub(crate) mod op_kind;
pub(crate) mod op_surface;
pub(crate) mod probe;
mod range_buckets;
mod recording;
pub(crate) mod relations;
mod rounds;
mod schedules;
mod soundness;
mod zoo;

pub(crate) use harness::{prove_verify_roundtrip, RoundtripOptions};
pub(crate) use model_builder::ModelBuilder;
