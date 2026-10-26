//! The crate's test suite, in one place.
//!
//! * [`harness`]: one prove → verify → serialize → deserialize → verify path
//!   that every end-to-end test goes through, generic over the PCS.
//! * [`op_kind`] + [`op_surface`]: the exhaustive single-op table. A new
//!   `atlas_trace::Op` variant fails to compile here until a case exists.
//! * [`op_surface_dory`]: a shape-class subset of the table under Dory.
//! * [`zoo`]: the checked-in model zoo (tiny GPT-2 / Qwen2 / BERT stand-ins,
//!   op gauntlet) on every run; the real downloads behind `#[ignore]`.
//! * [`soundness`] (+ [`mutate`], [`probe`], [`models`]): generic attacks on
//!   the proof system: proof-field mutation, witness tamper, opening forgery,
//!   binding tests, and pinned `known_gap_*` tests.
//! * [`digests`]: cross-stage pins of the zoo's proof bytes and final
//!   transcript state, so a coordinated prover+verifier change is visible.
//! * [`rounds`]: native sumcheck members driven round by round at eq points
//!   no end-to-end run reaches.
//! * [`relations`]: relation-defined ops' op binding and the claim flows
//!   they take part in.
//! * [`model_builder`]: programmatic model construction.
//!
//! This module is a descendant of `onnx_proof` so tests can reach the
//! prover and verifier helpers without widening any visibility. Proof fields
//! are reached only through `jolt-atlas-verifier`'s `test-util` accessors.

mod digests;
pub(crate) mod harness;
pub(crate) mod model_builder;
mod models;
mod mutate;
pub(crate) mod op_kind;
pub(crate) mod op_surface;
mod op_surface_dory;
pub(crate) mod probe;
pub(crate) mod relations;
mod rounds;
mod soundness;
mod zoo;

pub(crate) use harness::{prove_verify_roundtrip, RoundtripOptions};
pub(crate) use model_builder::ModelBuilder;
