//! The adapter over LayerZero's Akita PCS (`akita-*`), taken from upstream `jolt-akita`.
//!
//! [`AkitaBackend`] wraps the Akita PCS over its fp128 field (setup, commit, open, verify), and
//! [`AkitaNativeBatching`] delegates same-point batches to Akita's native batched opening
//! protocol. `super::AkitaScheme` implements atlas's `CommitmentScheme` over them.

// In the jolt-verifier runtime closure: stricter panic and unsafe discipline
// than the workspace lints (specs/verifier-closure-lints.md).
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

mod adapters;
pub mod configs;
pub mod family_contract;
mod native_batching;
mod one_hot_family;
mod schedule_artifacts;
pub mod schedule_registry;
pub mod schedules;
mod scheme;
mod shape_guard;
mod trace_onehot;

pub use akita_pcs::AkitaError;

pub use adapters::{
    AkitaBackendFlavor, AkitaBatchProof, AkitaCommitment, AkitaField, AkitaProverHint,
    AkitaProverSetup, AkitaSetupParams, AkitaVerifierSetup, AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256,
};
pub use configs::AkitaChunkProfile;
pub use native_batching::{
    AkitaNativeBatchPolynomials, AkitaNativeBatchStatement, AkitaNativeBatching,
};
pub use schedule_artifacts::{AkitaOneHotScheduleArtifact, AkitaScheduleArtifacts};
pub use schedule_registry::{DenseGroupLayout, GroupedScheduleParams};
pub use scheme::AkitaBackend;
pub use trace_onehot::{no_selected_row, TraceOneHotColumn, TraceOneHotRows};

/// Jolt↔Akita basis-order bridging, exposed so benchmarks measuring the raw
/// backend use the exact transform the adapter uses.
#[doc(hidden)]
pub use adapters::{jolt_to_akita_evals, jolt_to_akita_index, reverse_point};
