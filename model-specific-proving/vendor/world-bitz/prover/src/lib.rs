//! The BitZ prover.

pub mod fold;
pub mod prove;
mod reduce;
pub mod setup;

pub use fold::SendError;
pub use prove::{ProveError, VirtualWitness};
pub use reduce::gkr_reduce;
pub use setup::BitZProver;

#[cfg(feature = "gkr-backend")]
pub use gkr::{set_dense_backend, set_tree_backend, DenseRounds};
