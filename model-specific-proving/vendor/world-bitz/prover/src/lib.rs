//! The BitZ prover.

pub mod fold;
pub mod prove;
mod reduce;
pub mod setup;

pub use fold::SendError;
pub use prove::{ProveError, VirtualWitness};
pub use reduce::gkr_reduce;
pub use setup::BitZProver;
