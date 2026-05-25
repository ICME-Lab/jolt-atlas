//! Layer module guide.
//!
//! Read in this order:
//! 1. `prover.rs` for the complete proving entry point.
//! 2. `verifier.rs` for the matching verification entry point.
//! 3. `iop.rs` for the layer equations and reverse claim flow.
//! 4. `commitments.rs` and `openings.rs` for the PCS-facing code.
//! 5. `witness.rs` and `tensors.rs` only when you need materialization or op wiring details.

mod claims;
mod commitments;
mod iop;
mod openings;
mod prover;
mod tensors;
#[cfg(test)]
mod tests;
mod types;
mod verifier;
mod witness;

pub use commitments::{
    HiddenStateCommitments, LayerPolynomialMap, absorb_layer_commitments, commit_layer_polynomials,
    commit_layer_polynomials_streaming_onehot,
};
pub use prover::prove_layer;
pub use tensors::LayerTensorIds;
pub use types::{LayerClaims, LayerProof, LayerShape, LayerWeights};
pub use verifier::verify_layer;
pub use witness::LayerWitness;
