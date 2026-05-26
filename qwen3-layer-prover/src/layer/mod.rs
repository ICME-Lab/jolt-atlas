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
mod polys;
mod prover;
mod tensors;
#[cfg(all(test, feature = "layer-pcs-tests"))]
mod tests;
mod types;
mod verifier;
mod witness;

pub use commitments::{
    HiddenStateCommitments, LayerPolySet, absorb_layer_commitments, attach_layer_ra_commitments,
    commit_layer_polynomials, commit_layer_polynomials_streaming_onehot,
};
pub use polys::LayerPolys;
pub use prover::{
    prove_and_verify_layer_iop_only_from_witness, prove_layer, prove_layer_iop_only_from_witness,
    prove_layer_with_committed_polys,
};
pub use tensors::LayerTensorIds;
pub use types::{LayerClaims, LayerProof, LayerShape, LayerWeights};
pub use verifier::verify_layer;
pub use witness::LayerWitness;
