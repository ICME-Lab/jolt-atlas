//! Division helpers for neural teleportation (Cos/Sin's `a mod P` domain reduction).
//!
//! `a = P·q + r` holds exactly, elementwise, so as multilinear polynomials
//! `ã(X) = P·q̃(X) + r̃(X)` holds identically everywhere, not just on the boolean hypercube.
//! Once `q` and `r` each have an evaluation claim at the same point, deriving `a`'s claim is
//! pure field arithmetic — no sumcheck is needed to enforce the identity.

pub use jolt_atlas_verifier::onnx_proof::neural_teleport::division::*;
