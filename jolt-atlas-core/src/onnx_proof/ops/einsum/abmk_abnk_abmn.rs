//! `abmk,abnk->abmn` layout for the einsum [`dot`](super::dot) engine: a
//! batched matmul (RHS contraction-last) whose retained batch axes are packed
//! in front of both operands and the output. The rank-3 form `amk,ank->amn`
//! is the identical layout with a batch pack of `(1, a)` — the flattened
//! operand memory is the same, so both normalize here.
//!
//! The batch pack is bound first through a high eq polynomial
//! ([`EqSchedule::High`]); the contraction axis `k` follows.

pub use jolt_atlas_verifier::onnx_proof::ops::einsum::abmk_abnk_abmn::*;
