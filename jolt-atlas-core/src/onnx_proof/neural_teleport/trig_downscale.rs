//! Shared "downscale" (right-shift) stage for the Cos/Sin trig-table lookup.
//!
//! Right-shifts the teleportation remainder via a [`RightShiftTable`] read-raf, producing
//! `downscaled = remainder >> TRIG_DOWNSCALE_BITS` — the smaller index the Cos/Sin table
//! lookup reads. Also establishes `r`'s claim (`VirtualPoly::TeleportRemainder`) for
//! `division.rs`'s `a = P·q + r` derivation.
use crate::node_view::ComputationNode;
use crate::{onnx_proof::Prover, utils::opening_access::AccOpeningAccessor};
use atlas_algebra::{field::JoltField, poly::multilinear_polynomial::EvaluateMle};
use atlas_trace::Tensor;
use atlas_transcript::Transcript;
use common::{consts::TRIG_DOWNSCALE_BITS, VirtualPoly};
pub use jolt_atlas_verifier::onnx_proof::neural_teleport::trig_downscale::*;

/// Appends the downscaled value (`remainder >> TRIG_DOWNSCALE_BITS`) as a
/// `VirtualPoly::TrigDownscaled` advice, evaluated at the node's standard reduced
/// output-opening point (mirrors `fused_rebase::cache_remainder_prove`). Returns the
/// downscaled tensor so the caller (the Cos/Sin table lookup) doesn't have to recompute it.
pub fn cache_downscaled_prove<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
    remainder: &Tensor<i32>,
) -> Tensor<i32> {
    let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
    let (r_node_output, _) = accessor.get_reduced_opening();
    let downscaled = remainder.map(|v| v >> TRIG_DOWNSCALE_BITS);
    let eval = downscaled.evaluate_mle(&r_node_output.r);
    let mut provider = accessor.into_provider(&mut prover.transcript, r_node_output);
    provider.append_advice(VirtualPoly::TrigDownscaled, eval);
    downscaled
}
