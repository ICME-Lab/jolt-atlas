//! Shared "downscale" (right-shift) stage for the Cos/Sin trig-table lookup.
//!
//! Right-shifts the teleportation remainder via a [`RightShiftTable`] read-raf, producing
//! `downscaled = remainder >> TRIG_DOWNSCALE_BITS` — the smaller index the Cos/Sin table
//! lookup reads. Also establishes `r`'s claim (`VirtualPoly::TeleportRemainder`) for
//! `division.rs`'s `a = P·q + r` derivation.
use crate::node_view::ComputationNode;
use crate::onnx_proof::op_lookups::LookupWitness;
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::utils::lookup_bits::LookupBits;
use atlas_trace::Tensor;
use atlas_trace::Trace;
use common::consts::XLEN;
pub use jolt_atlas_verifier::onnx_proof::neural_teleport::trig_downscale::*;

impl LookupWitness for TrigDownscaleOperands {
    /// The teleportation remainder the trace captured.
    fn witness(&self, node: &ComputationNode, trace: &Trace) -> Tensor<i64> {
        TraceView::teleport_advice(trace, node)
            .remainder
            .map(|v| v as i64)
    }

    fn lookup_bits(&self, witness: &Tensor<i64>) -> Vec<LookupBits> {
        witness
            .iter()
            .map(|&v| LookupBits::new(v as u64, XLEN))
            .collect()
    }
}
