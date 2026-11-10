//! The saturating-clamp accumulation of `Add`/`Sub`/`Sum`/`Einsum`/`Mul`/`Square`/`Cube`/
//! `MeanOfSquares`:
//!
//! ```text
//! output(x) = SatClamp(acc(x))
//! ```
//!
//! A clamped node opens its **pre-clamp i64 accumulation** `acc` (`VirtualPoly::ClampAcc`) at
//! its output point through the generic [`OpLookupProvider`](super::op_lookups::OpLookupProvider),
//! parameterized by [`SaturatingAccClampOperands`], which appends that witness claim and nothing
//! else. The clamp itself is proven in the node's packed clamp bucket after the node loop
//! ([`clamp_split`](super::clamp_split)), by the tie `acc = out` for a model output whose public
//! value is unsaturated, or in the clear for a scalar node. The caller ties `acc` back to the
//! operands (`acc = left ± right` for `Add`/`Sub`, the axis reduction for `Sum`, the fused
//! rescale for the others).
//!
//! The prover recovers `acc` with [`clamp_intermediate`]: from the trace's fused intermediates,
//! or by re-executing the saturating binop or axis sum.

use crate::node_view::ComputationNode;
use crate::onnx_proof::op_lookups::{LookupOperandsTrait, LookupWitness};
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::utils::lookup_bits::LookupBits;
use atlas_trace::{op::sum_axes_i64, trace::LayerData, Op, Tensor, Trace};
use common::parallel::par_enabled;
pub use jolt_atlas_verifier::onnx_proof::clamp_lookups::*;
use rayon::prelude::*;

/// Re-execute the node's saturating accumulation and return the **padded** i64
/// intermediate (pre-clamp) tensor — the `raf`/lookup-index polynomial.
///
/// - `Add`/`Sub`: element-wise `left ± right` (via [`sat_binop_intermediate`]).
/// - `Sum`: the un-clamped sum over the reduced axes (via [`sum_axes_i64`]).
///
/// The padding (to the next power of two) matches the node-output MLE domain, so
/// the accumulation and output polynomials share the same `log_T`.
/// Pre-clamp `i64` accumulation of a binary saturating Add/Sub — the clamp
/// lookup index the prover re-derives.
fn sat_binop_intermediate(operator: &Op, lhs: &Tensor<i32>, rhs: &Tensor<i32>) -> Tensor<i64> {
    let combine: fn(i64, i64) -> i64 = match operator {
        Op::Add(_) => |a, b| a + b,
        Op::Sub(_) => |a, b| a - b,
        other => panic!("sat_binop_intermediate: expected Add or Sub, got {other}"),
    };
    lhs.zip_map(rhs, |a, b| combine(a as i64, b as i64))
        .expect("sat_binop_intermediate: operand dims must match")
}

/// The node's pre-clamp `i64` accumulation, re-executed from its operands in the trace.
pub fn clamp_intermediate(node: &ComputationNode, trace: &Trace) -> Tensor<i64> {
    // Fused rescale ops: the trace already holds the pre-clamp quotient.
    if let Some(cached) = trace.fused_intermediates(node.idx) {
        return cached.quotient.padded_pow2();
    }
    let LayerData { operands, .. } = TraceView::layer_data(trace, node);
    let raw = match &node.operator {
        Op::Add(_) | Op::Sub(_) => {
            let [left, right] = operands[..] else {
                panic!(
                    "clamp lookup (Add/Sub) expects two operands, got {}",
                    operands.len()
                )
            };
            sat_binop_intermediate(&node.operator, left, right)
        }
        Op::Sum(s) => {
            let [operand] = operands[..] else {
                panic!(
                    "clamp lookup (Sum) expects one operand, got {}",
                    operands.len()
                )
            };
            sum_axes_i64(operand, &s.axes)
        }
        // Einsum / Mul / Square / Cube / MeanOfSquares: the pre-clamp value
        // is always captured in the trace (handled by the cache hit above).
        other => panic!(
            "clamp lookup: unsupported operator {other} (fused ops must carry trace intermediates)"
        ),
    };
    raw.padded_pow2()
}

/// Clamp lookup indices: each i64 accumulation value's low `bits` bits (its
/// two's-complement encoding at the node's clamp width — see
/// `crate::node_view`), the address into
/// saturation table.
pub fn clamp_lookup_bits(intermediate: &Tensor<i64>, bits: usize) -> Vec<LookupBits> {
    intermediate
        .data()
        .par_iter()
        .with_min_len(par_enabled())
        .map(|&v| LookupBits::new(v as u64, bits))
        .collect()
}

impl LookupWitness for SaturatingAccClampOperands {
    fn witness(&self, node: &ComputationNode, trace: &Trace) -> Tensor<i64> {
        clamp_intermediate(node, trace)
    }

    fn lookup_bits(&self, witness: &Tensor<i64>) -> Vec<LookupBits> {
        clamp_lookup_bits(witness, self.log_k())
    }
}
