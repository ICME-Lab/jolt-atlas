//! 64-bit saturating-clamp lookup infrastructure for `Add`/`Sub`/`Sum`/`Einsum`/`Mul`/
//! `Square`/`Cube`/`MeanOfSquares`.
//!
//! Driven by the same generic [`OpLookupProvider`](super::op_lookups::OpLookupProvider)/
//! [`OpLookupEncoding`](super::op_lookups::OpLookupEncoding) as
//! [`op_lookups`](super::op_lookups), parameterized here by [`SaturatingAccClampOperands`]:
//!
//! ```text
//! output(x) = SatClamp(acc(x)),   acc(x) = left(x) ± right(x)
//! ```
//!
//! The lookup index is the **pre-clamp i64 accumulation** `acc`, recovered
//! by re-executing [`sat_binop_intermediate`] on the operands (no trace change).
//! The accumulation MLE is the `raf` polynomial ([`VirtualPoly::ClampAcc`]); [`SaturationTable`]
//! discharges the clamp (same table shape `SymmetricClampOperands` uses for the ONNX `Clamp`
//! op, generalized to bound 31); and the linear identity `acc(r) = left(r) ± right(r)` (checked
//! by the caller in `Add`/`Sub`) ties the accumulation back to the operands.
//!
//! The one-hot read-address checks are over a 64-bit address, so the decomposition has
//! `64 / log_k_chunk` committed chunks ([`CommittedPoly::ClampRaD`]).

use super::op_lookups::OpLookupProvider;
use crate::node_view::ComputationNode;
use crate::onnx_proof::op_lookups::OpLookupProve;
use crate::onnx_proof::op_lookups::{LookupOperandsTrait, LookupWitness};
use crate::onnx_proof::trace_view::TraceView;
use crate::onnx_proof::{
    clamp_split::NodeSplit, deferred_lookups::ProverLookupJob, ProofId, Prover,
};
use atlas_algebra::utils::lookup_bits::LookupBits;
use atlas_algebra::{field::JoltField, poly::multilinear_polynomial::EvaluateMle};
use atlas_sumcheck::{
    opening_proof::prover::ProverOpeningAccumulator, sumcheck::SumcheckInstanceProof,
};
use atlas_trace::{op::sum_axes_i64, trace::LayerData, Op, Tensor, Trace};
use atlas_transcript::Transcript;
use common::parallel::par_enabled;
pub use jolt_atlas_verifier::onnx_proof::clamp_lookups::*;
use rayon::prelude::*;

/// Append the accumulation (`ClampAcc`) opening at the node output point `r`,
/// returning the (padded) i64 intermediate for reuse.
///
/// Shared by the clamp lookup (where `acc` is the `raf`) and the scalar fallback
/// (where the verifier checks the clamp on `acc` directly). Fused-rescale
/// callers pass the precomputed (padded) `intermediate` so the expensive
/// accumulation is not re-run; `None` re-executes it via
/// [`clamp_intermediate`].
pub fn prove_append_acc<F, T>(
    node: &ComputationNode,
    trace: &Trace,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
    intermediate: Option<Tensor<i64>>,
) -> Tensor<i64>
where
    F: JoltField,
    T: Transcript,
{
    let intermediate = intermediate.unwrap_or_else(|| clamp_intermediate(node, trace));
    let r = accumulator.get_node_output_opening(node.idx).0;
    let acc_claim = intermediate.evaluate_mle(&r.r);
    accumulator.append_virtual(transcript, acc_opening_id(node.idx), r, acc_claim);
    intermediate
}

// ---------------------------------------------------------------------------
// Shared clamp sub-protocol used by Add / Sub / Sum / Einsum / Mul / Square / Cube /
// MeanOfSquares
// ---------------------------------------------------------------------------

/// Prove `output = SatClamp(acc)` for a non-scalar node. The accumulation
/// `acc` is appended as the lookup `raf` claim here (so the node's own
/// sumcheck can consume it); the clamp itself is *deferred* — the node's
/// split witness (see [`clamp_split`](crate::onnx_proof::clamp_split)) is
/// registered on the prover and proven after the node loop by
/// [`deferred_lookups::prove_all`](crate::onnx_proof::deferred_lookups::prove_all)
/// as part of batched sumchecks across all nodes. Returns no per-node proofs.
/// `intermediate` optionally provides the precomputed accumulation (see
/// [`prove_append_acc`]).
pub fn prove_clamp_lookup<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
    intermediate: Option<Tensor<i64>>,
) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
    let witness = intermediate.unwrap_or_else(|| clamp_intermediate(node, &prover.trace));
    let provider =
        OpLookupProvider::with_helper(node.clone(), SaturatingAccClampOperands::for_node(node));
    provider.append_witness_claim_for(&witness, &mut prover.accumulator, &mut prover.transcript);
    if crate::onnx_proof::clamp_split::exact_output_prover(
        prover.preprocessing.model(),
        &prover.trace,
        node.idx,
    ) {
        // Public unsaturated output: the verifier checks `ClampAcc == out`.
        return Vec::new();
    }
    let output = TraceView::layer_data(&prover.trace, node)
        .output
        .padded_pow2();
    let split = NodeSplit::new(witness.data(), output.data());
    prover.deferred.push(ProverLookupJob::Clamp {
        node: node.clone(),
        split,
    });
    Vec::new()
}

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
/// [`SaturationTable`]. Shared with `witness.rs`'s `ClampRaD` witness generation.
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
