//! Shared fused-rescaling (division + saturating clamp) sub-protocol .
//!
//! Several operators fuse an i64 accumulate, a rescaling division by `2^S`, and a
//! saturating clamp into a single node: einsum (`Σ_k L·R`), `Mul` (`left·right`),
//! `Square` (`x²`), and `Cube` (`x³`). They all share the same proof skeleton,
//! which lives here so each operator only has to supply its own *arithmetic*
//! sumcheck (`acc(r) = Σ … `):
//!
//! ```text
//! acc(r) = rescaled(r)·2^S + R(r)     (the operator's arithmetic sumcheck claim)
//! output = SatClamp(rescaled)         (clamp lookup, reuses `clamp_lookups`)
//! R ∈ [0, 2^S)                        (identity range-check on the remainder)
//! ```
//!
//! `rescaled` is the pre-clamp value ([`VirtualPoly::ClampAcc`]), recovered as the
//! floor-rebased accumulation; `R` is the rescaling remainder, appended as advice
//! ([`VirtualPoly::RescaleRemainder`]) and range-checked. The arithmetic sumcheck
//! reads both back through [`fused_input_claim`] so its initial claim is the raw
//! `acc(r)` while its `Σ L·R` body is unchanged.
//!
//! Scalar nodes (`log_T = 0`) skip the one-hot reductions and open `rescaled` and
//! `R` in the clear (mirroring the `Add`/`Sub`/`Sum` clamp fallback).

use atlas_trace::{Tensor, Trace};
pub use jolt_atlas_verifier::onnx_proof::fused_rebase::*;

use crate::{
    node_view::ComputationNode,
    onnx_proof::{
        clamp_lookups::{is_scalar, prove_append_acc, prove_clamp_lookup},
        deferred_lookups::ProverLookupJob,
        trace_view::TraceView,
        ProofId, Prover,
    },
    utils::opening_access::AccOpeningAccessor,
};
use atlas_algebra::{
    field::JoltField, poly::multilinear_polynomial::EvaluateMle, utils::lookup_bits::LookupBits,
};
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_transcript::Transcript;
use common::VirtualPoly;

// ---------------------------------------------------------------------------
// Prover / verifier orchestration (the stages around the arithmetic sumcheck)
// ---------------------------------------------------------------------------

/// Stage (1)+(2): append the remainder `R` advice and discharge the clamp
/// `output = SatClamp(rescaled)`. For a scalar node the clamp is checked in the
/// clear (only `rescaled` is appended); otherwise the clamp lookup
/// (`Execution` + `RaOneHotChecks`) is proven. Run *before* the operator's
/// arithmetic sumcheck so [`fused_input_claim`] can read both advices.
///
/// Also returns the (padded) remainder tensor so [`prove_remainder_rc`] can
/// reuse it instead of re-running the accumulation.
pub fn prove_pre<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> (crate::onnx_proof::ops::ExecutionProofs<F>, Tensor<i32>) {
    let RebaseIntermediates {
        quotient,
        remainder,
    } = rebase_intermediates(node, &prover.trace);
    cache_remainder_prove(node, prover, &remainder);
    let proofs = if is_scalar(node) {
        prove_append_acc(
            node,
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
            Some(quotient),
        );
        Vec::new()
    } else {
        prove_clamp_lookup(node, prover, Some(quotient))
    };
    (proofs, remainder)
}

/// Stage (4)+(5): the rescaling-remainder range check `R ∈ [0, 2^S)`
/// (`RangeCheck`) and its read-address one-hot checks
/// (`RescaleRemainderRaChecks`). Run *after* the operator's arithmetic sumcheck;
/// only for non-scalar nodes. `remainder` is the padded tensor returned by
/// [`prove_pre`].
pub fn prove_remainder_rc<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
    remainder: &Tensor<i32>,
) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
    let bits = rebase_bits(node).expect("fused op");
    let lookup_bits = remainder_lookup_bits(remainder, bits);
    // Deferred: proven after the node loop as part of the batched remainder
    // range check + one-hot checks (see `deferred_lookups`).
    prover.deferred.push(ProverLookupJob::RescaleRemainder {
        node: node.clone(),
        bits,
        lookup_bits,
    });
    Vec::new()
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Append the rescaling remainder `R` (padded to the node-output cycle domain,
/// as produced by [`rebase_intermediates`]) as a virtual advice opening at the
/// output point.
fn cache_remainder_prove<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
    remainder: &Tensor<i32>,
) {
    let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
    let r0 = accessor.get_reduced_opening().0;
    let eval = remainder.evaluate_mle(&r0.r);
    let mut provider = accessor.into_provider(&mut prover.transcript, r0);
    provider.append_advice(VirtualPoly::RescaleRemainder, eval);
}

/// Both fused-rescale intermediates — the pre-clamp quotient `rescaled = acc div D`
/// and the remainder `R = acc mod D` — derived from **one** accumulation pass and
/// padded to the node-output cycle domain.
///
/// The accumulation (e.g. the full einsum contraction) dominates the cost of
/// recovering either tensor, so anything that needs both must compute them
/// together rather than calling [`clamp_lookups::clamp_intermediate`] and
/// [`rebase_remainder`] separately.
pub(crate) struct RebaseIntermediates {
    /// Pre-clamp rescaled accumulation (the saturating-clamp lookup index).
    pub quotient: Tensor<i64>,
    /// Rescaling remainder `R ∈ [0, D)`.
    pub remainder: Tensor<i32>,
}

/// Compute a node's [`RebaseIntermediates`] with a single accumulation pass,
/// or `None` if the operator has no fused rescale (e.g. `Add`/`Sub`/`Sum`).
pub(crate) fn try_rebase_intermediates(
    node: &ComputationNode,
    trace: &Trace,
) -> Option<RebaseIntermediates> {
    // The executor always captures fused intermediates; nothing re-executes.
    let cached = trace.fused_intermediates(node.idx)?;
    Some(RebaseIntermediates {
        quotient: cached.quotient.padded_pow2(),
        remainder: cached.remainder.padded_pow2(),
    })
}

/// Panicking variant of [`try_rebase_intermediates`] for callers that know the
/// node fuses a rescale.
pub(crate) fn rebase_intermediates(node: &ComputationNode, trace: &Trace) -> RebaseIntermediates {
    try_rebase_intermediates(node, trace)
        .unwrap_or_else(|| panic!("fused_rebase: unsupported operator {}", node.operator))
}

/// The operator's rescaling remainder `R = acc mod D`, padded to the node-output
/// cycle domain, where `D` is the operator-specific divisor (usually `2^S`, but
/// e.g. `MeanOfSquares` uses `N·2^S`). Dispatches to the per-operator re-execution
/// kernel (no trace change), mirroring [`clamp_lookups::clamp_intermediate`].
///
/// Prefer [`rebase_intermediates`] when the quotient is needed too — this re-runs the full accumulation for the remainder alone.
pub(crate) fn rebase_remainder(node: &ComputationNode, trace: &Trace) -> Tensor<i32> {
    trace
        .fused_intermediates(node.idx)
        .unwrap_or_else(|| {
            panic!(
                "fused_rebase: node {} ({}) has no trace intermediates",
                node.idx, node.operator
            )
        })
        .remainder
        .padded_pow2()
}

/// A (padded) rescaling remainder tensor as `bits`-bit lookup indices.
pub(crate) fn remainder_lookup_bits(remainder: &Tensor<i32>, bits: i32) -> Vec<LookupBits> {
    remainder
        .data()
        .iter()
        .map(|&v| LookupBits::new(v as u64, bits as usize))
        .collect()
}
