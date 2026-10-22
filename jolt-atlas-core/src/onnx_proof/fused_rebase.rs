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

use atlas_trace::{Op, Tensor, Trace};

use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    onnx_proof::{
        clamp_lookups::{
            is_scalar, prove_append_acc, prove_clamp_lookup, recover_small_int, verify_append_acc,
            verify_clamp_lookup, verify_scalar_clamp,
        },
        deferred_lookups::{ProverLookupJob, VerifierLookupJob},
        ProofId, Prover, Verifier,
    },
    utils::opening_access::AccOpeningAccessor,
};
use common::{CommittedPoly, VirtualPoly};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::JoltField,
    poly::{
        multilinear_polynomial::EvaluateMle,
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::{
        identity_range_check::IdentityRCProvider, shout::RaOneHotEncoding,
        sumcheck::SumcheckInstanceProof,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, lookup_bits::LookupBits},
};

/// Rebase width `S` (in bits) of a node's fused rescaling division at the
/// model's log2 `scale`, or `None` when the operator does not fuse a
/// power-of-two rescale (`MeanOfSquares` divides by `count·2^S` and takes its
/// own path).
///
/// `Mul`/`Square` are at scale `2S` and rebase by `2^S`; `Cube` is at scale
/// `3S` and rebases by `2^(2S)`; einsum is a (sum of) products at scale `2S`.
pub fn op_rebase_bits(op: &Op, scale: u8) -> Option<i32> {
    let scale = i32::from(scale);
    match op {
        Op::Einsum(_) => Some(scale),
        Op::Mul(m) => Some(if m.rebase { scale } else { 0 }),
        Op::Square(s) => Some(if s.rebase { scale } else { 0 }),
        Op::Cube(c) => Some(if c.rebase { scale * 2 } else { 0 }),
        _ => None,
    }
}

/// [`op_rebase_bits`] of a node view (which carries the model scale).
pub fn rebase_bits(node: &ComputationNode) -> Option<i32> {
    op_rebase_bits(&node.operator, node.scale)
}

/// Whether a node fuses the rescaling division (`bits > 0`).
///
/// A `bits == 0` op is a raw product (no rebase, no clamp) — e.g. the
/// building-block `Mul`/`Square` used inside decompositions.
pub fn fuses_rebase(node: &ComputationNode) -> bool {
    rebase_bits(node).is_some_and(|b| b > 0)
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

/// Human-readable operator name for scalar-clamp error messages.
fn op_name(op: &Op) -> &'static str {
    match op {
        Op::Einsum(_) => "Einsum",
        Op::Mul(_) => "Mul",
        Op::Square(_) => "Square",
        Op::Cube(_) => "Cube",
        _ => "rescale-op",
    }
}

/// The arithmetic sumcheck's initial claim: the raw accumulation `acc(r)`.
///
/// For a fused op this is `rescaled(r)·2^S + R(r)`, recovered from the pre-clamp
/// `rescaled` ([`VirtualPoly::ClampAcc`]) and the remainder advice
/// ([`VirtualPoly::RescaleRemainder`]). For a non-fused (`bits == 0`) op it is
/// just the node-output opening, leaving the legacy raw-product behavior intact.
pub fn fused_input_claim<F: JoltField>(
    accumulator: &dyn OpeningAccumulator<F>,
    node: &ComputationNode,
) -> F {
    let accessor = AccOpeningAccessor::new(accumulator, node);
    match rebase_bits(node) {
        Some(bits) if bits > 0 => {
            let rescaled = accessor.get_advice(VirtualPoly::ClampAcc).1;
            let remainder = accessor.get_advice(VirtualPoly::RescaleRemainder).1;
            rescaled * F::from_u64(1u64 << bits) + remainder
        }
        _ => accessor.get_reduced_opening().1,
    }
}

/// The committed one-hot decomposition polynomials a fused node must commit to:
/// the rescaling-remainder range check (`bits`-wide address) plus the saturating
/// clamp (64-bit address). Empty for non-fused or scalar nodes.
pub fn committed_polys(node: &ComputationNode) -> Vec<CommittedPoly> {
    // Both the clamp and the remainder one-hots are committed per packed
    // bucket at the model level (see `global_clamp`), never per node.
    let _ = node;
    Vec::new()
}

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
) -> (crate::onnx_proof::ops::ExecutionProofs<F, T>, Tensor<i32>) {
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
) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
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

/// Verifier counterpart of [`prove_pre`].
pub fn verify_pre<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError> {
    cache_remainder_verify(node, &mut verifier.accumulator, &mut verifier.transcript);
    if is_scalar(node) {
        verify_append_acc(node, &mut verifier.accumulator, &mut verifier.transcript);
    } else {
        verify_clamp_lookup(node, verifier)?;
    }
    Ok(())
}

/// Verifier counterpart of [`prove_remainder_rc`], plus the scalar fallback:
/// for a scalar node `rescaled` and `R` are opened in the clear, so check
/// `R ∈ [0, 2^S)` and `output == SatClamp(rescaled)` directly.
pub fn verify_post<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError> {
    let bits = rebase_bits(node).expect("fused op");

    if is_scalar(node) {
        let accessor = AccOpeningAccessor::new(&verifier.accumulator, node);
        let r_claim = accessor.get_advice(VirtualPoly::RescaleRemainder).1;
        verify_scalar_remainder(r_claim, bits)?;
        let rescaled_claim = accessor.get_advice(VirtualPoly::ClampAcc).1;
        let output_claim = accessor.get_reduced_opening().1;
        verify_scalar_clamp(rescaled_claim, output_claim, op_name(&node.operator))?;
        return Ok(());
    }

    verifier.deferred.push(VerifierLookupJob::RescaleRemainder {
        node: node.clone(),
        bits,
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Append the rescaling remainder `R` (padded to the node-output cycle domain,
/// as produced by [`rebase_intermediates`]) as a virtual advice opening at the
/// output point. `pub(crate)`: also used by the ZK flow (`onnx_proof::zk`),
/// where the append is transcript-quiet (accumulator `zk_mode`).
pub(crate) fn cache_remainder_prove<F: JoltField, T: Transcript>(
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

/// Verifier counterpart of [`cache_remainder_prove`]: append the remainder
/// opening point (its claim is loaded from the proof's opening claims; a
/// placeholder in ZK mode, where the value reaches the proof only through
/// the prover-supplied baked initial claims of the following stages).
pub(crate) fn cache_remainder_verify<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) {
    let accessor = AccOpeningAccessor::new(&mut *accumulator, node);
    let r0 = accessor.get_reduced_opening().0;
    let mut provider = accessor.into_provider(transcript, r0);
    provider.append_advice(VirtualPoly::RescaleRemainder);
}

/// A (padded) rescaling remainder tensor as `bits`-bit lookup indices.
pub(crate) fn remainder_lookup_bits(remainder: &Tensor<i32>, bits: i32) -> Vec<LookupBits> {
    remainder
        .data()
        .iter()
        .map(|&v| LookupBits::new(v as u64, bits as usize))
        .collect()
}

/// Verify a scalar node's remainder directly: `R ∈ [0, 2^S)`.
///
/// Used when the output is a single element (`log_T = 0`), where the one-hot
/// range-check degenerates. `R` is opened in the clear, so the verifier recovers
/// the small integer and bounds it (mirrors the clamp scalar fallback).
fn verify_scalar_remainder<F: JoltField>(r_claim: F, bits: i32) -> Result<(), ProofVerifyError> {
    let value = recover_small_int(r_claim).ok_or_else(|| {
        ProofVerifyError::InvalidOpeningProof(
            "fused rescale (scalar): remainder claim is not a small integer".to_string(),
        )
    })?;
    if value < 0 || value >= (1i64 << bits) {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "fused rescale (scalar): remainder must lie in [0, 2^S)".to_string(),
        ));
    }
    Ok(())
}

/// Identity range-check provider for a fused rescaling remainder.
///
/// `log_K = bits`: the check proves `R ∈ [0, 2^bits)`.
pub struct RescaleRemainderRCProvider {
    node: ComputationNode,
    log_k: usize,
}

impl RescaleRemainderRCProvider {
    /// Create the provider for a `bits`-wide (`2^bits` divisor) rebase.
    pub fn new(node: ComputationNode, bits: i32) -> Self {
        Self {
            node,
            log_k: bits as usize,
        }
    }
}

impl<F: JoltField> IdentityRCProvider<F> for RescaleRemainderRCProvider {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        AccOpeningAccessor::new(accumulator, &self.node)
            .get_advice(VirtualPoly::RescaleRemainder)
            .1
    }

    fn log_K(&self) -> usize {
        self.log_k
    }

    fn r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F> {
        AccOpeningAccessor::new(accumulator, &self.node)
            .get_advice(VirtualPoly::RescaleRemainder)
            .0
    }

    fn ra_poly(&self) -> (VirtualPoly, SumcheckId) {
        (
            VirtualPoly::RescaleRemainderRa(self.node.idx),
            SumcheckId::NodeExecution(self.node.idx),
        )
    }
}

/// One-hot encoding for a fused rescaling-remainder range check.
pub struct RescaleRemainderRaEncoding {
    node_idx: usize,
    log_k: usize,
}

impl RescaleRemainderRaEncoding {
    /// Create the encoding for a `bits`-wide rebase.
    pub fn new(node_idx: usize, bits: i32) -> Self {
        Self {
            node_idx,
            log_k: bits as usize,
        }
    }
}

impl RaOneHotEncoding for RescaleRemainderRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::RescaleRemainderRaD(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        // The remainder is opened at the node-output point.
        OpeningId::new(
            VirtualPoly::NodeOutput(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::RescaleRemainderRa(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn log_k(&self) -> usize {
        self.log_k
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), self.log_k)
    }
}
