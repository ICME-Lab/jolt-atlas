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

use super::op_lookups::{DefaultLookupOperands, LookupOperandsTrait, OpLookupProvider};
use crate::onnx_proof::{
    clamp_split::NodeSplit,
    deferred_lookups::{ProverLookupJob, VerifierLookupJob},
    ProofId, Prover, Verifier,
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::{sat_binop_intermediate, sum::sum_axes_i64, Operator},
    tensor::Tensor,
};
use common::{parallel::par_enabled, CommittedPoly, VirtualPoly};
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, lookup_bits::LookupBits},
};
use rayon::prelude::*;

/// Widest address of the saturating-clamp lookup table (an i64 accumulation).
/// Individual nodes use `node.sat_clamp_bits ∈ {40, 48, 56, 64}` (see
/// `atlas_onnx_tracer::model::clamp_width`); this is the default / upper bound.
pub const CLAMP_LOG_K: usize = 64;

/// Whether `node`'s (padded) output is a single element (`log_T = 0`).
///
/// Such "scalar" `Add`/`Sub` nodes skip the clamp lookup: the one-hot PCS
/// opening reduction degenerates with an empty cycle dimension. This is sound
/// because a single element is opened *in the clear* at the empty point — the
/// verifier recovers the (committed, hence valid-`i32`) operands and checks the
/// saturating identity `output == SatClamp(left ± right)` directly (see
/// [`recover_small_int`]).
pub fn is_scalar(node: &ComputationNode) -> bool {
    node.pow2_padded_num_output_elements() == 1
}

/// Recover the signed integer encoded by a field element known to be a
/// small-magnitude `i32`/`i64` embedding (e.g. `left ± right` for `i32`
/// operands, which lies in `[2·i32::MIN, 2·i32::MAX]`).
///
/// Returns `None` if the element is neither a small non-negative value nor a
/// small negation — a malformed claim. Operand claims are otherwise bound to
/// committed `i32` node outputs, so honest values always recover.
pub fn recover_small_int<F: JoltField>(x: F) -> Option<i64> {
    if let Some(u) = x.to_u64() {
        i64::try_from(u).ok()
    } else {
        (-x).to_u64()
            .and_then(|u| i64::try_from(u).ok())
            .map(|n| -n)
    }
}

/// Opening id for a node's accumulation (`ClampAcc`) polynomial.
pub(crate) fn acc_opening_id(node_idx: usize) -> OpeningId {
    OpeningId::new(
        VirtualPoly::ClampAcc(node_idx),
        SumcheckId::NodeExecution(node_idx),
    )
}

/// Re-execute the node's saturating accumulation and return the **padded** i64
/// intermediate (pre-clamp) tensor — the `raf`/lookup-index polynomial.
///
/// - `Add`/`Sub`: element-wise `left ± right` (via [`sat_binop_intermediate`]).
/// - `Sum`: the un-clamped sum over the reduced axes (via [`sum_axes_i64`]).
///
/// The padding (to the next power of two) matches the node-output MLE domain, so
/// the accumulation and output polynomials share the same `log_T`.
pub(crate) fn clamp_intermediate(node: &ComputationNode, trace: &Trace) -> Tensor<i64> {
    // Fused rescale ops: the trace already holds the pre-clamp quotient.
    if let Some(cached) = trace.fused_intermediates(node.idx) {
        return cached.quotient.padded_next_power_of_two();
    }
    let LayerData { operands, .. } = Trace::layer_data(trace, node);
    let raw = match &node.operator {
        Operator::Add(_) | Operator::Sub(_) => {
            let [left, right] = operands[..] else {
                panic!(
                    "clamp lookup (Add/Sub) expects two operands, got {}",
                    operands.len()
                )
            };
            sat_binop_intermediate(&node.operator, left, right)
        }
        Operator::Sum(s) => {
            let [operand] = operands[..] else {
                panic!(
                    "clamp lookup (Sum) expects one operand, got {}",
                    operands.len()
                )
            };
            sum_axes_i64(operand, &s.axes).expect("clamp lookup: sum_axes_i64")
        }
        // Einsum / Mul / Square / Cube: the pre-clamp value is the floor-rebased
        // accumulation `rescaled = acc >> S` . The remainder of the
        // division is range-checked separately (see [`super::fused_rebase`]);
        // here we only discharge the saturating clamp `output = SatClamp(rescaled)`.
        Operator::Einsum(op) => atlas_onnx_tracer::ops::einsum::einsum_intermediate(op, &operands),
        Operator::Mul(op) => atlas_onnx_tracer::ops::mul::mul_intermediate(op, &operands),
        Operator::Square(op) => atlas_onnx_tracer::ops::square::square_intermediate(op, &operands),
        Operator::Cube(op) => atlas_onnx_tracer::ops::cube::cube_intermediate(op, &operands),
        Operator::MeanOfSquares(op) => {
            atlas_onnx_tracer::ops::mean_of_squares::mos_intermediate(op, &operands)
        }
        other => panic!("clamp lookup: unsupported operator {other:?}"),
    };
    raw.padded_next_power_of_two()
}

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
    let acc_claim = MultilinearPolynomial::from(intermediate.data().to_vec()).evaluate(&r.r);
    accumulator.append_virtual(transcript, acc_opening_id(node.idx), r, acc_claim);
    intermediate
}

/// Verifier counterpart of [`prove_append_acc`]: append the `ClampAcc` opening
/// point (claim is loaded from the proof's opening claims).
pub fn verify_append_acc<F, T>(
    node: &ComputationNode,
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) where
    F: JoltField,
    T: Transcript,
{
    let r = accumulator.get_node_output_opening(node.idx).0;
    accumulator.append_virtual(transcript, acc_opening_id(node.idx), r);
}

/// [`LookupOperandsTrait`] helper for the 64-bit accumulator-sourced saturating clamp, backed
/// by [`SaturationTable`].
pub(crate) struct SaturatingAccClampOperands {
    /// Set by fused callers (`fused_rebase`) that already computed the pre-clamp
    /// intermediate, to avoid re-deriving it. `None` re-executes via
    /// [`clamp_intermediate`].
    precomputed: Option<Tensor<i64>>,
    /// Lookup address width for this node (`node.sat_clamp_bits`).
    log_k: usize,
}

impl SaturatingAccClampOperands {
    /// Helper for `node`, using its statically derived clamp width.
    pub(crate) fn for_node(node: &ComputationNode) -> Self {
        Self {
            precomputed: None,
            log_k: node.sat_clamp_bits,
        }
    }
}

impl LookupOperandsTrait for SaturatingAccClampOperands {
    const LOG_K: usize = CLAMP_LOG_K;

    fn rv_claim<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn joltworks::poly::opening_proof::OpeningAccumulator<F>,
    ) -> F {
        DefaultLookupOperands::rv_claim(node, accumulator)
    }

    fn ra_virtual_poly(node_idx: usize) -> VirtualPoly {
        VirtualPoly::ClampRa(node_idx)
    }

    fn ra_committed_poly(node_idx: usize, d: usize) -> CommittedPoly {
        CommittedPoly::ClampRaD(node_idx, d)
    }

    fn witness_opening_id(node: &ComputationNode) -> OpeningId {
        acc_opening_id(node.idx)
    }

    fn r_cycle<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        DefaultLookupOperands::r_cycle(node, accumulator)
    }

    fn r_cycle_source(node_idx: usize) -> OpeningId {
        DefaultLookupOperands::r_cycle_source(node_idx)
    }

    fn witness(&self, node: &ComputationNode, trace: &Trace) -> Tensor<i64> {
        self.precomputed
            .clone()
            .unwrap_or_else(|| clamp_intermediate(node, trace))
    }

    fn lookup_bits(&self, witness: &Tensor<i64>) -> Vec<LookupBits> {
        clamp_lookup_bits(witness, self.log_k)
    }

    fn log_k(&self) -> usize {
        self.log_k
    }
}

/// Clamp lookup indices: each i64 accumulation value's low `bits` bits (its
/// two's-complement encoding at the node's clamp width — see
/// `atlas_onnx_tracer::model::clamp_width`), the address into
/// [`SaturationTable`]. Shared with `witness.rs`'s `ClampRaD` witness generation.
pub(crate) fn clamp_lookup_bits(intermediate: &Tensor<i64>, bits: usize) -> Vec<LookupBits> {
    intermediate
        .data()
        .par_iter()
        .with_min_len(par_enabled())
        .map(|&v| LookupBits::new(v as u64, bits))
        .collect()
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
) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
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
    let output = Trace::layer_data(&prover.trace, node)
        .output
        .padded_next_power_of_two();
    let split = NodeSplit::new(witness.data(), output.data());
    prover.deferred.push(ProverLookupJob::Clamp {
        node: node.clone(),
        split,
    });
    Vec::new()
}

/// Verifier counterpart of [`prove_clamp_lookup`]: appends the `raf` opening
/// and registers the deferred lookup for
/// [`deferred_lookups::verify_all`](crate::onnx_proof::deferred_lookups::verify_all).
pub fn verify_clamp_lookup<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError> {
    let provider =
        OpLookupProvider::with_helper(node.clone(), SaturatingAccClampOperands::for_node(node));
    provider.append_witness_claim_verifier(&mut verifier.accumulator, &mut verifier.transcript);
    if crate::onnx_proof::clamp_split::exact_output_verifier(
        verifier.preprocessing.model(),
        verifier.io,
        node.idx,
    ) {
        let acc = verifier
            .accumulator
            .get_virtual_polynomial_opening(acc_opening_id(node.idx))
            .1;
        let out = verifier.accumulator.get_node_output_opening(node.idx).1;
        if acc != out {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "output node {}: unsaturated public output must equal the pre-clamp accumulation",
                node.idx
            )));
        }
        return Ok(());
    }
    verifier
        .deferred
        .push(VerifierLookupJob::Clamp { node: node.clone() });
    Ok(())
}

/// A clamped node commits **no** per-node polynomials: its clamp one-hots live
/// in a packed bucket (see [`global_clamp`](crate::onnx_proof::global_clamp)),
/// whose chunks are listed at the model level by
/// `global_clamp::bucket_committed_polys`.
pub fn clamp_committed_polys(_node: &ComputationNode) -> Vec<CommittedPoly> {
    Vec::new()
}

/// Verify a scalar node's clamp directly: `output_claim == SatClamp(combined)`,
/// where `combined` is the in-the-clear accumulation (`left ± right`, or the
/// axis sum). `op` names the operator for error messages.
pub fn verify_scalar_clamp<F: JoltField>(
    combined: F,
    output_claim: F,
    op: &str,
) -> Result<(), ProofVerifyError> {
    let value = recover_small_int(combined).ok_or_else(|| {
        ProofVerifyError::InvalidOpeningProof(format!(
            "{op} (scalar): accumulation claim is not a small signed-integer encoding"
        ))
    })?;
    let expected = F::from_i32(value.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
    if output_claim != expected {
        return Err(ProofVerifyError::InvalidOpeningProof(format!(
            "{op} (scalar): output must equal SatClamp(input)"
        )));
    }
    Ok(())
}
