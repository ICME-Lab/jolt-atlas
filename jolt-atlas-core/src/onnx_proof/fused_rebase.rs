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

use crate::{
    onnx_proof::{
        clamp_lookups::{
            clamp_committed_polys, is_scalar, prove_append_acc, prove_clamp_lookup,
            recover_small_int, verify_append_acc, verify_clamp_lookup, verify_scalar_clamp,
        },
        ProofId, ProofType, Prover, Verifier,
    },
    utils::opening_access::AccOpeningAccessor,
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::{
        cube::cube_remainder, einsum::einsum_remainder, mean_of_squares::mos_remainder,
        mul::mul_remainder, square::square_remainder, Operator,
    },
    tensor::Tensor,
};
use common::{CommittedPoly, VirtualPoly};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::JoltField,
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::{
        identity_range_check::{
            identity_rangecheck_prover, identity_rangecheck_verifier, IdentityRCProvider,
        },
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, lookup_bits::LookupBits},
};

/// Rebase width `S` (in bits) of an operator's fused rescaling division, or
/// `None` if the operator does not fuse a rescale.
///
/// `Mul`/`Square` are at scale `2S` and rebase by `2^S`; `Cube` is at scale `3S`
/// and rebases by `2^(2S)`; einsum is a (sum of) products at scale `2S`.
pub fn rebase_bits(op: &Operator) -> Option<i32> {
    match op {
        Operator::Einsum(e) => Some(e.scale),
        Operator::Mul(m) => Some(m.scale),
        Operator::Square(s) => Some(s.scale),
        Operator::Cube(c) => Some(c.scale * 2),
        _ => None,
    }
}

/// Whether a node fuses the rescaling division (`bits > 0`).
///
/// A `bits == 0` op is a raw product (no rebase, no clamp) — e.g. the
/// building-block `Mul`/`Square` used inside decompositions.
pub fn fuses_rebase(op: &Operator) -> bool {
    rebase_bits(op).is_some_and(|b| b > 0)
}

/// The operator's rescaling remainder `R = acc mod 2^S`, padded to the
/// node-output cycle domain. Dispatches to the per-operator re-execution kernel
/// (no trace change), mirroring [`clamp_lookups::clamp_intermediate`].
pub(crate) fn rebase_remainder(node: &ComputationNode, trace: &Trace) -> Tensor<i32> {
    let LayerData { operands, .. } = Trace::layer_data(trace, node);
    let remainder = match &node.operator {
        Operator::Einsum(op) => einsum_remainder(op, &operands),
        Operator::Mul(op) => mul_remainder(op, &operands),
        Operator::Square(op) => square_remainder(op, &operands),
        Operator::Cube(op) => cube_remainder(op, &operands),
        Operator::MeanOfSquares(op) => mos_remainder(op, &operands),
        other => panic!("fused_rebase: unsupported operator {other:?}"),
    };
    remainder.padded_next_power_of_two()
}

/// Human-readable operator name for scalar-clamp error messages.
fn op_name(op: &Operator) -> &'static str {
    match op {
        Operator::Einsum(_) => "Einsum",
        Operator::Mul(_) => "Mul",
        Operator::Square(_) => "Square",
        Operator::Cube(_) => "Cube",
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
    match rebase_bits(&node.operator) {
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
    if !fuses_rebase(&node.operator) || is_scalar(node) {
        return vec![];
    }
    let bits = rebase_bits(&node.operator).expect("fused op") as usize;
    let d = OneHotParams::from_config_and_log_K(&OneHotConfig::default(), bits).instruction_d;
    let mut polys: Vec<CommittedPoly> = (0..d)
        .map(|i| CommittedPoly::RescaleRemainderRaD(node.idx, i))
        .collect();
    polys.extend(clamp_committed_polys(node));
    polys
}

// ---------------------------------------------------------------------------
// Prover / verifier orchestration (the stages around the arithmetic sumcheck)
// ---------------------------------------------------------------------------

/// Stage (1)+(2): append the remainder `R` advice and discharge the clamp
/// `output = SatClamp(rescaled)`. For a scalar node the clamp is checked in the
/// clear (only `rescaled` is appended); otherwise the clamp lookup
/// (`Execution` + `RaOneHotChecks`) is proven. Run *before* the operator's
/// arithmetic sumcheck so [`fused_input_claim`] can read both advices.
pub fn prove_pre<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
    cache_remainder_prove(node, prover);
    if is_scalar(node) {
        prove_append_acc(
            node,
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        Vec::new()
    } else {
        prove_clamp_lookup(node, prover)
    }
}

/// Stage (4)+(5): the rescaling-remainder range check `R ∈ [0, 2^S)`
/// (`RangeCheck`) and its read-address one-hot checks
/// (`RescaleRemainderRaChecks`). Run *after* the operator's arithmetic sumcheck;
/// only for non-scalar nodes.
pub fn prove_remainder_rc<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
    let bits = rebase_bits(&node.operator).expect("fused op");
    let lookup_bits = rebase_remainder_lookup_bits(node, &prover.trace, bits);
    let lookup_indices: Vec<usize> = lookup_bits.iter().map(|&x| x.into()).collect();

    let rc_provider = RescaleRemainderRCProvider::new(node.clone(), bits);
    let mut rc = identity_rangecheck_prover(&rc_provider, lookup_bits, &mut prover.accumulator);
    let (rc_proof, _) = Sumcheck::prove(&mut rc, &mut prover.accumulator, &mut prover.transcript);

    let encoding = RescaleRemainderRaEncoding::new(node.idx, bits);
    let [ra, hw, boolean] = shout::ra_onehot_provers(
        &encoding,
        &lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let mut instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = vec![ra, hw, boolean];
    let (ra_proof, _) = BatchedSumcheck::prove(
        instances.iter_mut().map(|v| &mut **v as _).collect(),
        &mut prover.accumulator,
        &mut prover.transcript,
    );

    vec![
        (ProofId(node.idx, ProofType::RangeCheck), rc_proof),
        (
            ProofId(node.idx, ProofType::RescaleRemainderRaChecks),
            ra_proof,
        ),
    ]
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
    let bits = rebase_bits(&node.operator).expect("fused op");

    if is_scalar(node) {
        let accessor = AccOpeningAccessor::new(&verifier.accumulator, node);
        let r_claim = accessor.get_advice(VirtualPoly::RescaleRemainder).1;
        verify_scalar_remainder(r_claim, bits)?;
        let rescaled_claim = accessor.get_advice(VirtualPoly::ClampAcc).1;
        let output_claim = accessor.get_reduced_opening().1;
        verify_scalar_clamp(rescaled_claim, output_claim, op_name(&node.operator))?;
        return Ok(());
    }

    let rc_proof = verifier
        .proofs
        .get(&ProofId(node.idx, ProofType::RangeCheck))
        .ok_or(ProofVerifyError::MissingProof(node.idx))?;
    let rc_provider = RescaleRemainderRCProvider::new(node.clone(), bits);
    let rc = identity_rangecheck_verifier(&rc_provider, &mut verifier.accumulator);
    Sumcheck::verify(
        rc_proof,
        &rc,
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )?;

    let ra_proof = verifier
        .proofs
        .get(&ProofId(node.idx, ProofType::RescaleRemainderRaChecks))
        .ok_or(ProofVerifyError::MissingProof(node.idx))?;
    let encoding = RescaleRemainderRaEncoding::new(node.idx, bits);
    let [ra, hw, boolean] =
        shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript);
    BatchedSumcheck::verify(
        ra_proof,
        vec![&*ra, &*hw, &*boolean],
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Re-execute the rescaling remainder `R` (padded to the node-output cycle
/// domain) and append it as a virtual advice opening at the output point.
fn cache_remainder_prove<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) {
    let remainder = rebase_remainder(node, &prover.trace);
    let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
    let r0 = accessor.get_reduced_opening().0;
    let eval = MultilinearPolynomial::from(remainder).evaluate(&r0.r);
    let mut provider = accessor.into_provider(&mut prover.transcript, r0);
    provider.append_advice(VirtualPoly::RescaleRemainder, eval);
}

/// Verifier counterpart of [`cache_remainder_prove`]: append the remainder
/// opening point (its claim is loaded from the proof's opening claims).
fn cache_remainder_verify<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) {
    let accessor = AccOpeningAccessor::new(&mut *accumulator, node);
    let r0 = accessor.get_reduced_opening().0;
    let mut provider = accessor.into_provider(transcript, r0);
    provider.append_advice(VirtualPoly::RescaleRemainder);
}

/// The rescaling remainder `R` as `bits`-bit lookup indices, padded to the
/// node-output cycle domain.
fn rebase_remainder_lookup_bits(
    node: &ComputationNode,
    trace: &Trace,
    bits: i32,
) -> Vec<LookupBits> {
    rebase_remainder(node, trace)
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
