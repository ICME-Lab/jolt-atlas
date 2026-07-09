//! 64-bit saturating-clamp lookup infrastructure for `Add`/`Sub`.
//!
//! This is the 64-bit sibling of [`op_lookups`](super::op_lookups). Where
//! `op_lookups` proves a unary lookup whose index is a node *operand* (32-bit),
//! the clamp lookup proves
//!
//! ```text
//! output(x) = SatClamp(acc(x)),   acc(x) = left(x) ± right(x)
//! ```
//!
//! Here the lookup index is the **pre-clamp i64 accumulation** `acc`, recovered
//! by re-executing [`sat_binop_intermediate`] on the operands (no trace change).
//! The accumulation MLE is the `raf` polynomial ([`VirtualPoly::ClampAcc`]); the
//! `SatClampTable<64>` discharges the clamp; and the linear identity
//! `acc(r) = left(r) ± right(r)` (checked by the caller in `Add`/`Sub`) ties the
//! accumulation back to the operands.
//!
//! The one-hot read-address checks ([`ClampEncoding`]) are over a 64-bit address,
//! so the decomposition has `64 / log_k_chunk` committed chunks
//! ([`CommittedPoly::ClampRaD`]).

use crate::onnx_proof::{ProofId, ProofType, Prover, Verifier};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::{sat_binop_intermediate, sum::sum_axes_i64, Operator},
    tensor::Tensor,
};
use common::{parallel::par_enabled, CommittedPoly, VirtualPoly};
use joltworks::{
    config::OneHotParams,
    field::JoltField,
    lookup_tables::{sat_clamp::SatClampTable, JoltLookupTable, PrefixSuffixDecompositionTrait},
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
    },
    subprotocols::{
        ps_shout::{
            unary::{
                ps_read_raf_prover, ps_read_raf_verifier, PrefixSuffixShoutProvider, ReadRafClaims,
                UnaryReadRafSumcheckProver, UnaryReadRafSumcheckVerifier,
            },
            RafShoutProvider,
        },
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, lookup_bits::LookupBits},
};
use rayon::prelude::*;

/// Address width of the saturating-clamp lookup table.
///
/// The accumulation is widened to `i64` before clamping, so the table is indexed
/// by all 64 bits of the (two's-complement) accumulation value.
pub const CLAMP_LOG_K: usize = 64;

/// The saturating-clamp lookup table used by `Add`/`Sub`, indexed by the i64
/// accumulation.
pub type ClampTable = SatClampTable<CLAMP_LOG_K>;

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
fn acc_opening_id(node_idx: usize) -> OpeningId {
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

/// 64-bit clamp lookup indices: each i64 accumulation value reinterpreted as the
/// `u64` (two's-complement) address into [`SatClampTable`].
pub(crate) fn clamp_lookup_bits(intermediate: &Tensor<i64>) -> Vec<LookupBits> {
    intermediate
        .data()
        .par_iter()
        .with_min_len(par_enabled())
        .map(|&v| LookupBits::new(v as u64, CLAMP_LOG_K))
        .collect()
}

/// Provider for the saturating-clamp read-raf execution sumcheck.
///
/// Mirrors [`OpLookupProvider`](super::op_lookups::OpLookupProvider) but at
/// `LOG_K = 64` and with the lookup index sourced from the re-executed i64
/// accumulation rather than a node operand.
pub struct ClampLookupProvider {
    computation_node: ComputationNode,
}

impl ClampLookupProvider {
    /// Create a new clamp lookup provider for the given (binary `Add`/`Sub`) node.
    pub fn new(computation_node: ComputationNode) -> Self {
        Self { computation_node }
    }

    fn acc_id(&self) -> OpeningId {
        acc_opening_id(self.computation_node.idx)
    }

    /// Prover flow: append the accumulation (`raf`) claim, then build the
    /// read-raf sumcheck prover. Returns `(prover, lookup_indices)`, where the
    /// indices are reused for the one-hot checks. `intermediate` optionally
    /// provides the precomputed (padded) accumulation (see
    /// [`prove_append_acc`]).
    pub fn read_raf_prove<F, T>(
        &self,
        trace: &Trace,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        intermediate: Option<Tensor<i64>>,
    ) -> (
        UnaryReadRafSumcheckProver<F, ClampTable, CLAMP_LOG_K>,
        Vec<usize>,
    )
    where
        F: JoltField,
        T: Transcript,
    {
        let intermediate = prove_append_acc(
            &self.computation_node,
            trace,
            accumulator,
            transcript,
            intermediate,
        );
        let lookup_bits = clamp_lookup_bits(&intermediate);
        let lookup_indices: Vec<usize> = lookup_bits.iter().map(|&x| x.into()).collect();
        let prover = ps_read_raf_prover(self, lookup_bits, accumulator, transcript);
        (prover, lookup_indices)
    }

    /// Verifier flow: append the accumulation (`raf`) opening point (claim was
    /// loaded from the proof), then build the read-raf sumcheck verifier.
    pub fn read_raf_verify<F, T>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> UnaryReadRafSumcheckVerifier<F, ClampTable, CLAMP_LOG_K>
    where
        F: JoltField,
        T: Transcript,
    {
        verify_append_acc(&self.computation_node, accumulator, transcript);
        ps_read_raf_verifier(self, accumulator, transcript)
    }
}

impl<F, LUT> RafShoutProvider<F, LUT> for ClampLookupProvider
where
    F: JoltField,
    LUT: JoltLookupTable + Default,
{
    fn r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F> {
        accumulator
            .get_node_output_opening(self.computation_node.idx)
            .0
    }

    fn ra_poly(&self) -> (VirtualPoly, SumcheckId) {
        (
            VirtualPoly::ClampRa(self.computation_node.idx),
            SumcheckId::NodeExecution(self.computation_node.idx),
        )
    }
}

impl<F, LUT> PrefixSuffixShoutProvider<F, LUT, CLAMP_LOG_K> for ClampLookupProvider
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<CLAMP_LOG_K> + Default,
{
    fn read_raf_claims(&self, accumulator: &dyn OpeningAccumulator<F>) -> ReadRafClaims<F> {
        let (_, rv_claim) = accumulator.get_node_output_opening(self.computation_node.idx);
        let (_, operand_claim) = accumulator.get_virtual_polynomial_opening(self.acc_id());
        ReadRafClaims {
            rv_claim,
            operand_claim,
        }
    }
}

/// One-hot encoding for the saturating-clamp read-address checks (booleanity,
/// hamming-weight, ra-virtualization) over the 64-bit clamp address.
pub struct ClampEncoding {
    /// Index of the computation node using this lookup encoding.
    pub node_idx: usize,
    /// log₂(T): number of (padded) output elements in the node.
    pub log_t: usize,
}

impl ClampEncoding {
    /// Create a new clamp encoding for the given computation node.
    pub fn new(computation_node: &ComputationNode) -> Self {
        use joltworks::utils::math::Math;
        Self {
            node_idx: computation_node.idx,
            log_t: computation_node.pow2_padded_num_output_elements().log_2(),
        }
    }
}

impl RaOneHotEncoding for ClampEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::ClampRaD(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::NodeOutput(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::ClampRa(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn log_k(&self) -> usize {
        CLAMP_LOG_K
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::new(self.log_t, self.log_k())
    }
}

// ---------------------------------------------------------------------------
// Shared clamp sub-protocol used by Add / Sub / Sum
// ---------------------------------------------------------------------------

/// Prove `output = SatClamp(acc)` for a non-scalar node: the clamp read-raf
/// lookup ([`ProofType::Execution`]) plus the read-address one-hot checks
/// ([`ProofType::RaOneHotChecks`]). The accumulation `acc` is appended as the
/// lookup `raf` (see [`ClampLookupProvider`]); `intermediate` optionally
/// provides it precomputed (see [`prove_append_acc`]).
pub fn prove_clamp_lookup<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
    intermediate: Option<Tensor<i64>>,
) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
    let provider = ClampLookupProvider::new(node.clone());
    let (mut execution_sumcheck, lookup_indices) = provider.read_raf_prove::<F, T>(
        &prover.trace,
        &mut prover.accumulator,
        &mut prover.transcript,
        intermediate,
    );
    let (execution_proof, _) = Sumcheck::prove(
        &mut execution_sumcheck,
        &mut prover.accumulator,
        &mut prover.transcript,
    );

    let encoding = ClampEncoding::new(node);
    let [ra_prover, hw_prover, bool_prover] = shout::ra_onehot_provers(
        &encoding,
        &lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
        vec![ra_prover, hw_prover, bool_prover];
    let (ra_one_hot_proof, _) = BatchedSumcheck::prove(
        instances.iter_mut().map(|v| &mut **v as _).collect(),
        &mut prover.accumulator,
        &mut prover.transcript,
    );

    vec![
        (ProofId(node.idx, ProofType::Execution), execution_proof),
        (
            ProofId(node.idx, ProofType::RaOneHotChecks),
            ra_one_hot_proof,
        ),
    ]
}

/// Verifier counterpart of [`prove_clamp_lookup`].
pub fn verify_clamp_lookup<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError> {
    let provider = ClampLookupProvider::new(node.clone());
    let execution_verifier =
        provider.read_raf_verify::<F, T>(&mut verifier.accumulator, &mut verifier.transcript);
    let execution_proof = verifier
        .proofs
        .get(&ProofId(node.idx, ProofType::Execution))
        .ok_or(ProofVerifyError::MissingProof(node.idx))?;
    Sumcheck::verify(
        execution_proof,
        &execution_verifier,
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )?;

    let encoding = ClampEncoding::new(node);
    let [ra_verifier, hw_verifier, bool_verifier] =
        shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript);
    let ra_one_hot_proof = verifier
        .proofs
        .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
        .ok_or(ProofVerifyError::MissingProof(node.idx))?;
    BatchedSumcheck::verify(
        ra_one_hot_proof,
        vec![&*ra_verifier, &*hw_verifier, &*bool_verifier],
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )?;
    Ok(())
}

/// The committed one-hot decomposition polynomials a clamped node must commit to
/// (empty for scalar nodes, which prove the clamp directly).
pub fn clamp_committed_polys(node: &ComputationNode) -> Vec<CommittedPoly> {
    if is_scalar(node) {
        return Vec::new();
    }
    let d = ClampEncoding::new(node).one_hot_params().instruction_d;
    (0..d)
        .map(|i| CommittedPoly::ClampRaD(node.idx, i))
        .collect()
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
