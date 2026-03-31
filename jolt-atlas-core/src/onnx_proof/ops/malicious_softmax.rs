//! Malicious variant of the SoftmaxLastAxis prover for soundness testing.
//!
//! Demonstrates the remainder range-check attack: the cheating prover tampers
//! the softmax output in the trace, then forges the remainder evaluation `R(τ)`
//! so that the reciprocal-multiply sumcheck still passes.  Because `R` is not
//! yet PCS-committed and no range check constrains it to `[0, S)`, the verifier
//! accepts.
//!
//! Once Stage 1b (Shout range check on `R`) is implemented, forged `R` values
//! (huge field elements representing negative integers) will be caught, and
//! proofs produced by this module will be rejected.

use crate::onnx_proof::{
    ops::softmax_last_axis::recip_mult::{RecipMultParams, RecipMultProver},
    ProofId, ProofType, Prover,
};
use atlas_onnx_tracer::{node::ComputationNode, ops::softmax::softmax_last_axis};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{OpeningAccumulator, SumcheckId},
    },
    subprotocols::sumcheck::{Sumcheck, SumcheckInstanceProof},
    transcripts::Transcript,
};

/// Run the malicious SoftmaxLastAxis prover for a single node.
///
/// Assumes the trace output for `node` has already been tampered by the caller.
/// Recomputes the honest `exp_q` / `inv_sum_expanded` from the *untampered*
/// operand, then forges `R(τ)` so that
///
/// ```text
///   tampered_softmax(τ) · S  +  R'(τ)  =  Σ_j eq(τ,j) · exp_q[j] · inv_sum[j]
/// ```
///
/// holds over the field.  The sumcheck prover operates on the honest `exp_q`
/// and `inv_sum_expanded` polynomials, so the sumcheck itself is sound — only
/// the link *from* `R'(τ)` to a committed polynomial is missing.
pub fn malicious_softmax_prove<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    scale: i32,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
    // ── 1. Recompute the honest witness from the (untampered) operand ──────
    let softmax_input = prover.trace.operand_tensors(node)[0];
    let (_, softmax_trace) = softmax_last_axis(softmax_input, scale);

    // ── 2. Read τ and the *tampered* softmax claim from the accumulator ────
    let (opening_point, tampered_softmax_claim) =
        prover.accumulator.get_node_output_opening(node.idx);
    let r = opening_point.r;

    // ── 3. Compute the honest sum  Σ eq(τ,j) · exp_q[j] · inv_sum_expanded[j] ──
    let eq = EqPolynomial::evals(&r);
    let honest_sum: F = softmax_trace
        .exp_q
        .iter()
        .zip(softmax_trace.inv_sum_expanded().iter())
        .zip(eq.iter())
        .map(|((e, inv), eq_val)| F::from_i32(*e) * F::from_i32(*inv) * *eq_val)
        .sum();

    // ── 4. Forge R'(τ)  ────────────────────────────────────────────────────
    //   input_claim  =  softmax_claim · S  +  R_claim
    //   We need input_claim == honest_sum, so:
    //   R'(τ) = honest_sum − tampered_softmax(τ) · S
    let fake_r_eval = honest_sum - tampered_softmax_claim * F::from_i32(scale);

    prover.accumulator.append_dense(
        &mut prover.transcript,
        CommittedPolynomial::SoftmaxRecipMultRemainder(node.idx),
        SumcheckId::Execution,
        r.clone(),
        fake_r_eval,
    );

    // ── 5. Run the *honest* reciprocal-multiply sumcheck ───────────────────
    //   The prover polynomials (exp_q, inv_sum_expanded) are genuine, and
    //   input_claim now equals honest_sum, so the sumcheck validates.
    let recip_mult_params = RecipMultParams::new(node.idx, scale, &prover.accumulator);
    let mut recip_mult_prover = RecipMultProver::initialize(
        softmax_trace.exp_q.clone(),
        softmax_trace.inv_sum_expanded(),
        recip_mult_params,
    );
    let (recip_mult_proof, _) = Sumcheck::prove(
        &mut recip_mult_prover,
        &mut prover.accumulator,
        &mut prover.transcript,
    );

    // ── 6. DUMMY operand linking (identical to honest path) ────────────────
    let (opening_point, _claim) = prover.accumulator.get_node_output_opening(node.idx);
    let operand = prover.trace.operand_tensors(node)[0];
    let operand_claim = MultilinearPolynomial::from(operand.clone()).evaluate(&opening_point.r);
    prover.accumulator.append_virtual(
        &mut prover.transcript,
        VirtualPolynomial::NodeOutput(node.inputs[0]),
        SumcheckId::NodeExecution(node.idx),
        opening_point,
        operand_claim,
    );

    vec![(
        ProofId(node.idx, ProofType::SoftmaxRecipMult),
        recip_mult_proof,
    )]
}
