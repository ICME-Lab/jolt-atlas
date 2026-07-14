//! Malicious variant of the Sub prover for soundness testing.
//!
//! The honest `Sub` proof (see [`crate::onnx_proof::ops::sub`]) proves
//! `output = SatClamp(left - right)` via a clamp lookup + one-hot checks, then
//! ties the accumulation to the operands with `left(r) - right(r) == acc(r)`.
//!
//! This module runs the lookup and one-hot checks **honestly** (so the proof is
//! structurally complete and the committed `ClampRaD` polynomials get opened),
//! but forges the left operand opening (off by one) so that
//! `left(r) - right(r) != acc(r)`, exercising that the verifier's operand tie
//! rejects the attack. Used exclusively by
//! [`MaliciousONNXProof`](crate::onnx_proof::malicious_prover).

use crate::{
    onnx_proof::{
        clamp_lookups::{ClampEncoding, ClampLookupProvider},
        ProofId, ProofType, Prover,
    },
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
};
use joltworks::{
    field::JoltField,
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    subprotocols::{
        shout,
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
    },
    transcripts::Transcript,
};

/// Run the malicious Sub prover for a single node.
///
/// Mirrors the honest [`Sub::prove`](crate::onnx_proof::ops::sub) — including
/// the clamp execution lookup and the read-address one-hot checks — but forges
/// the left operand opening as `left(r) + 1`, leaving the right operand honest.
/// The verifier's `left - right == acc` tie then necessarily fails.
pub fn malicious_sub_prove<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
    let (opening_point, _claim) =
        AccOpeningAccessor::new(&prover.accumulator, node).get_reduced_opening();

    let mut results = Vec::new();

    // (1) Honest clamp lookup: output(r) = SatClamp(acc(r)).
    let provider = ClampLookupProvider::new(node.clone());
    let (mut execution_sumcheck, lookup_indices) = provider.read_raf_prove::<F, T>(
        &prover.trace,
        &mut prover.accumulator,
        &mut prover.transcript,
        None,
    );
    let (execution_proof, _) = Sumcheck::prove(
        &mut execution_sumcheck,
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    results.push((ProofId(node.idx, ProofType::Execution), execution_proof));

    // (2) Honest one-hot checks on the clamp read-address polynomial.
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
    results.push((
        ProofId(node.idx, ProofType::RaOneHotChecks),
        ra_one_hot_proof,
    ));

    // (3) FORGED operand tie: claim `left(r) + 1` instead of `left(r)`.
    let LayerData { operands, .. } = Trace::layer_data(&prover.trace, node);
    let [left, right] = operands[..] else {
        panic!("Expected two operands for Sub operation")
    };
    let forged_left = MultilinearPolynomial::from(left.padded_next_power_of_two())
        .evaluate(&opening_point.r)
        + F::one();
    let right_claim =
        MultilinearPolynomial::from(right.padded_next_power_of_two()).evaluate(&opening_point.r);
    let mut provider = AccOpeningAccessor::new(&mut prover.accumulator, node)
        .into_provider(&mut prover.transcript, opening_point);
    provider.append_nodeio(Target::Input(0), forged_left);
    provider.append_nodeio(Target::Input(1), right_claim);

    results
}
