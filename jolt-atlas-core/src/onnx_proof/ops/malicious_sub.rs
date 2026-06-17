//! Malicious variant of the Sub prover for soundness testing.
//!
//! The honest `Sub` proof is a no-sumcheck op (see [`crate::onnx_proof::ops::sub`]):
//! it opens both operands at the node's reduced output point `r` and the verifier
//! checks `left(r) - right(r) == output(r)` directly. This module forges the left
//! operand opening (off by one) so that `left(r) - right(r) != output(r)`, exercising
//! that the verifier's direct difference check rejects the attack. Used exclusively by
//! [`MaliciousONNXProof`](crate::onnx_proof::malicious_prover).

use crate::{
    onnx_proof::{ProofId, Prover},
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
};
use joltworks::{
    field::JoltField,
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
};

/// Run the malicious Sub prover for a single node.
///
/// Mirrors the honest no-sumcheck `Sub::prove` but forges the left operand
/// opening as `left(r) + 1`, leaving the right operand honest. The verifier's
/// `left - right == output` check then necessarily fails. Returns `vec![]` (no
/// execution proof), exactly like the honest op.
pub fn malicious_sub_prove<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
    let (opening_point, _claim) =
        AccOpeningAccessor::new(&prover.accumulator, node).get_reduced_opening();

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
    vec![]
}
