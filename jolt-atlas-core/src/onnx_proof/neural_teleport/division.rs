//! Division helpers for neural teleportation (Cos/Sin's `a mod P` domain reduction).
//!
//! `a = P·q + r` holds exactly, elementwise, so as multilinear polynomials
//! `ã(X) = P·q̃(X) + r̃(X)` holds identically everywhere, not just on the boolean hypercube.
//! Once `q` and `r` each have an evaluation claim at the same point, deriving `a`'s claim is
//! pure field arithmetic — no sumcheck is needed to enforce the identity.

use crate::onnx_proof::{Prover, Verifier};
use crate::utils::opening_access::{AccOpeningAccessor, Target};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    tensor::Tensor,
};
use common::{CommittedPoly, VirtualPoly};
use joltworks::{
    field::JoltField,
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

/// Computes quotient and remainder for neural teleportation division.
/// Returns (quotient, remainder) where input = DIVISOR * quotient + remainder
pub fn compute_division(input: &Tensor<i32>, tau: i32) -> (Tensor<i32>, Tensor<i32>) {
    let (remainder_data, quotient_data): (Vec<i32>, Vec<i32>) = input
        .iter()
        .map(|&x| {
            let mut r = x % tau;
            let mut q = x / tau;
            // Ensure remainder has same sign as divisor (Euclidean division)
            if (r < 0 && tau > 0) || (r > 0 && tau < 0) {
                r += tau;
                q -= 1;
            }
            (r, q)
        })
        .unzip();

    let quotient = Tensor::<i32>::construct(quotient_data, input.dims().to_vec());
    let remainder = Tensor::<i32>::construct(remainder_data, input.dims().to_vec());

    (quotient, remainder)
}

/// Caches the quotient's claim (`CommittedPoly::TeleportNodeQuotient`) at the node's
/// standard reduced output-opening point, by directly evaluating the honest quotient tensor.
/// Soundness comes from the global end-of-proof PCS opening of this committed polynomial,
/// exactly as for any other committed advice — this function itself proves nothing.
pub fn cache_teleport_quotient_prove<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
    tau: i32,
) {
    let LayerData { operands, .. } = Trace::layer_data(&prover.trace, node);
    let (quotient, _remainder) = compute_division(operands[0], tau);

    let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
    let (r_node_output, _) = accessor.get_reduced_opening();
    let q_eval = MultilinearPolynomial::from(quotient).evaluate(&r_node_output.r);

    accessor
        .into_provider(&mut prover.transcript, r_node_output)
        .append_advice(CommittedPoly::TeleportNodeQuotient, q_eval);
}

/// Verifier counterpart of [`cache_teleport_quotient_prove`].
pub fn cache_teleport_quotient_verify<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) {
    let accessor = AccOpeningAccessor::new(&mut verifier.accumulator, node);
    let (r_node_output, _) = accessor.get_reduced_opening();

    accessor
        .into_provider(&mut verifier.transcript, r_node_output)
        .append_advice(CommittedPoly::TeleportNodeQuotient);
}

/// Derives the trig node's input claim `a = P·q + r` from the already-established `q` and
/// `r` claims and caches it as the node's `Input(0)` claim (i.e. the upstream node's output
/// claim at this point). Must run after both `cache_teleport_quotient_prove`/
/// `cache_teleport_quotient_verify` and the remainder-establishing downscale lookup have run
/// for this node.
pub fn cache_teleport_input_claim_prove<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
    tau: i32,
) {
    let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
    let (r_node_output, _) = accessor.get_reduced_opening();
    let q_claim = accessor.get_advice(CommittedPoly::TeleportNodeQuotient).1;
    let r_claim = accessor.get_advice(VirtualPoly::TeleportRemainder).1;
    let a_claim = F::from_i32(tau) * q_claim + r_claim;

    accessor
        .into_provider(&mut prover.transcript, r_node_output)
        .append_nodeio(Target::Input(0), a_claim);
}

/// Verifier counterpart of [`cache_teleport_input_claim_prove`]. Unlike the prover, the verifier
/// doesn't get to just compute and trust `a_claim` — it registers whatever the prover
/// claimed for `Input(0)`, then independently recomputes `P·q + r` from the (already
/// separately bound) quotient and remainder claims and checks the two agree.
pub fn verify_teleport_input_claim<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
    tau: i32,
) -> Result<(), ProofVerifyError> {
    let accessor = AccOpeningAccessor::new(&mut verifier.accumulator, node);
    let (r_node_output, _) = accessor.get_reduced_opening();

    let mut provider = accessor.into_provider(&mut verifier.transcript, r_node_output);
    provider.append_nodeio(Target::Input(0));
    let q_claim = provider.get_advice(CommittedPoly::TeleportNodeQuotient).1;
    let r_claim = provider.get_advice(VirtualPoly::TeleportRemainder).1;
    let expected = F::from_i32(tau) * q_claim + r_claim;
    let input_claim = provider.get_nodeio(Target::Input(0)).1;

    if input_claim != expected {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "Teleport input claim does not match P*q + r derived from the quotient and \
             remainder claims"
                .to_string(),
        ));
    }
    Ok(())
}
