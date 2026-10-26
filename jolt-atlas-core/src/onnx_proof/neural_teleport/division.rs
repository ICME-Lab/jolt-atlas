//! Division helpers for neural teleportation (Cos/Sin's `a mod P` domain reduction).
//!
//! `a = P·q + r` holds exactly, elementwise, so as multilinear polynomials
//! `ã(X) = P·q̃(X) + r̃(X)` holds identically everywhere, not just on the boolean hypercube.
//! Once `q` and `r` each have an evaluation claim at the same point, deriving `a`'s claim is
//! pure field arithmetic — no sumcheck is needed to enforce the identity.

use atlas_trace::trace::LayerData;
pub use jolt_atlas_verifier::onnx_proof::neural_teleport::division::*;

use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use crate::onnx_proof::Prover;
use crate::utils::opening_access::{AccOpeningAccessor, Target};
use atlas_algebra::{
    field::JoltField,
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
};
use atlas_transcript::Transcript;
use common::{CommittedPoly, VirtualPoly};

/// Caches the quotient's claim (`CommittedPoly::TeleportNodeQuotient`) at the node's
/// standard reduced output-opening point, by directly evaluating the honest quotient tensor.
/// Soundness comes from the global end-of-proof PCS opening of this committed polynomial,
/// exactly as for any other committed advice — this function itself proves nothing.
pub fn cache_teleport_quotient_prove<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
    tau: i32,
) {
    let LayerData { operands, .. } = TraceView::layer_data(&prover.trace, node);
    let (quotient, _remainder) = compute_division(operands[0], tau);

    let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
    let (r_node_output, _) = accessor.get_reduced_opening();
    let q_eval = MultilinearPolynomial::from(quotient).evaluate(&r_node_output.r);

    accessor
        .into_provider(&mut prover.transcript, r_node_output)
        .append_advice(CommittedPoly::TeleportNodeQuotient, q_eval);
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
