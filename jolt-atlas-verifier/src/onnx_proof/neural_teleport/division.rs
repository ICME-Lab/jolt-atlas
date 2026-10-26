//! Division helpers for neural teleportation (Cos/Sin's `a mod P` domain reduction).
//!
//! `a = P·q + r` holds exactly, elementwise, so as multilinear polynomials
//! `ã(X) = P·q̃(X) + r̃(X)` holds identically everywhere, not just on the boolean hypercube.
//! Once `q` and `r` each have an evaluation claim at the same point, deriving `a`'s claim is
//! pure field arithmetic — no sumcheck is needed to enforce the identity.

use atlas_trace::Tensor;

use crate::node_view::ComputationNode;
use crate::onnx_proof::Verifier;
use crate::utils::opening_access::{AccOpeningAccessor, Target};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_transcript::Transcript;
use common::{CommittedPoly, VirtualPoly};

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
