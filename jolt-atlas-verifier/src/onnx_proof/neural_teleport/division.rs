//! Division helpers for neural teleportation (Cos/Sin's `a mod P` domain reduction).
//!
//! `a = P·q + r` holds exactly, elementwise, so as multilinear polynomials
//! `ã(X) = P·q̃(X) + r̃(X)` holds identically everywhere, not just on the boolean hypercube.
//! Once `q` and `r` each have an evaluation claim at the same point, deriving `a`'s claim is
//! pure field arithmetic — no sumcheck is needed to enforce the identity.

use crate::node_view::ComputationNode;
use crate::utils::opening_access::{AccOpeningAccessor, Target};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_sumcheck::opening_proof::VerifierOpeningAccumulator;
use atlas_transcript::Transcript;
use common::{CommittedPoly, VirtualPoly};

/// The teleport identity in the clear: the node's input, opened at the reduced output point,
/// is `P·q + r` over the quotient and remainder claims already bound there.
///
/// Unlike the prover, the verifier doesn't get to just compute and trust the input's claim —
/// it registers whatever the prover claimed for `Input(0)`, then independently recomputes
/// `P·q + r` from the (already separately bound) quotient and remainder claims and checks the
/// two agree.
pub struct TeleportInputRelation {
    /// The trig node.
    pub node: ComputationNode,
    /// The period `P`.
    pub tau: i32,
}

impl TeleportInputRelation {
    /// Append the input's opening and check the identity. `Err` if a claim is missing or the
    /// identity fails.
    pub fn verify<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        let accessor = AccOpeningAccessor::new(accumulator, &self.node);
        let (r_node_output, _) = accessor.get_reduced_opening()?;

        let mut provider = accessor.into_provider(transcript, r_node_output);
        provider.append_nodeio(Target::Input(0))?;
        let q_claim = provider.get_advice(CommittedPoly::TeleportNodeQuotient)?.1;
        let r_claim = provider.get_advice(VirtualPoly::TeleportRemainder)?.1;
        let expected = F::from_i32(self.tau) * q_claim + r_claim;
        let input_claim = provider.get_nodeio(Target::Input(0))?.1;

        if input_claim != expected {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Teleport input claim does not match P*q + r derived from the quotient and \
                 remainder claims"
                    .to_string(),
            ));
        }
        Ok(())
    }
}
