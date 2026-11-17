//! The prover's side of the period reduction of Cos and Sin (`a mod P`).

pub use jolt_atlas_verifier::onnx_proof::neural_teleport::division::*;

use crate::node_view::ComputationNode;
use crate::utils::opening_access::AccOpeningAccessor;
use atlas_algebra::field::JoltField;
use atlas_sumcheck::opening_proof::OpeningAccumulator;
use common::{consts::TRIG_PERIOD_MODULUS, CommittedPoly, VirtualPoly};

/// The claim of a [`trig_period`]: the input at the reduced output point is `P·q + rem` over
/// the quotient and remainder claims held there.
pub fn trig_period_claims<F: JoltField>(
    node: &ComputationNode,
    accumulator: &dyn OpeningAccumulator<F>,
) -> Vec<F> {
    let expect = "the prover holds every claim it reads";
    let accessor = AccOpeningAccessor::new(accumulator, node);
    let quotient = accessor
        .get_advice(CommittedPoly::TeleportNodeQuotient)
        .expect(expect)
        .1;
    let remainder = accessor
        .get_advice(VirtualPoly::TeleportRemainder)
        .expect(expect)
        .1;
    vec![F::from_i32(TRIG_PERIOD_MODULUS as i32) * quotient + remainder]
}
