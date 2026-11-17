//! The period reduction of Cos and Sin (`a mod P`).
//!
//! `a = P·q + r` holds exactly, elementwise, so as multilinear polynomials
//! `ã(X) = P·q̃(X) + r̃(X)` holds identically everywhere, not just on the boolean hypercube.
//! Once `q` and `r` each have an evaluation claim at the same point, deriving `a`'s claim is
//! pure field arithmetic — no sumcheck is needed to enforce the identity.

use crate::node_view::ComputationNode;
use crate::onnx_proof::relation::{OpRel, OpRelKind, RelationOpening};
use crate::utils::opening_access::{OpeningIdBuilder, Target};
use atlas_algebra::field::JoltField;
use atlas_claims::{constant, opening, Int};
use common::{consts::TRIG_PERIOD_MODULUS, CommittedPoly, VirtualPoly};

/// The period reduction of a trig node as a tie at the reduced output point `r`: the node's
/// input, opened there, is `P·q + rem` over the quotient and remainder claims already bound
/// there.
///
/// ```text
/// input claim   P·q(r) + rem(r)        (the committed quotient, the remainder)
/// output claim  input(r)               (the operand, opened at r)
/// ```
///
/// **S ⊆ R.** `atlas_trace` reduces the input by the period `P` into a quotient and a
/// remainder with `input = P·q + rem` at every element, so the identity holds between the
/// three extensions at every point. The remainder's range is the downscale lookup's, outside
/// the relation.
pub fn trig_period<F: JoltField>(node: &ComputationNode) -> OpRel<F> {
    let ids = OpeningIdBuilder::new(node);
    let input = ids.nodeio(Target::Input(0));
    OpRel {
        degree: 1,
        input: constant(Int(i128::from(TRIG_PERIOD_MODULUS as i32)))
            * opening(RelationOpening::Id(
                ids.advice(CommittedPoly::TeleportNodeQuotient),
            ))
            + opening(RelationOpening::Id(
                ids.advice(VirtualPoly::TeleportRemainder),
            )),
        output: opening(RelationOpening::Id(input)),
        produced: vec![input],
        ..OpRel::base(OpRelKind::TrigPeriod, node)
    }
}
