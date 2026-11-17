use atlas_algebra::field::JoltField;
pub use jolt_atlas_verifier::onnx_proof::ops::neg::*;

use crate::node_view::ComputationNode;
use crate::onnx_proof::relation::OpRelKind;
use atlas_sumcheck::opening_proof::OpeningAccumulator;

/// The claim of a [`signed_copy`]: `operand(r) = ±out(r)`, the reduced output claim, signed.
pub fn signed_copy_claims<F: JoltField>(
    node: &ComputationNode,
    kind: OpRelKind,
    accumulator: &dyn OpeningAccumulator<F>,
) -> Vec<F> {
    let output = accumulator
        .get_node_output_opening(node.idx)
        .expect("the prover holds every claim it reads")
        .1;
    vec![F::from_i64(sign(kind)) * output]
}
