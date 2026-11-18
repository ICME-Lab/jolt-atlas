use atlas_algebra::field::JoltField;
pub use jolt_atlas_verifier::onnx_proof::ops::reshape::*;

use crate::node_view::ComputationNode;
use atlas_sumcheck::opening_proof::OpeningAccumulator;

/// The claim of a [`flat_copy`]: the operand at the output point, which is the output's claim
/// there, since the two are one flat vector.
pub fn flat_copy_claims<F: JoltField>(
    node: &ComputationNode,
    accumulator: &dyn OpeningAccumulator<F>,
) -> Vec<F> {
    let output = accumulator
        .get_node_output_opening(node.idx)
        .expect("the prover holds every claim it reads")
        .1;
    vec![output]
}
