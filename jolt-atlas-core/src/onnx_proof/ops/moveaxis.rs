use atlas_algebra::field::JoltField;
pub use jolt_atlas_verifier::onnx_proof::ops::moveaxis::*;

use crate::node_view::ComputationNode;
use atlas_sumcheck::opening_proof::OpeningAccumulator;

/// The claim of an [`axis_permutation`]: the operand at the permuted point is the output at
/// `r`, the reduced output claim.
pub fn axis_permutation_claims<F: JoltField>(
    node: &ComputationNode,
    accumulator: &dyn OpeningAccumulator<F>,
) -> Vec<F> {
    vec![
        accumulator
            .get_node_output_opening(node.idx)
            .expect("the prover holds every claim it reads")
            .1,
    ]
}
