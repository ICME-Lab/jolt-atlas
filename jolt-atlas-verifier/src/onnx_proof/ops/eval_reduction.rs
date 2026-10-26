use crate::node_view::ComputationNode;
use crate::onnx_proof::Verifier;
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_sumcheck::evaluation_reduction::{EvalReductionProof, EvalReductionProtocol};
use atlas_transcript::Transcript;

/// Node-scoped evaluation reduction helper used by ONNX proving and verification.
pub struct NodeEvalReduction;

impl NodeEvalReduction {
    /// Run node-output evaluation reduction on the verifier side for the given node.
    pub fn verify<F: JoltField, T: Transcript>(
        verifier: &mut Verifier<'_, F, T>,
        computation_node: &ComputationNode,
        eval_reduction_proof: &EvalReductionProof<F>,
    ) -> Result<(), ProofVerifyError> {
        let node_idx = computation_node.idx;
        let openings = verifier.accumulator.get_node_openings(node_idx);

        let reduced_instance = EvalReductionProtocol::verify(
            &openings,
            eval_reduction_proof,
            &mut verifier.transcript,
        )?;

        verifier
            .accumulator
            .reduced_evaluations
            .insert(node_idx, reduced_instance);

        Ok(())
    }
}
