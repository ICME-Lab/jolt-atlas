use crate::node_view::ComputationNode;
use crate::onnx_proof::Verifier;
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_sumcheck::evaluation_reduction::{EvalReductionProtocol, ReducedInstance};
use atlas_transcript::Transcript;

/// Node-scoped evaluation reduction helper used by ONNX proving and verification.
pub struct NodeEvalReduction;

impl NodeEvalReduction {
    /// Reduce the claims on `computation_node`'s output to one. A single claim is the reduced
    /// instance and the proof carries no entry for the node; several are reduced with the
    /// node's evaluation-reduction proof.
    pub fn verify<F: JoltField, T: Transcript>(
        verifier: &mut Verifier<'_, F, T>,
        computation_node: &ComputationNode,
    ) -> Result<(), ProofVerifyError> {
        let node_idx = computation_node.idx;
        let openings = verifier.accumulator.get_node_openings(node_idx);

        let reduced_instance = match openings.as_slice() {
            [] => return Err(ProofVerifyError::EmptyInput),
            [(point, claim)] => ReducedInstance {
                r: point.r.clone(),
                claim: *claim,
            },
            _ => {
                let proof = verifier.eval_reductions.take(node_idx)?;
                EvalReductionProtocol::verify(&openings, proof, &mut verifier.transcript)?
            }
        };

        verifier
            .accumulator
            .reduced_evaluations
            .insert(node_idx, reduced_instance);

        Ok(())
    }
}
