use crate::onnx_proof::Verifier;
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_sumcheck::evaluation_reduction::{EvalReductionProtocol, ReducedInstance};
use atlas_transcript::VerifyingTranscript;

/// Node-scoped evaluation reduction helper used by ONNX proving and verification.
pub struct NodeEvalReduction;

impl NodeEvalReduction {
    /// Reduce the claims on node `node_idx`'s output to one. A single claim is the reduced
    /// instance and the proof carries no entry for the node; several are reduced with the
    /// node's evaluation-reduction proof.
    pub fn verify<F: JoltField, T: VerifyingTranscript>(
        verifier: &mut Verifier<'_, F, T>,
        node_idx: usize,
    ) -> Result<(), ProofVerifyError> {
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
