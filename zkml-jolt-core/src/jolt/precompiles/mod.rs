//! Precompile proofs for ONNX runtime.
//!
//! This module provides a custom SNARK for precompiles in the [`ONNXJoltVM`].
//! These precompile proofs are sum-check based.

use itertools::Itertools;
use jolt_core::{
    field::JoltField,
    subprotocols::sumcheck::{BatchableSumcheckInstance, BatchedSumcheck, SumcheckInstanceProof},
    utils::{errors::ProofVerifyError, transcript::Transcript},
};
use onnx_tracer::trace_types::ONNXInstr;
use serde::{Deserialize, Serialize};

use crate::jolt::{
    execution_trace::JoltONNXCycle,
    precompiles::matmult::{
        MatMultClaims, MatMultPrecompile, MatMultPrecompileDims, MatMultProverState,
        MatMultSumcheck, MatMultVerifierState,
    },
};
pub mod matmult;

/// Specifies the ONNX precompile operators used in the Jolt ONNX VM.
/// Used to specifiy the precompile type and its input's in the [`JoltONNXTraceStep`]
#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum PrecompileOp {
    /// Matrix multiplication precompile.
    MatMult(MatMultPrecompile),
}

/// Preprocessing of the models matrices for the precompile proof.
/// Store the dimensions of the matrix multiplication precompile.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrecompilePreprocessing {
    /// The dimensions used in the matrix multiplication precompile's.
    pub mat_mult_precompile_dims: Vec<MatMultPrecompileDims>,
}

impl PrecompilePreprocessing {
    /// Preprocess the ONNX model to extract the dimensions of the matrix multiplication precompile.
    #[tracing::instrument(skip_all, name = "PrecompilePreprocessing::preprocess")]
    pub fn preprocess(instrs: &[ONNXInstr]) -> Self {
        // For each matmult instruction store the [`MatMultPrecompileDims`]
        // We pad the dimensions to the next power of two.
        let mat_mult_precompile_dims = instrs
            .iter()
            .filter_map(|instr| instr.matmult_dims())
            .collect_vec();
        Self {
            mat_mult_precompile_dims,
        }
    }

    /// Check if no precompiles
    pub fn is_empty(&self) -> bool {
        self.mat_mult_precompile_dims.is_empty()
    }
}

/// A special-purpose SNARK designed for specific functionality, such as ONNX operators that are more efficient to prove using a sum-check precompile than an [`InstructionLookupProof`].
/// This is a sum-check-based precompile proof tailored for ONNX runtime.
/// It is used to prove the correctness of certain ONNX operators via a custom sum-check precompile instead of a lookup-based approach.
pub struct PrecompileProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    init_claims: Vec<F>,
    final_claims: Vec<MatMultClaims<F>>,
}

impl<F, ProofTranscript> PrecompileProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    /// Run the precompile sum-check instances through [`BatchedSumcheck::prove`] protcol.
    #[tracing::instrument(skip_all, name = "PrecompileProof::prove")]
    pub fn prove(
        pp: &PrecompilePreprocessing,
        execution_trace: &[JoltONNXCycle],
        transcript: &mut ProofTranscript,
    ) -> Option<Self> {
        if pp.is_empty() {
            return None;
        }
        // Given the execution trace, construct the polynomials used in the batched sum-check proof.
        // The witness polynomials are abstracted as `MatMultSumcheck` instances, which hold a MatMultProverState which contains the witness polynomials `a` & `b` for the matrix multiplication precompile.
        //
        // # Note
        // - We require the `transcript` to generate the challenges for the matrix multiplication precompile.
        //
        // Filter the operations to only include those that are proven with precompiles.
        // For each precompile operator, initialize the prover state and create a new `MatMultSumcheck`.
        let mut witness = execution_trace
            .iter()
            .filter_map(|op| match &op.precompile {
                Some(PrecompileOp::MatMult(mat_mult)) => {
                    // Initialize the prover state for the matrix multiplication precompile.
                    // `MatMultProverState::initialize` constructs the witness polynomials `a` & `b` for the matrix multiplication precompile.
                    // It takes the `transcript` as an argument to generate the challenges, rx & ry to compute the evaluation for Sum_k A(rx, k) & B(ry, k)
                    let prover_state: MatMultProverState<F> =
                        MatMultProverState::initialize(mat_mult, transcript);

                    // Create a new `MatMultSumcheck` instance with the prover state.
                    Some(MatMultSumcheck::new(Some(prover_state), None, None))
                }
                _ => None,
            })
            .collect_vec();
        let init_claims = witness
            .iter()
            .map(|p| p.prover_state.as_ref().unwrap().input_claim)
            .collect_vec();
        let trait_objects: Vec<&mut dyn BatchableSumcheckInstance<F, ProofTranscript>> = witness
            .iter_mut()
            .map(|p| p as &mut dyn BatchableSumcheckInstance<F, ProofTranscript>)
            .collect();
        let (sumcheck_proof, _rsc) = BatchedSumcheck::prove(trait_objects, transcript);
        let final_claims = witness
            .iter()
            .map(|p| p.claims.as_ref().unwrap().clone())
            .collect_vec(); // TODO: Append these claims to opening accumulator
        Some(Self {
            sumcheck_proof,
            init_claims,
            final_claims,
        })
    }

    /// Verify the sum-check precompile instances via [`BatchedSumcheck::verify`].
    #[tracing::instrument(skip_all, name = "PrecompileProof::verify")]
    pub fn verify(
        &self,
        pp: &PrecompilePreprocessing,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let vsumcheck_instances =
            Self::initialize_verifier(pp, &self.init_claims, &self.final_claims, transcript);
        let trait_objects: Vec<&dyn BatchableSumcheckInstance<F, ProofTranscript>> =
            vsumcheck_instances
                .iter()
                .map(|p| p as &dyn BatchableSumcheckInstance<F, ProofTranscript>)
                .collect();
        let _ = BatchedSumcheck::verify(&self.sumcheck_proof, trait_objects, transcript)?;
        Ok(())
    }

    /// Initialize the verifier states for the precompile sum-check instances.
    /// Updates the transcript to be in sync with the prover's transcript.
    ///
    /// # Panics
    /// Panics if the length of `init_claims` and `final_claims` does not match the number of matrix multiplication precompile's
    fn initialize_verifier(
        pp: &PrecompilePreprocessing,
        init_claims: &[F],
        final_claims: &[MatMultClaims<F>],
        transcript: &mut ProofTranscript,
    ) -> Vec<MatMultSumcheck<F>> {
        let dims = &pp.mat_mult_precompile_dims;
        dims.iter()
            .zip_eq(init_claims.iter())
            .zip_eq(final_claims.iter())
            .map(|((dim, init_claim), final_claim)| {
                // Initialize verifier state. We update transcript state as well, generating the challenges rx & ry & appending init_claim
                // for the matmult sum-check precompile proof.
                let verifier_state =
                    MatMultVerifierState::initialize(dim.0, dim.1, dim.2, *init_claim, transcript);

                // Create a new `MatMultSumcheck` instance with the verifier state and final claims.
                MatMultSumcheck::new(None, Some(verifier_state), Some(final_claim.clone()))
            })
            .collect_vec()
    }
}

#[cfg(test)]
mod tests {
    use crate::jolt::execution_trace::jolt_execution_trace;

    use super::{PrecompilePreprocessing, PrecompileProof};
    use ark_bn254::Fr;
    use jolt_core::utils::transcript::{KeccakTranscript, Transcript};
    use onnx_tracer::{builder, decode_model, tensor::Tensor};

    #[test]
    fn test_precompile_proof() {
        let matmult_model = builder::simple_matmult_model();
        let program = decode_model(matmult_model.clone());
        let pp = PrecompilePreprocessing::preprocess(&program);
        let input = vec![1, 2, 3, 4];

        // Prover
        let (raw_trace, _) = onnx_tracer::execution_trace(
            matmult_model,
            &Tensor::new(Some(&input), &[1, 4]).unwrap(),
        );
        let execution_trace = jolt_execution_trace(raw_trace.clone());

        let mut prover_transcript = KeccakTranscript::new(b"test");
        let proof =
            PrecompileProof::<Fr, _>::prove(&pp, &execution_trace, &mut prover_transcript).unwrap();

        // Verifier
        let mut verifier_transcript = KeccakTranscript::new(b"test");
        proof.verify(&pp, &mut verifier_transcript).unwrap();
    }
}
