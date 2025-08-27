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
use onnx_tracer::trace_types::{ONNXInstr, ONNXOpcode};
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
#[derive(Clone, Debug)]
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
            .filter_map(|instr| match instr.opcode {
                ONNXOpcode::MatMult => {
                    let m = instr.output_dims[0].next_power_of_two();
                    let n = instr.output_dims[1].next_power_of_two();
                    let k = instr
                        .imm
                        .as_ref()
                        .map(|imm| imm.dims()[1])
                        .unwrap_or(1)
                        .next_power_of_two();
                    Some((m, n, k))
                }
                _ => None,
            })
            .collect_vec();
        Self {
            mat_mult_precompile_dims,
        }
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
    /// Given the execution trace, construct the polynomials used in the batched sum-check proof.
    /// The witness polynomials are abstracted as `MatMultSumcheck` instances, which hold a MatMultProverState which contains the witness polynomials `a` & `b` for the matrix multiplication precompile.
    ///
    /// # Note
    /// - We require the `transcript` to generate the challenges for the matrix multiplication precompile.
    pub fn generate_witness<InstructionSet>(
        ops: &[JoltONNXCycle],
        transcript: &mut ProofTranscript,
    ) -> Vec<MatMultSumcheck<F>> {
        // Filter the operations to only include those that are proven with precompiles.
        // For each precompile operator, initialize the prover state and create a new `MatMultSumcheck`.
        ops.iter()
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
            .collect_vec()
    }

    /// Run the precompile sum-check instances through [`BatchedSumcheck::prove`] protcol.
    #[tracing::instrument(skip_all, name = "PrecompileProof::prove")]
    pub fn prove(
        _pp: &PrecompilePreprocessing,
        witness: &mut [MatMultSumcheck<F>],
        transcript: &mut ProofTranscript,
    ) -> Self {
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
        Self {
            sumcheck_proof,
            init_claims,
            final_claims,
        }
    }

    /// Verify the sum-check precompile instances via [`BatchedSumcheck::verify`].
    #[tracing::instrument(skip_all, name = "PrecompileProof::verify")]
    pub fn verify(
        pp: &PrecompilePreprocessing,
        proof: &Self,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let vsumcheck_instances =
            Self::initialize_verifier(pp, &proof.init_claims, &proof.final_claims, transcript);
        let trait_objects: Vec<&dyn BatchableSumcheckInstance<F, ProofTranscript>> =
            vsumcheck_instances
                .iter()
                .map(|p| p as &dyn BatchableSumcheckInstance<F, ProofTranscript>)
                .collect();
        let _ = BatchedSumcheck::verify(&proof.sumcheck_proof, trait_objects, transcript)?;
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
    use crate::jolt::precompiles::matmult::{
        MatMultPrecompile, MatMultPrecompileDims, MatMultProverState, MatMultSumcheck,
    };

    use super::{PrecompilePreprocessing, PrecompileProof};
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use jolt_core::utils::transcript::{KeccakTranscript, Transcript};
    use rand_core::RngCore;

    // #[test]
    // fn test_precompile_proof() {
    //     let mut rng = test_rng();
    //     let mut program = ONNXProgram::new("onnx/mlp/perceptron_2.onnx", None);
    //     let pp = PrecompilePreprocessing::preprocess(&program.decode());
    //     let input = random_floatvec(&mut rng, 4);
    //     program.set_input(input);

    //     // Prover
    //     let (_io, trace) = program.trace();
    //     let mut ptranscript = KeccakTranscript::new(b"test");
    //     let mut witness = PrecompileProof::<Fr, _>::generate_witness(&trace, &mut ptranscript);
    //     assert!(!witness.is_empty());
    //     let proof = PrecompileProof::<Fr, _>::prove(&pp, &mut witness, &mut ptranscript);

    //     // Verifier
    //     let mut vtranscript = KeccakTranscript::new(b"test");
    //     PrecompileProof::<Fr, _>::verify(&pp, &proof, &mut vtranscript).unwrap();
    // }

    #[test]
    fn test_random_execution_trace() {
        let mut rng = test_rng();
        let trace_length = 100;
        let mut pp: Vec<MatMultPrecompileDims> = Vec::with_capacity(trace_length);
        let mut ptranscript = KeccakTranscript::new(b"test");
        let mut sumcheck_instances = Vec::with_capacity(trace_length);
        for _ in 0..trace_length {
            let precompile = MatMultPrecompile::random(&mut rng);
            pp.push(precompile.dims());
            let prover_state = MatMultProverState::<Fr>::initialize(&precompile, &mut ptranscript);
            let sumcheck_instance = MatMultSumcheck::new(Some(prover_state), None, None);
            sumcheck_instances.push(sumcheck_instance);
        }

        // Preprocessing
        let pp = PrecompilePreprocessing {
            mat_mult_precompile_dims: pp,
        };

        // Prover
        let proof = PrecompileProof::<Fr, _>::prove(&pp, &mut sumcheck_instances, &mut ptranscript);

        // Verifier
        let mut vtranscript = KeccakTranscript::new(b"test");
        PrecompileProof::<Fr, _>::verify(&pp, &proof, &mut vtranscript)
            .expect("Verification failed");
    }
}
