//! Verifier state and internal verification helpers for [`ONNXProof`].
//!
//! The public entry point is [`ONNXProof::verify`] (defined in the parent module).
//! This file houses the [`Verifier`] struct and the private helper methods that
//! `verify` delegates to.

use super::{types::ProofId, AtlasSharedPreprocessing, AtlasVerifierPreprocessing, ONNXProof};
use crate::onnx_proof::ops::OperatorVerifier;
use atlas_onnx_tracer::model::{trace::ModelExecutionIO, Model};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, SumcheckId, VerifierOpeningAccumulator,
        },
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::AppendToTranscript,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Verifier state
// ---------------------------------------------------------------------------

/// Verifier state that owns all data needed during verification.
/// Created once before the verification loop and passed to operator handlers.
pub struct Verifier<'a, F: JoltField, T: Transcript> {
    /// Shared preprocessing data (model structure).
    pub preprocessing: &'a AtlasSharedPreprocessing,
    /// Opening accumulator for batching polynomial openings.
    pub accumulator: VerifierOpeningAccumulator<F>,
    /// Interactive proof transcript.
    pub transcript: T,
    /// Map of proof IDs to sumcheck proofs.
    pub proofs: &'a BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
    /// Model execution inputs and outputs.
    pub io: &'a ModelExecutionIO,
}

impl<'a, F: JoltField, T: Transcript> Verifier<'a, F, T> {
    /// Create a new verifier with the given preprocessing, proofs, and IO
    pub fn new(
        preprocessing: &'a AtlasSharedPreprocessing,
        proofs: &'a BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
        io: &'a ModelExecutionIO,
    ) -> Self {
        Self {
            preprocessing,
            accumulator: VerifierOpeningAccumulator::new(),
            transcript: T::new(b"ONNXProof"),
            proofs,
            io,
        }
    }

    /// Verify and apply pre-node NodeOutput evaluation reduction (2-to-1 only).
    pub(super) fn perform_eval_reduction(
        &mut self,
        computation_node: &atlas_onnx_tracer::node::ComputationNode,
    ) -> Result<(), ProofVerifyError> {
        let producer_idx = computation_node.idx;
        let lo = OpeningId::Virtual(
            VirtualPolynomial::NodeOutput(producer_idx),
            SumcheckId::NodeExecution(producer_idx + 1),
        );
        let hi = OpeningId::Virtual(
            VirtualPolynomial::NodeOutput(producer_idx),
            SumcheckId::NodeExecution(usize::MAX),
        );
        let candidate_keys: Vec<OpeningId> = self
            .accumulator
            .openings
            .range(lo..=hi)
            .map(|(k, _)| *k)
            .collect();

        if candidate_keys.len() != 2 {
            return Ok(());
        }

        let h = self
            .accumulator
            .eval_reduction_h_polys
            .get(&candidate_keys[0])
            .or_else(|| {
                self.accumulator
                    .eval_reduction_h_polys
                    .get(&candidate_keys[1])
            })
            .ok_or_else(|| {
                ProofVerifyError::InvalidOpeningProof(
                    "missing evaluation-reduction h polynomial for node-output opening".to_string(),
                )
            })?;

        h.append_to_transcript(&mut self.transcript);
        let _ = self.transcript.challenge_scalar_optimized::<F>();

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Internal verifier helpers on ONNXProof
// ---------------------------------------------------------------------------

impl<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> ONNXProof<F, T, PCS> {
    /// Populate the verifier accumulator with opening claims and
    /// commitments from the proof.
    pub(super) fn populate_accumulator(&self, verifier: &mut Verifier<'_, F, T>) {
        // Load all opening claims from the proof (NodeOutput+Execution claims
        // are now stored directly in opening_claims alongside everything else).
        for (key, (_, claim)) in &self.opening_claims.0 {
            verifier
                .accumulator
                .openings
                .insert(*key, (OpeningPoint::default(), *claim));
        }
        verifier.accumulator.eval_reduction_h_polys = self.eval_reduction_h_polys.clone();

        for commitment in &self.commitments {
            verifier.transcript.append_serializable(commitment);
        }
    }

    /// Verify that the output MLE evaluates correctly at the random challenge point τ.
    pub(super) fn verify_output_claim(
        model: &Model,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let output_index = model.outputs()[0];
        let output_computation_node = &model[output_index];
        let r_node_output = verifier.transcript.challenge_vector_optimized::<F>(
            output_computation_node
                .pow2_padded_num_output_elements()
                .log_2(),
        );
        let expected_output_claim =
            MultilinearPolynomial::from(verifier.io.outputs[0].padded_next_power_of_two())
                .evaluate(&r_node_output);

        // append_virtual now handles both transcript append and opening point update.
        // The claim was loaded from opening_claims in populate_accumulator.
        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            VirtualPolynomial::NodeOutput(output_computation_node.idx),
            // NodeOutput claims are generally produced by subsequent nodes during proving; emulate that here.
            SumcheckId::NodeExecution(output_computation_node.idx + 1),
            r_node_output.clone().into(),
        );
        // Read the prover's claimed value and compare against IO.
        let output_claim = verifier
            .accumulator
            .get_node_output_opening(output_computation_node.idx)
            .1;
        if expected_output_claim != output_claim {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Expected output claim does not match actual output claim".to_string(),
            ));
        }
        Ok(())
    }

    /// Iterate over computation graph in reverse topological order and verify each operation.
    #[tracing::instrument(skip_all, name = "ONNXProof::verify_iop")]
    pub(super) fn verify_iop(
        model: &Model,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        for (_, computation_node) in model.graph.nodes.iter().rev() {
            // Before each node is proven,
            // perform eval reduction to reduce openings to a unique claim for the node output.

            // verifier.perform_eval_reduction(computation_node)?;
            let res = OperatorVerifier::verify(computation_node, verifier);
            #[cfg(test)]
            {
                if let Err(e) = &res {
                    println!("Verification failed at node {computation_node:#?}: {e:?}");
                }
            }
            res?;
        }
        Ok(())
    }

    /// Verify the reduced opening proof (sumcheck reduction + PCS verification).
    #[tracing::instrument(skip_all, name = "ONNXProof::verify_reduced_openings")]
    pub(super) fn verify_reduced_openings(
        &self,
        pp: &AtlasVerifierPreprocessing<F, PCS>,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        if let Some(reduced_opening_proof) = &self.reduced_opening_proof {
            verifier
                .accumulator
                .prepare_for_sumcheck(&reduced_opening_proof.sumcheck_claims);

            let reduction_res = verifier.accumulator.verify_batch_opening_sumcheck(
                &reduced_opening_proof.sumcheck_proof,
                &mut verifier.transcript,
            );
            #[cfg(test)]
            {
                if let Err(e) = &reduction_res {
                    println!("Opening reduction via sumcheck failed: {e:?}");
                }
            }
            let r_sumcheck = reduction_res?;

            let verifier_state = verifier.accumulator.finalize_batch_opening_sumcheck(
                r_sumcheck,
                &reduced_opening_proof.sumcheck_claims,
                &mut verifier.transcript,
            );

            let joint_commitment =
                PCS::combine_commitments(&self.commitments, &verifier_state.gamma_powers);

            verifier.accumulator.verify_joint_opening::<_, PCS>(
                &pp.generators,
                &reduced_opening_proof.joint_opening_proof,
                &joint_commitment,
                &verifier_state,
                &mut verifier.transcript,
            )?;
        } else {
            let committed_polys = pp.shared.get_models_committed_polynomials::<F, T>();
            if !committed_polys.is_empty() {
                return Err(ProofVerifyError::MissingReductionProof);
            }
        }
        Ok(())
    }
}
