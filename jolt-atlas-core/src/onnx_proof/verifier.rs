//! Verifier state and internal verification helpers for [`ONNXProof`].
//!
//! The public entry point is [`ONNXProof::verify`] (defined in the parent module).
//! This file houses the [`Verifier`] struct and the private helper methods that
//! `verify` delegates to.

use super::{types::ProofId, AtlasSharedPreprocessing, AtlasVerifierPreprocessing, ONNXProof};
use crate::onnx_proof::ops::OperatorVerifier;
use atlas_onnx_tracer::{
    model::{trace::ModelExecutionIO, Model},
    node::ComputationNode,
};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{OpeningAccumulator, OpeningPoint, SumcheckId, VerifierOpeningAccumulator},
    },
    subprotocols::{
        evaluation_reduction::{EvalReductionProof, EvalReductionProtocol},
        sumcheck::SumcheckInstanceProof,
    },
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
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(output_computation_node.idx),
                SumcheckId::NodeExecution(output_computation_node.idx + 1),
            )
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
        &self,
        model: &Model,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        for (_, node) in model.graph.nodes.iter().rev() {
            // Before each node is proven,
            // perform eval reduction to reduce openings to a unique claim for the node output.

            let eval_red_res =
                EvalReductionVerifier::verify(verifier, node, &self.eval_reduction_proofs);
            #[cfg(test)]
            if let Err(e) = &eval_red_res {
                println!("Verification failed at node {node:#?}: {e:?}");
            }
            eval_red_res?;
            let res = OperatorVerifier::verify(node, verifier);
            #[cfg(test)]
            if let Err(e) = &res {
                println!("Verification failed at node {node:#?}: {e:?}");
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

struct EvalReductionVerifier;

impl EvalReductionVerifier {
    /// Verify and apply pre-node NodeOutput evaluation reduction (2-to-1 only).
    ///
    /// # Assumptions
    /// - The openings mapping only has entries for legitimate claims made by the prover.
    ///   The prover might fake those claims, which will be caught, but it cannot add arbitrary claims.
    ///   Concretely, the verifier only fills the opening points for legitimately cached openings,
    ///   so if the prover tries to fake extra claims, those will also get caught due to still having empty opening point.
    // TODO: Verify this assumption
    pub(super) fn verify<F: JoltField, T: Transcript>(
        verifier: &mut Verifier<'_, F, T>,
        computation_node: &ComputationNode,
        eval_reduction_proofs: &BTreeMap<usize, EvalReductionProof<F>>,
    ) -> Result<(), ProofVerifyError> {
        let node_idx = computation_node.idx;
        let eval_reduction_proof = eval_reduction_proofs.get(&node_idx).ok_or_else(|| {
            ProofVerifyError::InvalidOpeningProof(format!(
                "Missing evaluation reduction proof for node index {node_idx}"
            ))
        })?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_proof::{AtlasSharedPreprocessing, Claims, HyperKZG, ONNXProof, Prover};
    use ark_bn254::{Bn254, Fr};
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, trace::Trace, Model},
        ops::{Add, Operator},
        tensor::Tensor,
    };
    use common::VirtualPolynomial;
    use joltworks::{
        poly::{
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            opening_proof::{OpeningId, OpeningPoint, SumcheckId},
        },
        subprotocols::evaluation_reduction::EvalReductionProtocol,
        transcripts::Blake2bTranscript,
        utils::errors::ProofVerifyError,
    };
    use rand::{rngs::StdRng, SeedableRng};
    use std::collections::BTreeMap;

    // All node output is used 1 to 2 times.
    fn one_sub_model(rng: &mut StdRng, t: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![t]);
        let c = b.constant(Tensor::random_small(rng, &[t]));
        let z = b.sub(i, c);
        let out = b.sub(z, c);
        b.mark_output(out);
        b.build()
    }

    #[test]
    fn eval_reduction_verifier_rejects_missing_artifact() {
        let n = 1 << 3; // 8 points, i.e. 3 variables
        let model = Model::default();
        let producer_idx = 0;

        let preprocessing = AtlasSharedPreprocessing::preprocess(model);
        let trace = preprocessing.model().trace(&[]);
        let io = Trace::io(&trace, preprocessing.model());

        let proofs = BTreeMap::new();
        let mut verifier = Verifier::<Fr, Blake2bTranscript>::new(&preprocessing, &proofs, &io);

        let producer_node = ComputationNode::new(producer_idx, Operator::Add(Add), vec![], vec![n]);
        let eval_reduction_proofs = BTreeMap::new();

        let err =
            EvalReductionVerifier::verify(&mut verifier, &producer_node, &eval_reduction_proofs)
                .expect_err("verification should fail when eval-reduction artifact is missing");

        assert!(matches!(
            err,
            ProofVerifyError::InvalidOpeningProof(msg)
            if msg.contains("Missing evaluation reduction proof for node index")
        ));
    }

    #[test]
    fn eval_reduction_verifier_single_opening_is_propagated() {
        let n = 1 << 3; // 8 points, i.e. 3 variables
        let output: Tensor<i32> = Tensor::construct((0..n).map(|i| i as i32).collect(), vec![n]);
        let model = Model::default();
        let producer_idx = 0;
        let consumer = 1;

        let preprocessing = AtlasSharedPreprocessing::preprocess(model);
        let trace = preprocessing.model().trace(&[]);
        let io = Trace::io(&trace, preprocessing.model());

        let proofs = BTreeMap::new();
        let mut verifier = Verifier::<Fr, Blake2bTranscript>::new(&preprocessing, &proofs, &io);

        let producer_node = ComputationNode::new(producer_idx, Operator::Add(Add), vec![], vec![n]);

        let output_mle = MultilinearPolynomial::from(output.padded_next_power_of_two());
        let num_vars = output_mle.get_num_vars();

        let point = verifier
            .transcript
            .challenge_vector_optimized::<Fr>(num_vars);
        let claim = output_mle.evaluate(&point);
        let key = OpeningId::Virtual(
            VirtualPolynomial::NodeOutput(producer_idx),
            SumcheckId::NodeExecution(consumer),
        );
        let opening = (OpeningPoint::new(point.clone()), claim);
        verifier.accumulator.openings.insert(key, opening);

        let mut prover_transcript = Blake2bTranscript::new(b"ONNXProof");
        let (proof, _reduced_prover) = EvalReductionProtocol::prove(
            &verifier.accumulator.get_node_openings(producer_idx),
            output_mle,
            &mut prover_transcript,
        )
        .expect("single opening should be propagated by prover-side protocol");

        let mut eval_reduction_proofs = BTreeMap::new();
        eval_reduction_proofs.insert(producer_idx, proof);

        EvalReductionVerifier::verify(&mut verifier, &producer_node, &eval_reduction_proofs)
            .expect("single opening should be accepted and propagated by verifier");

        let reduced = verifier
            .accumulator
            .reduced_evaluations
            .get(&producer_idx)
            .expect("single opening should be propagated to reduced_evaluations");

        assert_eq!(
            reduced.r,
            point.iter().map(|&c| c.into()).collect::<Vec<_>>()
        );
        assert_eq!(reduced.claim, claim);
    }

    #[test]
    fn verify_iop_runs_in_isolation_after_verify_output_claim() {
        let t = 1 << 4;
        let mut rng = StdRng::seed_from_u64(0x2020);
        let input = Tensor::<i32>::random_small(&mut rng, &[t]);
        let model = one_sub_model(&mut rng, t);

        let shared = AtlasSharedPreprocessing::preprocess(model);
        let trace = shared.model().trace(&[input]);
        let io = Trace::io(&trace, shared.model());

        // Build just enough prover state for (opening_claims, proofs, eval_reduction_proofs)
        // using output_claim + iop (no commitments, no reduced opening proof stage).
        let mut prover = Prover::<_, Blake2bTranscript>::new(shared.clone(), trace);
        let mut proofs = BTreeMap::new();
        let mut eval_reduction_proofs = BTreeMap::new();

        ONNXProof::<_, _, HyperKZG<Bn254>>::output_claim(&mut prover);
        ONNXProof::<_, _, HyperKZG<Bn254>>::iop(
            shared.model().nodes(),
            &mut prover,
            &mut proofs,
            &mut eval_reduction_proofs,
        );

        let opening_claims = Claims(prover.accumulator.take());
        let partial_proof = ONNXProof::<_, _, HyperKZG<Bn254>> {
            opening_claims,
            proofs,
            commitments: vec![],
            eval_reduction_proofs,
            reduced_opening_proof: None,
        };

        let mut verifier = Verifier::new(&shared, &partial_proof.proofs, &io);
        partial_proof.populate_accumulator(&mut verifier);
        ONNXProof::<_, _, HyperKZG<Bn254>>::verify_output_claim(shared.model(), &mut verifier)
            .expect("verify_output_claim should succeed for consistent partial proof state");
        partial_proof
            .verify_iop(shared.model(), &mut verifier)
            .expect("verify_iop should succeed in isolation after output-claim initialization");

        let output_idx = shared.model().outputs()[0];
        assert!(verifier
            .accumulator
            .reduced_evaluations
            .contains_key(&output_idx));
    }
}
