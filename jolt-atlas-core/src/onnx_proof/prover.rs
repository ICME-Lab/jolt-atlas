//! Prover state and internal proving helpers for [`ONNXProof`].
//!
//! The public entry point is [`ONNXProof::prove`] (defined in the parent module).
//! This file houses the [`Prover`] struct and the private helper methods that
//! `prove` delegates to.

use super::{
    types::{Claims, ProofId, ProverDebugInfo},
    AtlasSharedPreprocessing, ONNXProof, ReducedOpeningProof,
};
use crate::onnx_proof::{
    ops::{NodeCommittedPolynomials, OperatorProver},
    witness::WitnessGenerator,
};
use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, ModelExecutionIO, Trace},
        Model,
    },
    node::ComputationNode,
};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{ProverOpeningAccumulator, SumcheckId},
        rlc_polynomial::build_materialized_rlc,
    },
    subprotocols::{
        evaluation_reduction::{EvalReductionProof, EvalReductionProtocol},
        sumcheck::SumcheckInstanceProof,
    },
    transcripts::Transcript,
    utils::math::Math,
};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Prover state
// ---------------------------------------------------------------------------

/// Prover state that owns all data needed during proving.
/// Created once before the proving loop and passed to operator handlers.
pub struct Prover<F: JoltField, T: Transcript> {
    /// Execution trace of the neural network model.
    pub trace: Trace,
    /// Shared preprocessing data (model structure).
    pub preprocessing: AtlasSharedPreprocessing,
    /// Opening accumulator for batching polynomial openings.
    pub accumulator: ProverOpeningAccumulator<F>,
    /// Interactive proof transcript.
    pub transcript: T,
}

impl<F: JoltField, T: Transcript> Prover<F, T> {
    /// Create a new prover with the given preprocessing and trace
    pub fn new(preprocessing: AtlasSharedPreprocessing, trace: Trace) -> Self {
        Self {
            trace,
            preprocessing,
            accumulator: ProverOpeningAccumulator::new(),
            transcript: T::new(b"ONNXProof"),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal prover helpers on ONNXProof
// ---------------------------------------------------------------------------

impl<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> ONNXProof<F, T, PCS> {
    /// Build the witness polynomial map, commit to each polynomial, and append
    /// all commitments to the transcript.
    #[tracing::instrument(skip_all, name = "ONNXProof::commit_witness_polynomials")]
    pub(super) fn commit_witness_polynomials(
        model: &Model,
        trace: &Trace,
        generators: &PCS::ProverSetup,
        transcript: &mut T,
    ) -> (
        BTreeMap<CommittedPolynomial, MultilinearPolynomial<F>>,
        Vec<PCS::Commitment>,
    ) {
        let poly_map = Self::polynomial_map(model, trace);
        let commitments = Self::commit_to_polynomials(&poly_map, generators);
        for commitment in &commitments {
            transcript.append_serializable(commitment);
        }
        (poly_map, commitments)
    }

    pub(super) fn output_claim(prover: &mut Prover<F, T>) {
        // Construct output polynomial
        let output_index = prover.preprocessing.model().outputs()[0];
        let output_computation_node = &prover.preprocessing.model()[output_index];
        let LayerData {
            operands: _,
            output,
        } = Trace::layer_data(&prover.trace, output_computation_node);

        // Pad output tensor to power of 2 with 0 for MLE constructions
        let output = output.padded_next_power_of_two();

        // Sample challenge from verifier
        let r_node_output = prover
            .transcript
            .challenge_vector_optimized::<F>(output.len().log_2());

        // Evaluate output polynomial at r_node_output
        let output_claim = MultilinearPolynomial::from(output.clone()).evaluate(&r_node_output);

        // append_virtual handles both transcript append and insertion into openings
        prover.accumulator.append_virtual(
            &mut prover.transcript,
            VirtualPolynomial::NodeOutput(output_computation_node.idx),
            // NodeOutput claims are generally produced by subsequent nodes during proving; emulate that here.
            SumcheckId::NodeExecution(output_computation_node.idx + 1),
            r_node_output.clone().into(),
            output_claim,
        );
    }

    /// Iterate over computation graph in reverse topological order
    /// Prove each operation using sum-check and virtual polynomials
    #[tracing::instrument(skip_all, name = "ONNXProof::iop")]
    pub(super) fn iop(
        computation_nodes: &BTreeMap<usize, ComputationNode>,
        prover: &mut Prover<F, T>,
        proofs: &mut BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
        eval_reduction_proofs: &mut BTreeMap<usize, EvalReductionProof<F>>,
    ) {
        for (_, node) in computation_nodes.iter().rev() {
            // Before each node is proven,
            // perform eval reduction to reduce openings to a unique claim for the node output.
            eval_reduction_proofs.insert(node.idx, EvalReductionProver::prove(prover, node));

            proofs.extend(OperatorProver::prove(node, prover));
        }
    }

    #[tracing::instrument(skip_all, name = "ONNXProof::prove_reduced_openings")]
    pub(super) fn prove_reduced_openings(
        prover: &mut Prover<F, T>,
        poly_map: &BTreeMap<CommittedPolynomial, MultilinearPolynomial<F>>,
        generators: &PCS::ProverSetup,
    ) -> Option<ReducedOpeningProof<F, T, PCS>> {
        if poly_map.is_empty() {
            None
        } else {
            prover.accumulator.prepare_for_sumcheck(poly_map);

            // Run sumcheck
            let (accumulator_sumcheck_proof, r_sumcheck_acc) = prover
                .accumulator
                .prove_batch_opening_sumcheck(&mut prover.transcript);

            // Finalize sumcheck (uses claims cached via cache_openings, derives gamma, cleans up)
            let state = prover
                .accumulator
                .finalize_batch_opening_sumcheck(r_sumcheck_acc.clone(), &mut prover.transcript);
            let sumcheck_claims: Vec<F> = state.sumcheck_claims.clone();
            // Build RLC
            let rlc = build_materialized_rlc(&state.gamma_powers, poly_map);
            // Create joint opening proof
            let joint_opening_proof = PCS::prove(
                generators,
                &rlc,
                &state.r_sumcheck,
                None,
                &mut prover.transcript,
            );
            Some(ReducedOpeningProof {
                sumcheck_proof: accumulator_sumcheck_proof,
                sumcheck_claims,
                joint_opening_proof,
            })
        }
    }

    pub(super) fn finalize_proof(
        mut prover: Prover<F, T>,
        io: ModelExecutionIO,
        commitments: Vec<PCS::Commitment>,
        proofs: BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
        eval_reduction_proofs: BTreeMap<usize, EvalReductionProof<F>>,
        reduced_opening_proof: Option<ReducedOpeningProof<F, T, PCS>>,
    ) -> (Self, ModelExecutionIO, Option<ProverDebugInfo<F, T>>) {
        #[cfg(test)]
        let debug_info = Some(ProverDebugInfo {
            transcript: prover.transcript.clone(),
            opening_accumulator: prover.accumulator.clone(),
        });
        #[cfg(not(test))]
        let debug_info = None;
        let opening_claims = prover.accumulator.take();
        (
            Self {
                proofs,
                opening_claims: Claims(opening_claims),
                commitments,
                eval_reduction_proofs,
                reduced_opening_proof,
            },
            io,
            debug_info,
        )
    }

    #[tracing::instrument(skip_all)]
    pub(super) fn polynomial_map(
        model: &Model,
        trace: &Trace,
    ) -> BTreeMap<CommittedPolynomial, MultilinearPolynomial<F>> {
        model
            .graph
            .nodes
            .values()
            .flat_map(|node| NodeCommittedPolynomials::get_committed_polynomials::<F, T>(node))
            .map(|committed_poly| {
                let witness = committed_poly.generate_witness(model, trace);
                (committed_poly, witness)
            })
            .collect()
    }

    #[tracing::instrument(skip_all)]
    fn commit_to_polynomials(
        poly_map: &BTreeMap<CommittedPolynomial, MultilinearPolynomial<F>>,
        pcs: &PCS::ProverSetup,
    ) -> Vec<PCS::Commitment> {
        poly_map
            .values()
            .map(|poly| PCS::commit(poly, pcs).0)
            .collect()
    }
}

struct EvalReductionProver;

impl EvalReductionProver {
    /// Reduce dual NodeOutput openings for a producer node before proving that node.
    ///
    /// This is the first integration step for PAZK 4.5.2 (2-to-1 only).
    pub(super) fn prove<F: JoltField, T: Transcript>(
        prover: &mut Prover<F, T>,
        computation_node: &ComputationNode,
    ) -> EvalReductionProof<F> {
        let node_idx = computation_node.idx;
        let openings = prover.accumulator.get_node_openings(node_idx);

        let LayerData {
            operands: _,
            output,
        } = Trace::layer_data(&prover.trace, computation_node);
        let output_mle = MultilinearPolynomial::from(output.padded_next_power_of_two());

        let (proof, reduced) =
            EvalReductionProtocol::prove(&openings, output_mle, &mut prover.transcript)
                .expect("Proving evaluation reduction should not fail");

        prover
            .accumulator
            .reduced_evaluations
            .insert(node_idx, reduced);

        proof
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_proof::{AtlasSharedPreprocessing, HyperKZG};
    use ark_bn254::{Bn254, Fr};
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, Model},
        ops::{Add, Operator},
        tensor::Tensor,
    };
    use common::VirtualPolynomial;
    use joltworks::{
        poly::{
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            opening_proof::{OpeningPoint, SumcheckId},
        },
        transcripts::Blake2bTranscript,
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
    fn eval_reduction_adds_reduced_evaluations() {
        let n = 1 << 3; // 8 points, i.e. 3 variables
        let output: Tensor<i32> = Tensor::construct((0..n).map(|n| n as i32).collect(), vec![n]);
        let model = Model::default();
        let producer_idx = 0;
        let consumer1 = 1;
        let consumer2 = 2;

        let preprocessing = AtlasSharedPreprocessing::preprocess(model);
        let mut trace = preprocessing.model().trace(&[]);
        trace.node_outputs.insert(0, output.clone());
        let mut prover = Prover::<Fr, Blake2bTranscript>::new(preprocessing, trace);

        let producer_node = ComputationNode::new(0, Operator::Add(Add), vec![], vec![n]);

        let output_mle = MultilinearPolynomial::from(output.padded_next_power_of_two());
        let num_vars = output_mle.get_num_vars();

        let point1 = prover.transcript.challenge_vector_optimized::<Fr>(num_vars);
        let point2 = prover.transcript.challenge_vector_optimized::<Fr>(num_vars);
        let claim1 = output_mle.evaluate(&point1);
        let claim2 = output_mle.evaluate(&point2);

        prover.accumulator.append_virtual(
            &mut prover.transcript,
            VirtualPolynomial::NodeOutput(producer_idx),
            SumcheckId::NodeExecution(consumer1),
            OpeningPoint::new(point1),
            claim1,
        );
        prover.accumulator.append_virtual(
            &mut prover.transcript,
            VirtualPolynomial::NodeOutput(producer_idx),
            SumcheckId::NodeExecution(consumer2),
            OpeningPoint::new(point2),
            claim2,
        );

        let _ = EvalReductionProver::prove(&mut prover, &producer_node);

        let entries: Vec<_> = prover.accumulator.get_node_openings(producer_idx);

        // We only seed two per-consumer openings and run eval-reduction proving.
        assert_eq!(entries.len(), 2);
        assert!(prover
            .accumulator
            .reduced_evaluations
            .contains_key(&producer_idx));
    }

    #[test]
    fn eval_reduction_single_opening_is_propagated() {
        let n = 1 << 3; // 8 points, i.e. 3 variables
        let output: Tensor<i32> = Tensor::construct((0..n).map(|n| n as i32).collect(), vec![n]);
        let model = Model::default();
        let producer_idx = 0;
        let consumer = 1;

        let preprocessing = AtlasSharedPreprocessing::preprocess(model);
        let mut trace = preprocessing.model().trace(&[]);
        trace.node_outputs.insert(producer_idx, output.clone());
        let mut prover = Prover::<Fr, Blake2bTranscript>::new(preprocessing, trace);

        let producer_node = ComputationNode::new(producer_idx, Operator::Add(Add), vec![], vec![n]);

        let output_mle = MultilinearPolynomial::from(output.padded_next_power_of_two());
        let num_vars = output_mle.get_num_vars();

        let point = prover.transcript.challenge_vector_optimized::<Fr>(num_vars);
        let claim = output_mle.evaluate(&point);

        prover.accumulator.append_virtual(
            &mut prover.transcript,
            VirtualPolynomial::NodeOutput(producer_idx),
            SumcheckId::NodeExecution(consumer),
            OpeningPoint::new(point.clone()),
            claim,
        );

        let _ = EvalReductionProver::prove(&mut prover, &producer_node);

        let reduced = prover
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
    fn iop_runs_in_isolation_after_output_claim() {
        let t = 1 << 4;
        let mut rng = StdRng::seed_from_u64(0x1010);
        let input = Tensor::<i32>::random_small(&mut rng, &[t]);
        let model = one_sub_model(&mut rng, t);

        let preprocessing = AtlasSharedPreprocessing::preprocess(model);
        let trace = preprocessing.model().trace(&[input]);
        let mut prover = Prover::<Fr, Blake2bTranscript>::new(preprocessing, trace);

        let mut proofs: BTreeMap<ProofId, SumcheckInstanceProof<Fr, Blake2bTranscript>> =
            BTreeMap::new();
        let mut eval_reduction_proofs = BTreeMap::new();

        ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::output_claim(&mut prover);

        let output_idx = prover.preprocessing.model().outputs()[0];
        assert_eq!(prover.accumulator.get_node_openings(output_idx).len(), 1);
        let computation_nodes = prover.preprocessing.model().nodes().clone();

        ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::iop(
            &computation_nodes,
            &mut prover,
            &mut proofs,
            &mut eval_reduction_proofs,
        );

        assert!(!proofs.is_empty());
        assert!(eval_reduction_proofs.contains_key(&output_idx));
    }
}
