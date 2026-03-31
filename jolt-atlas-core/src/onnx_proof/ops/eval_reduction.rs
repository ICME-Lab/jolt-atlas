use crate::onnx_proof::{Prover, Verifier};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
};
use joltworks::{
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
    subprotocols::evaluation_reduction::{EvalReductionProof, EvalReductionProtocol},
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

/// Node-scoped evaluation reduction helper used by ONNX proving and verification.
pub struct NodeEvalReduction;

impl NodeEvalReduction {
    /// Run node-output evaluation reduction on the prover side for the given node.
    pub fn prove<F: JoltField, T: Transcript>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_proof::{AtlasSharedPreprocessing, Claims, HyperKZG, ONNXProof, ProofId};
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
        subprotocols::{
            evaluation_reduction::EvalReductionProtocol, sumcheck::SumcheckInstanceProof,
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

        let _ = NodeEvalReduction::prove(&mut prover, &producer_node);

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

        let _ = NodeEvalReduction::prove(&mut prover, &producer_node);

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

        NodeEvalReduction::verify(&mut verifier, &producer_node, &proof)
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
