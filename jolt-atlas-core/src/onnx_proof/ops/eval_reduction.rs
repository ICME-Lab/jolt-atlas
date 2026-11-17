use atlas_trace::trace::LayerData;
pub use jolt_atlas_verifier::onnx_proof::ops::eval_reduction::*;

use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use crate::onnx_proof::Prover;
use atlas_algebra::{field::JoltField, poly::multilinear::MultilinearPolynomial};
use atlas_sumcheck::evaluation_reduction::{EvalReductionProtocol, ReducedInstance};
use atlas_transcript::ProvingTranscript;

/// Prover-side methods of [`NodeEvalReduction`].
pub trait NodeEvalReductionProve {
    /// Run node-output evaluation reduction on the prover side for the given node. A node with
    /// a single claim on its output sends nothing: the claim is the reduced instance.
    fn prove<F: JoltField, T: ProvingTranscript>(
        prover: &mut Prover<F, T>,
        computation_node: &ComputationNode,
    );
}

impl NodeEvalReductionProve for NodeEvalReduction {
    /// Run node-output evaluation reduction on the prover side for the given node.
    #[tracing::instrument(skip_all, name = "NodeEvalReduction::prove", fields(node = computation_node.idx))]
    fn prove<F: JoltField, T: ProvingTranscript>(
        prover: &mut Prover<F, T>,
        computation_node: &ComputationNode,
    ) {
        let node_idx = computation_node.idx;
        let openings = prover.accumulator.get_node_openings(node_idx);

        // Single consumer: the protocol's short path returns the opening as the
        // reduced instance and never touches the output MLE (or the transcript),
        // so skip converting the output tensor to field elements — that
        // conversion is the whole cost for the ~2k single-consumer nodes.
        let reduced = if openings.len() == 1 {
            let (r, claim) = (openings[0].0.clone(), openings[0].1);
            ReducedInstance { r: r.r, claim }
        } else {
            let LayerData {
                operands: _,
                output,
            } = TraceView::layer_data(&prover.trace, computation_node);
            let output_mle = MultilinearPolynomial::from(output.padded_pow2());
            EvalReductionProtocol::prove(&openings, output_mle, &mut prover.transcript)
                .expect("Proving evaluation reduction should not fail")
        };

        prover
            .accumulator
            .reduced_evaluations
            .insert(node_idx, reduced);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_view::ModelView;
    use crate::onnx_proof::testing::ModelBuilder;
    use crate::onnx_proof::{
        prover, schedule::Schedule, verifier, AtlasSharedPreprocessing, ONNXProof, Verifier,
    };
    use atlas_algebra::field::fp128::Fp128 as Fr;
    use atlas_algebra::poly::{
        multilinear::{MultilinearPolynomial, PolynomialEvaluation},
        point::OpeningPoint,
    };
    use atlas_sumcheck::opening_proof::{OpeningId, SumcheckId};
    use atlas_trace::op::Add;
    use atlas_trace::trace::ModelExecutionIO;
    use atlas_trace::Model;
    use atlas_trace::Op;
    use atlas_trace::Tensor;
    use atlas_transcript::{
        Blake2bProver, Blake2bVerifier, ProvingTranscript, Transcript, VerifyingTranscript,
    };
    use common::VirtualPoly;
    use rand::{rngs::StdRng, SeedableRng};

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
        let model = {
            let mut b = ModelBuilder::new();
            let x = b.input(vec![n]);
            b.mark_output(x);
            b.build()
        };
        let producer_idx = 0;
        let consumer1 = 1;
        let consumer2 = 2;

        let preprocessing = AtlasSharedPreprocessing::preprocess(model);
        let trace = preprocessing
            .model()
            .trace(std::slice::from_ref(&output))
            .expect("model execution failed");
        let mut prover = Prover::<Fr, Blake2bProver>::new(preprocessing, trace);

        let producer_node = ComputationNode::new(0, Op::Add(Add), vec![], vec![n]);

        let output_mle = MultilinearPolynomial::from(output.padded_pow2());
        let num_vars = output_mle.get_num_vars();

        let point1 = prover.transcript.challenge_vector_optimized::<Fr>(num_vars);
        let point2 = prover.transcript.challenge_vector_optimized::<Fr>(num_vars);
        let claim1 = output_mle.evaluate(&point1);
        let claim2 = output_mle.evaluate(&point2);

        let key1 = OpeningId::new(
            VirtualPoly::NodeOutput(producer_idx),
            SumcheckId::NodeExecution(consumer1),
        );
        let key2 = OpeningId::new(
            VirtualPoly::NodeOutput(producer_idx),
            SumcheckId::NodeExecution(consumer2),
        );
        prover.accumulator.append_virtual(
            &mut prover.transcript,
            key1,
            OpeningPoint::new(point1),
            claim1,
        );
        prover.accumulator.append_virtual(
            &mut prover.transcript,
            key2,
            OpeningPoint::new(point2),
            claim2,
        );

        NodeEvalReduction::prove(&mut prover, &producer_node);

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
        let model = {
            let mut b = ModelBuilder::new();
            let x = b.input(vec![n]);
            b.mark_output(x);
            b.build()
        };
        let producer_idx = 0;
        let consumer = 1;

        let preprocessing = AtlasSharedPreprocessing::preprocess(model);
        let trace = preprocessing
            .model()
            .trace(std::slice::from_ref(&output))
            .expect("model execution failed");
        let mut prover = Prover::<Fr, Blake2bProver>::new(preprocessing, trace);

        let producer_node = ComputationNode::new(producer_idx, Op::Add(Add), vec![], vec![n]);

        let output_mle = MultilinearPolynomial::from(output.padded_pow2());
        let num_vars = output_mle.get_num_vars();

        let point = prover.transcript.challenge_vector_optimized::<Fr>(num_vars);
        let claim = output_mle.evaluate(&point);

        let key = OpeningId::new(
            VirtualPoly::NodeOutput(producer_idx),
            SumcheckId::NodeExecution(consumer),
        );
        prover.accumulator.append_virtual(
            &mut prover.transcript,
            key,
            OpeningPoint::new(point.clone()),
            claim,
        );

        NodeEvalReduction::prove(&mut prover, &producer_node);

        let reduced = prover
            .accumulator
            .reduced_evaluations
            .get(&producer_idx)
            .expect("single opening should be propagated to reduced_evaluations");

        assert_eq!(reduced.r, point.to_vec());
        assert_eq!(reduced.claim, claim);
    }

    #[test]
    fn eval_reduction_verifier_single_opening_is_propagated() {
        let n = 1 << 3; // 8 points, i.e. 3 variables
        let output: Tensor<i32> = Tensor::construct((0..n).map(|i| i as i32).collect(), vec![n]);
        let producer_idx = 0;
        let consumer = 1;

        let io = ModelExecutionIO {
            inputs: Vec::new(),
            outputs: Vec::new(),
            input_indices: Vec::new(),
            output_indices: Vec::new(),
        };

        let mut verifier = Verifier::<Fr, Blake2bVerifier>::new(&io, &[]);

        let output_mle = MultilinearPolynomial::from(output.padded_pow2());
        let num_vars = output_mle.get_num_vars();

        let point = verifier
            .transcript_mut()
            .challenge_vector_optimized::<Fr>(num_vars);
        let claim = output_mle.evaluate(&point);
        let key = OpeningId::new(
            VirtualPoly::NodeOutput(producer_idx),
            SumcheckId::NodeExecution(consumer),
        );
        // The claim reaches the accumulator through a stream of its own.
        let mut claims = Blake2bProver::new(b"claims");
        claims.send_scalar(&claim);
        verifier
            .accumulator_mut()
            .append_virtual(
                &mut Blake2bVerifier::new(b"claims", claims.into_narg()),
                key,
                OpeningPoint::new(point.clone()),
            )
            .expect("the stream carries the claim");

        NodeEvalReduction::verify(&mut verifier, producer_idx)
            .expect("single opening should be accepted and propagated by verifier");

        let reduced = verifier
            .accumulator()
            .reduced_evaluations
            .get(&producer_idx)
            .expect("single opening should be propagated to reduced_evaluations");

        assert_eq!(reduced.r, point.to_vec());
        assert_eq!(reduced.claim, claim);
    }

    #[test]
    fn iop_runs_in_isolation_after_output_claim() {
        let t = 1 << 4;
        let mut rng = StdRng::seed_from_u64(0x1010);
        let input = Tensor::<i32>::random_small(&mut rng, &[t]);
        let model = one_sub_model(&mut rng, t);

        let preprocessing = AtlasSharedPreprocessing::preprocess(model);
        let trace = preprocessing
            .model()
            .trace(&[input])
            .expect("model execution failed");
        let mut prover = Prover::<Fr, Blake2bProver>::new(preprocessing, trace);

        prover::output_claim(&mut prover);

        let output_idx = prover.preprocessing.model().output_indices()[0];
        assert_eq!(prover.accumulator.get_node_openings(output_idx).len(), 1);
        let schedule =
            Schedule::compile(prover.preprocessing.model()).expect("the model has a schedule");

        prover::iop(&schedule, &mut prover);

        // A single claim is its own reduced instance.
        let (_, claim) = prover.accumulator.get_node_openings(output_idx)[0].clone();
        assert_eq!(
            prover.accumulator.reduced_evaluations[&output_idx].claim,
            claim
        );
    }

    #[test]
    fn schedule_runs_in_isolation_after_verify_output_claim() {
        let t = 1 << 4;
        let mut rng = StdRng::seed_from_u64(0x2020);
        let input = Tensor::<i32>::random_small(&mut rng, &[t]);
        let model = one_sub_model(&mut rng, t);

        let shared = AtlasSharedPreprocessing::preprocess(model);
        let trace = shared
            .model()
            .trace(&[input])
            .expect("model execution failed");
        let io = trace.io(shared.model());

        // Build just enough prover state for the stream of the output claim and the schedule
        // (no commitments, no opening reduction).
        let mut prover = Prover::<_, Blake2bProver>::new(shared.clone(), trace);

        let schedule = Schedule::compile(shared.model()).expect("the model has a schedule");
        prover::output_claim(&mut prover);
        prover::iop(&schedule, &mut prover);

        let partial_proof = ONNXProof(prover.transcript.into_narg());

        let mut verifier = Verifier::<Fr, Blake2bVerifier>::new(&io, &partial_proof.0);
        verifier::verify_output_claim(shared.model(), &mut verifier)
            .expect("verify_output_claim should succeed for consistent partial proof state");
        verifier
            .run(&schedule)
            .expect("the schedule should run in isolation after output-claim initialization");

        let output_idx = shared.model().output_indices()[0];
        assert!(verifier
            .accumulator()
            .reduced_evaluations
            .contains_key(&output_idx));
    }
}
