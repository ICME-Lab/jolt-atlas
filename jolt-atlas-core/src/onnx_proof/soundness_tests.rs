use crate::onnx_proof::{
    malicious_prover::MaliciousONNXProof, AtlasProverPreprocessing, AtlasSharedPreprocessing,
    AtlasVerifierPreprocessing, ONNXProof,
};
use ark_bn254::{Bn254, Fr};
use atlas_onnx_tracer::{
    model::{test::ModelBuilder, Model},
    node::ComputationNode,
    ops::Operator,
    tensor::Tensor,
};
use common::VirtualPoly;
use joltworks::{
    poly::{
        commitment::hyperkzg::HyperKZG,
        opening_proof::{OpeningId, SumcheckId},
    },
    transcripts::Blake2bTranscript,
};
use rand::{rngs::StdRng, SeedableRng};

fn sub_model(rng: &mut StdRng, t: usize) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![t]);
    let c = b.constant(Tensor::random_small(rng, &[t]));
    let out = b.sub(i, c);
    b.mark_output(out);
    b.build()
}

fn sub_model_const_2() -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![1]);
    let c = b.constant(Tensor::new(Some(&[2]), &[1]).expect("constant tensor should be valid"));
    let out = b.sub(i, c);
    b.mark_output(out);
    b.build()
}

fn find_sub_node(model: &Model) -> ComputationNode {
    model
        .graph
        .nodes
        .values()
        .find(|node| matches!(node.operator, Operator::Sub(_)))
        .cloned()
        .expect("sub node should exist")
}

fn fanout_sub_model(rng: &mut StdRng, t: usize) -> (Model, usize, usize, usize) {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![t]);
    let c0 = b.constant(Tensor::random_small(rng, &[t]));
    let c1 = b.constant(Tensor::random_small(rng, &[t]));
    let c2 = b.constant(Tensor::random_small(rng, &[t]));

    // shared producer
    let x = b.sub(i, c0);
    // two consumers of x
    let y = b.sub(x, c1);
    let z = b.sub(x, c2);
    let o = b.add(y, z);
    b.mark_output(o);
    (b.build(), x, y, z)
}

fn duplicate_operand_sub_model(rng: &mut StdRng, t: usize) -> (Model, usize, usize) {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![t]);
    let c0 = b.constant(Tensor::random_small(rng, &[t]));
    // shared producer
    let x = b.sub(i, c0);
    // same consumer, same producer twice
    let y = b.sub(x, x);
    b.mark_output(y);
    (b.build(), x, y)
}

#[should_panic = "called `Result::unwrap()` on an `Err` value: InvalidOpeningProof(\"Const claim does not match expected claim\")"]
#[test]
fn soundness_sub_virtual_operand_attack_is_rejected() {
    // This test demonstrates the virtual-operand-claim attack shape:
    // 1) malicious_sub forges operand claims (writes forged values into
    //    openings[NodeOutput(input), NodeExecution(sub_node)] via append_virtual)
    // 2) The forged claim propagates to the input/constant node verification,
    //    which detects the mismatch against the known values.
    let t = 1 << 12;
    let mut rng = StdRng::seed_from_u64(0xA77ACCEE);
    let input = Tensor::<i32>::random_small(&mut rng, &[t]);
    let model = sub_model(&mut rng, t);
    let sub_node = find_sub_node(&model);

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
    let (proof, io, _debug_info) =
        MaliciousONNXProof::prove::<Fr, Blake2bTranscript, HyperKZG<Bn254>>(&prover_pp, &[input]);
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);

    // Confirm the malicious prover stored forged NodeOutput claims in opening_claims.
    for &input_idx in sub_node.inputs.iter() {
        let key = OpeningId::new(
            VirtualPoly::NodeOutput(input_idx),
            SumcheckId::NodeExecution(sub_node.idx),
        );
        assert!(
            proof.opening_claims.0.contains_key(&key),
            "sub node forged NodeOutput({input_idx}) claim should exist in opening_claims"
        );
    }

    // The forged NodeOutput claims propagate to input/constant verification,
    // which panics on mismatch.
    proof.verify(&verifier_pp, &io, None).unwrap();
}

#[should_panic = "called `Result::unwrap()` on an `Err` value: InvalidOpeningProof(\"Const claim does not match expected claim\")"]
#[test]
fn soundness_sub_trace_tamper_3_minus_2_becomes_0_is_rejected() {
    let model = sub_model_const_2();
    let input = Tensor::new(Some(&[3]), &[1]).expect("input tensor should be valid");

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
    let (proof, io, _debug_info) = MaliciousONNXProof::prove_with_sub_trace_tamper_zero::<
        Fr,
        Blake2bTranscript,
        HyperKZG<Bn254>,
    >(&prover_pp, &[input]);
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);

    // Tampered trace output is forced to zero (even though 3 - 2 should be 1).
    assert_eq!(io.outputs[0].data()[0], 0);
    // Verifier now correctly rejects the tampered proof.
    proof.verify(&verifier_pp, &io, None).unwrap();
}

#[test]
fn soundness_fanout_nodeoutput_openings_should_be_reduced() {
    // #138 structural issue: one producer (x) consumed by two nodes (y, z)
    // produces two per-consumer openings for NodeOutput(x), keyed by
    // NodeExecution(y) and NodeExecution(z). These should be reduced to a
    // single opening via PAZK 4.5.2, but currently are not — only one gets
    // transitively verified against the committed polynomial.
    let t = 1 << 8;
    let mut rng = StdRng::seed_from_u64(0x138138);
    let input = Tensor::<i32>::random_small(&mut rng, &[t]);
    let (model, x_idx, _y_idx, _z_idx) = fanout_sub_model(&mut rng, t);

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
    let (proof, _io, _debug_info) =
        ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(&prover_pp, &[input]);

    let eval_reduction_proofs = proof.eval_reduction_proofs;

    // Desired property (#138): all per-consumer openings for NodeOutput(x)
    // should be reduced to a single opening via PAZK 4.5.2.
    // This assertion FAILS today (entries.len() == 2) because reduction is not implemented.
    assert!(
        eval_reduction_proofs.contains_key(&x_idx),
        "NodeOutput({x_idx}) should have an evaluation reduction proof, but none found"
    );
}

#[test]
fn soundness_same_consumer_duplicate_operand_should_track_both() {
    // y = sub(x, x): both operands write NodeOutput(x) + NodeExecution(y).
    // Both operand openings are at the same opening point and with the same
    // claimed value, so they can be represented as a single opening entry.
    // This test asserts that this deduped representation still verifies.
    let t = 1 << 8;
    let mut rng = StdRng::seed_from_u64(0xD0011CAA);
    let input = Tensor::<i32>::random_small(&mut rng, &[t]);
    let (model, x_idx, y_idx) = duplicate_operand_sub_model(&mut rng, t);

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
    let (proof, _io, _debug_info) =
        ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(&prover_pp, &[input]);
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);

    let y_node = &verifier_pp.model().graph.nodes[&y_idx];
    assert_eq!(
        y_node.inputs[0], y_node.inputs[1],
        "test precondition: y must consume x twice"
    );

    // Count entries for NodeOutput(x) in opening_claims.
    let lo = OpeningId::new(VirtualPoly::NodeOutput(x_idx), SumcheckId::NodeExecution(0));
    let hi = OpeningId::new(
        VirtualPoly::NodeOutput(x_idx),
        SumcheckId::NodeExecution(usize::MAX),
    );
    let entries: Vec<_> = proof.opening_claims.0.range(lo..=hi).collect();

    // Duplicate-operand openings are expected to collapse into one entry.
    assert_eq!(
        entries.len(),
        1,
        "duplicate operand openings should collapse into one entry, but found {}",
        entries.len()
    );

    // The proving flow should still complete and verify successfully.
    proof.verify(&verifier_pp, &_io, None).unwrap();
}
