use crate::onnx_proof::{
    malicious_prover::MaliciousONNXProof, AtlasProverPreprocessing, AtlasSharedPreprocessing,
    AtlasVerifierPreprocessing, ONNXProof,
};
use ark_bn254::{Bn254, Fr};
use onnx_tracer::{
    model::{test::ModelBuilder, Model},
    node::ComputationNode,
    ops::Operator,
    tensor::Tensor,
};
use common::VirtualPolynomial;
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
        let key = OpeningId::Virtual(
            VirtualPolynomial::NodeOutput(input_idx),
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
#[ignore = "Known issue tracked by #138: multiple NodeOutput openings not yet reduced"]
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

    // Count how many per-consumer entries exist for NodeOutput(x).
    let lo = OpeningId::Virtual(
        VirtualPolynomial::NodeOutput(x_idx),
        SumcheckId::NodeExecution(0),
    );
    let hi = OpeningId::Virtual(
        VirtualPolynomial::NodeOutput(x_idx),
        SumcheckId::NodeExecution(usize::MAX),
    );
    let entries: Vec<_> = proof.opening_claims.0.range(lo..=hi).collect();

    // Desired property (#138): all per-consumer openings for NodeOutput(x)
    // should be reduced to a single opening via PAZK 4.5.2.
    // This assertion FAILS today (entries.len() == 2) because reduction is not implemented.
    assert_eq!(
        entries.len(),
        1,
        "NodeOutput(x) should have exactly one (reduced) opening, but found {}",
        entries.len()
    );
}

#[test]
#[ignore = "Known issue tracked by #138: duplicate-operand NodeOutput openings collapse to last-writer"]
fn soundness_same_consumer_duplicate_operand_should_track_both() {
    // y = sub(x, x): both operands write NodeOutput(x) + NodeExecution(y).
    // The second write overwrites the first, collapsing two distinct operand
    // openings into one entry. #138 should independently track both.
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
    let lo = OpeningId::Virtual(
        VirtualPolynomial::NodeOutput(x_idx),
        SumcheckId::NodeExecution(0),
    );
    let hi = OpeningId::Virtual(
        VirtualPolynomial::NodeOutput(x_idx),
        SumcheckId::NodeExecution(usize::MAX),
    );
    let entries: Vec<_> = proof.opening_claims.0.range(lo..=hi).collect();

    // Desired property (#138): both operand openings for the same producer
    // should be independently tracked (2 entries), not collapsed via overwrite.
    // This assertion FAILS today (entries.len() == 1) because the second
    // append_virtual overwrites the first for the same key.
    assert_eq!(
        entries.len(),
        2,
        "duplicate operand openings should be independently tracked, but found {}",
        entries.len()
    );
}
