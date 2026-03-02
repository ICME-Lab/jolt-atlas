use crate::onnx_proof::{
    malicious_prover::MaliciousONNXProof, AtlasProverPreprocessing, AtlasSharedPreprocessing,
    AtlasVerifierPreprocessing, ONNXProof, Verifier,
};
use ark_bn254::{Bn254, Fr};
use atlas_onnx_tracer::{
    model::{test::ModelBuilder, Model},
    node::ComputationNode,
    ops::Operator,
    tensor::Tensor,
};
use common::VirtualPolynomial;
use joltworks::poly::opening_proof::{OpeningId, SumcheckId};
use joltworks::{poly::commitment::hyperkzg::HyperKZG, transcripts::Blake2bTranscript};
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

#[should_panic]
#[test]
fn soundness_sub_virtual_operand_attack_is_rejected() {
    // This test demonstrates the virtual-operand-claim attack shape:
    // 1) malicious_sub forges operand claims in proof.virtual_operand_claims
    // 2) The verifier derives openings[NodeOutput(input)] from virtual_operand_claims
    //    (single source of truth). The forged claim propagates to the input/constant
    //    node verification, which detects the mismatch against the known values.
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

    // Confirm the malicious prover stored forged claims in virtual_operand_claims.
    let _virtual_claims = proof
        .virtual_operand_claims
        .get(&sub_node.idx)
        .expect("sub node virtual operand claims should exist");

    // The Sub node's operand NodeOutput entries come exclusively from
    // virtual_operand_claims (single source of truth for interior nodes).
    // The forged claims propagate to input/constant verification, which
    // panics on mismatch.
    proof.verify(&verifier_pp, &io, None).unwrap();
}

#[should_panic]
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
#[ignore = "Known issue tracked by #138: NodeOutput openings collapse to last-writer"]
fn soundness_fanout_nodeoutput_opening_is_last_writer_only() {
    // This test captures the #138 structural issue:
    // one producer node output (x) consumed by two nodes (y, z) yields two
    // distinct operand openings for x, but populate_accumulator materializes
    // a single openings[NodeOutput(x)] entry using last-writer overwrite.
    let t = 1 << 8;
    let mut rng = StdRng::seed_from_u64(0x138138);
    let input = Tensor::<i32>::random_small(&mut rng, &[t]);
    let (model, x_idx, y_idx, z_idx) = fanout_sub_model(&mut rng, t);

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
    let (proof, io, _debug_info) =
        ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(&prover_pp, &[input]);
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);

    let y_claim_for_x = proof
        .virtual_operand_claims
        .get(&y_idx)
        .expect("y virtual operand claims should exist")[0]
        .clone();
    let z_claim_for_x = proof
        .virtual_operand_claims
        .get(&z_idx)
        .expect("z virtual operand claims should exist")[0]
        .clone();

    assert_ne!(
        y_claim_for_x, z_claim_for_x,
        "fan-out should produce two distinct openings for NodeOutput(x)"
    );

    // Run exactly the verifier pre-loading step to inspect reconstructed openings.
    let mut verifier: Verifier<Fr, Blake2bTranscript> =
        Verifier::new(&verifier_pp.shared, &proof.proofs, &io);
    proof.populate_accumulator(verifier_pp.model(), &mut verifier);

    let key = OpeningId::Virtual(VirtualPolynomial::NodeOutput(x_idx), SumcheckId::Execution);
    let materialized = verifier
        .accumulator
        .openings
        .get(&key)
        .expect("NodeOutput(x) opening should be materialized")
        .clone();

    // Desired property (future fix): both consumer openings for shared x should be
    // independently tracked/verified, rather than collapsing to a single materialized
    // NodeOutput(x) opening. The current implementation collapses to one and should fail this check.
    assert_ne!(
        materialized, y_claim_for_x,
        "bug: shared NodeOutput(x) collapsed to y's opening (last-writer behavior)"
    );
    assert_ne!(
        materialized, z_claim_for_x,
        "bug: shared NodeOutput(x) collapsed to z's opening (last-writer behavior)"
    );
}

#[test]
#[ignore = "Known issue tracked by #138: duplicate-operand NodeOutput openings collapse to last-writer"]
fn soundness_same_consumer_duplicate_operand_opening_collapses_to_last_writer() {
    // Same-consumer duplicate-operand variant: y = sub(x, x).
    // Two operand openings are produced for the same producer node x in the same
    // consumer y, but openings[NodeOutput(x)] materializes as a single entry.
    let t = 1 << 8;
    let mut rng = StdRng::seed_from_u64(0xD0011CAA);
    let input = Tensor::<i32>::random_small(&mut rng, &[t]);
    let (model, x_idx, y_idx) = duplicate_operand_sub_model(&mut rng, t);

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
    let (proof, io, _debug_info) =
        ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(&prover_pp, &[input]);
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);

    let y_node = &verifier_pp.model().graph.nodes[&y_idx];
    assert_eq!(
        y_node.inputs[0], y_node.inputs[1],
        "test precondition: y must consume x twice"
    );

    let y_openings = proof
        .virtual_operand_claims
        .get(&y_idx)
        .expect("y virtual operand claims should exist");
    assert_eq!(
        y_openings.len(),
        2,
        "y should expose two operand openings (left and right)"
    );
    assert_eq!(
        y_openings[0].0, y_openings[1].0,
        "duplicate operand openings should share the same opening point"
    );

    let mut verifier: Verifier<Fr, Blake2bTranscript> =
        Verifier::new(&verifier_pp.shared, &proof.proofs, &io);
    proof.populate_accumulator(verifier_pp.model(), &mut verifier);

    let key = OpeningId::Virtual(VirtualPolynomial::NodeOutput(x_idx), SumcheckId::Execution);
    let materialized = verifier
        .accumulator
        .openings
        .get(&key)
        .expect("NodeOutput(x) opening should be materialized")
        .clone();

    // Desired property (future fix): duplicate operand openings for the same producer
    // should not collapse to a single materialized NodeOutput(x) opening.
    // The current implementation collapses and should fail these checks.
    assert_ne!(
        materialized, y_openings[0],
        "bug: duplicate-operand opening[0] collapsed into NodeOutput(x)"
    );
    assert_ne!(
        materialized, y_openings[1],
        "bug: duplicate-operand opening[1] collapsed into NodeOutput(x)"
    );
}
