use crate::onnx_proof::{
    malicious_prover::MaliciousONNXProof, AtlasProverPreprocessing, AtlasSharedPreprocessing,
    AtlasVerifierPreprocessing,
};
use ark_bn254::{Bn254, Fr};
use atlas_onnx_tracer::{
    model::{test::ModelBuilder, Model},
    node::ComputationNode,
    ops::Operator,
    tensor::Tensor,
};
use common::VirtualPolynomial;
use joltworks::poly::commitment::hyperkzg::HyperKZG;
use joltworks::poly::opening_proof::{OpeningId, SumcheckId};
use joltworks::transcripts::Blake2bTranscript;
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

#[test]
fn soundness_sub_virtual_operand_attack_still_verifies() {
    // This test demonstrates the virtual-operand-claim attack shape:
    // 1) malicious_sub forges operand claims used by verifier expected_output_claim
    // 2) those forged claims differ from the actual NodeOutput opening claims
    // 3) verification still succeeds
    let t = 1 << 12;
    let mut rng = StdRng::seed_from_u64(0xA77ACCEE);
    let input = Tensor::<i32>::random_small(&mut rng, &[t]);
    let model = sub_model(&mut rng, t);
    let sub_node = find_sub_node(&model);
    let left_input_idx = sub_node.inputs[0];
    let right_input_idx = sub_node.inputs[1];

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
    let (proof, io, debug_info) =
        MaliciousONNXProof::prove::<Fr, Blake2bTranscript, HyperKZG<Bn254>>(&prover_pp, &[input]);
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);

    // Forged claims consumed by get_operand_claims(node_idx).
    let virtual_claims = proof
        .virtual_operand_claims
        .get(&sub_node.idx)
        .expect("sub node virtual operand claims should exist");
    // Honest opening claims for the same input nodes at the execution opening.
    let left_opening_claim = proof
        .opening_claims
        .0
        .get(&OpeningId::Virtual(
            VirtualPolynomial::NodeOutput(left_input_idx),
            SumcheckId::Execution,
        ))
        .expect("left opening claim should exist")
        .1;
    let right_opening_claim = proof
        .opening_claims
        .0
        .get(&OpeningId::Virtual(
            VirtualPolynomial::NodeOutput(right_input_idx),
            SumcheckId::Execution,
        ))
        .expect("right opening claim should exist")
        .1;

    // Attack witness: operand claims are not the same as opening claims.
    assert_ne!(
        virtual_claims[0], left_opening_claim,
        "left operand virtual claim should be forged"
    );
    assert_ne!(
        virtual_claims[1], right_opening_claim,
        "right operand virtual claim should be forged"
    );

    // Sound behavior should reject forged operand claims.
    let res = proof.verify(&verifier_pp, &io, debug_info);
    assert!(
        res.is_err(),
        "soundness bug: forged virtual operand claims were accepted (verification unexpectedly succeeded): {res:?}"
    );
}

#[test]
fn soundness_sub_trace_tamper_3_minus_2_becomes_0_and_verifies() {
    let model = sub_model_const_2();
    let input = Tensor::new(Some(&[3]), &[1]).expect("input tensor should be valid");

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);
    let (proof, io, debug_info) =
        MaliciousONNXProof::prove_with_sub_trace_tamper_zero::<
            Fr,
            Blake2bTranscript,
            HyperKZG<Bn254>,
        >(&prover_pp, &[input]);
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);

    // Tampered trace output is forced to zero (even though 3 - 2 should be 1).
    assert_eq!(io.outputs[0].data()[0], 0);
    let res = proof.verify(&verifier_pp, &io, debug_info);
    assert!(
        res.is_err(),
        "soundness bug: tampered-trace proof was accepted (verification unexpectedly succeeded): {res:?}"
    );
}
