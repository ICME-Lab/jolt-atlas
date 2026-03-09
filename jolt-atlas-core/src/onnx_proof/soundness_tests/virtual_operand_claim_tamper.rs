use super::{
    find_sub_node, seeded_rng, sub_model, sub_model_const_2, TestField, TestPCS, TestTranscript,
};
use crate::onnx_proof::{
    malicious_prover::MaliciousONNXProof, AtlasProverPreprocessing, AtlasSharedPreprocessing,
    AtlasVerifierPreprocessing,
};
use common::VirtualPolynomial;
use joltworks::poly::opening_proof::{OpeningId, SumcheckId};

#[should_panic = "called `Result::unwrap()` on an `Err` value: InvalidOpeningProof(\"Const claim does not match expected claim\")"]
#[test]
fn soundness_sub_virtual_operand_attack_is_rejected() {
    // This test demonstrates the virtual-operand-claim attack shape:
    // 1) malicious_sub forges operand claims (writes forged values into
    //    openings[NodeOutput(input), NodeExecution(sub_node)] via append_virtual)
    // 2) The forged claim propagates to the input/constant node verification,
    //    which detects the mismatch against the known values.
    let t = 1 << 12;
    let mut rng = seeded_rng(0xA77ACCEE);
    let input = atlas_onnx_tracer::tensor::Tensor::<i32>::random_small(&mut rng, &[t]);
    let model = sub_model(&mut rng, t);
    let sub_node = find_sub_node(&model);

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<TestField, TestPCS>::new(pp);
    let (proof, io, _debug_info) =
        MaliciousONNXProof::prove::<TestField, TestTranscript, TestPCS>(&prover_pp, &[input]);
    let verifier_pp = AtlasVerifierPreprocessing::<TestField, TestPCS>::from(&prover_pp);

    for &input_idx in &sub_node.inputs {
        let key = OpeningId::Virtual(
            VirtualPolynomial::NodeOutput(input_idx),
            SumcheckId::NodeExecution(sub_node.idx),
        );
        assert!(
            proof.opening_claims.0.contains_key(&key),
            "sub node forged NodeOutput({input_idx}) claim should exist in opening_claims"
        );
    }

    proof.verify(&verifier_pp, &io, None).unwrap();
}

#[should_panic = "called `Result::unwrap()` on an `Err` value: InvalidOpeningProof(\"Const claim does not match expected claim\")"]
#[test]
fn soundness_sub_trace_tamper_3_minus_2_becomes_0_is_rejected() {
    let model = sub_model_const_2();
    let input = atlas_onnx_tracer::tensor::Tensor::new(Some(&[3]), &[1])
        .expect("input tensor should be valid");

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<TestField, TestPCS>::new(pp);
    let (proof, io, _debug_info) = MaliciousONNXProof::prove_with_sub_trace_tamper_zero::<
        TestField,
        TestTranscript,
        TestPCS,
    >(&prover_pp, &[input]);
    let verifier_pp = AtlasVerifierPreprocessing::<TestField, TestPCS>::from(&prover_pp);

    assert_eq!(io.outputs[0].data()[0], 0);
    proof.verify(&verifier_pp, &io, None).unwrap();
}
