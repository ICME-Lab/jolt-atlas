use super::{
    duplicate_operand_sub_model, fanout_sub_model, seeded_rng, TestField, TestPCS, TestTranscript,
};
use crate::onnx_proof::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing, ONNXProof,
};
use common::VirtualPolynomial;
use joltworks::poly::opening_proof::{OpeningId, SumcheckId};

#[test]
#[ignore = "Known issue tracked by #138: multiple NodeOutput openings not yet reduced"]
fn soundness_fanout_nodeoutput_openings_should_be_reduced() {
    // #138 structural issue: one producer (x) consumed by two nodes (y, z)
    // produces two per-consumer openings for NodeOutput(x), keyed by
    // NodeExecution(y) and NodeExecution(z). These should be reduced to a
    // single opening via PAZK 4.5.2, but currently are not.
    let t = 1 << 8;
    let mut rng = seeded_rng(0x138138);
    let input = atlas_onnx_tracer::tensor::Tensor::<i32>::random_small(&mut rng, &[t]);
    let (model, x_idx, _y_idx, _z_idx) = fanout_sub_model(&mut rng, t);

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<TestField, TestPCS>::new(pp);
    let (proof, _io, _debug_info) =
        ONNXProof::<TestField, TestTranscript, TestPCS>::prove(&prover_pp, &[input]);

    let lo = OpeningId::Virtual(
        VirtualPolynomial::NodeOutput(x_idx),
        SumcheckId::NodeExecution(0),
    );
    let hi = OpeningId::Virtual(
        VirtualPolynomial::NodeOutput(x_idx),
        SumcheckId::NodeExecution(usize::MAX),
    );
    let entries: Vec<_> = proof.opening_claims.0.range(lo..=hi).collect();

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
    let mut rng = seeded_rng(0xD0011CAA);
    let input = atlas_onnx_tracer::tensor::Tensor::<i32>::random_small(&mut rng, &[t]);
    let (model, x_idx, y_idx) = duplicate_operand_sub_model(&mut rng, t);

    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<TestField, TestPCS>::new(pp);
    let (proof, _io, _debug_info) =
        ONNXProof::<TestField, TestTranscript, TestPCS>::prove(&prover_pp, &[input]);
    let verifier_pp = AtlasVerifierPreprocessing::<TestField, TestPCS>::from(&prover_pp);

    let y_node = &verifier_pp.model().graph.nodes[&y_idx];
    assert_eq!(
        y_node.inputs[0], y_node.inputs[1],
        "test precondition: y must consume x twice"
    );

    let lo = OpeningId::Virtual(
        VirtualPolynomial::NodeOutput(x_idx),
        SumcheckId::NodeExecution(0),
    );
    let hi = OpeningId::Virtual(
        VirtualPolynomial::NodeOutput(x_idx),
        SumcheckId::NodeExecution(usize::MAX),
    );
    let entries: Vec<_> = proof.opening_claims.0.range(lo..=hi).collect();

    assert_eq!(
        entries.len(),
        2,
        "duplicate operand openings should be independently tracked, but found {}",
        entries.len()
    );
}
