use super::*;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use atlas_onnx_tracer::{model::test::ModelBuilder, ops::*};
use joltworks::poly::commitment::{
    commitment_scheme::CommitmentScheme,
    dory::{native_graph::NativeGraphProof, DoryScheme},
};
use std::sync::LazyLock;

fn tensor(values: &[i32], shape: &[usize]) -> Tensor<i32> {
    Tensor::new(Some(values), shape).unwrap()
}

fn encoded(value: &impl CanonicalSerialize) -> Vec<u8> {
    let mut bytes = vec![];
    value.serialize_compressed(&mut bytes).unwrap();
    bytes
}

fn roundtrip<T: CanonicalSerialize + CanonicalDeserialize>(value: &T) -> T {
    let bytes = encoded(value);
    let mut remaining = bytes.as_slice();
    let result = T::deserialize_compressed(&mut remaining).unwrap();
    assert!(remaining.is_empty());
    result
}

// Cargo tests share one initialization. Nextest serializes this package with
// joltworks because the Dory dependency writes its setup cache without a lock.
static SETUP: LazyLock<DoryProverSetup> = LazyLock::new(|| DoryScheme::setup_prover(12));

fn unary(op: Operator, shape: &[usize], output: &[usize]) -> Model {
    let mut model = Model::default();
    model.graph.nodes.insert(
        3,
        ComputationNode::new(3, Operator::Input(Input), vec![], shape.to_vec()),
    );
    model
        .graph
        .nodes
        .insert(7, ComputationNode::new(7, op, vec![3], output.to_vec()));
    model.graph.inputs = vec![3];
    model.graph.outputs = vec![7];
    model
}

// Compare every original intermediate with Model::execute_graph, not another
// copy of the native lowering. Exposing intermediates as outputs is test-only.
fn parity(model: &Model, inputs: &[Tensor<i32>]) {
    let compiled = NativeModel::new(model, b"parity".to_vec()).unwrap();
    let repeated = NativeModel::new(model, b"parity".to_vec()).unwrap();
    assert_eq!(encoded(compiled.graph()), encoded(repeated.graph()));
    let mut graph = compiled.graph.clone();
    graph.outputs = compiled.tensors.values().copied().collect();
    let mut values = compiled.constants.clone();
    values.extend(
        compiled
            .inputs
            .iter()
            .copied()
            .zip(inputs.iter().map(|t| t.inner.clone())),
    );
    let witness = NativeGraphWitness::uncommitted(&graph, values.into_values().collect()).unwrap();
    let reference = model.execute_graph(inputs);
    for ((id, _), value) in compiled.tensors.iter().zip(witness.outputs()) {
        assert_eq!(value, reference[id].data(), "model node {id}");
    }
}

#[test]
fn arithmetic_and_integer_boundaries_match_atlas() {
    for operator in [
        Operator::Add(Add),
        Operator::Sub(Sub),
        Operator::Mul(Mul { scale: 4 }),
    ] {
        let mut model = unary(operator, &[4], &[4]);
        model.graph.nodes.insert(
            5,
            ComputationNode::new(5, Operator::Input(Input), vec![], vec![4]),
        );
        model.graph.inputs.push(5);
        model.graph.nodes.get_mut(&7).unwrap().inputs.push(5);
        parity(
            &model,
            &[
                tensor(&[i32::MAX, i32::MIN, -7, 11], &[4]),
                tensor(&[2, -2, 3, -5], &[4]),
            ],
        );
    }
    for (op, shape, output, values) in [
        (
            Operator::Square(Square { scale: 4 }),
            vec![4],
            vec![4],
            vec![i32::MIN, i32::MAX, -7, 11],
        ),
        (
            Operator::Sum(Sum { axes: vec![1] }),
            vec![2, 2],
            vec![2, 1],
            vec![i32::MAX, 1, i32::MIN, -1],
        ),
        (
            Operator::ScalarConstDiv(ScalarConstDiv { divisor: 3 }),
            vec![4],
            vec![4],
            vec![-7, -1, 0, 7],
        ),
        (
            Operator::Rsqrt(Rsqrt { scale: 4 }),
            vec![4],
            vec![4],
            vec![i32::MIN, 0, 1, 256],
        ),
        (
            Operator::MeanOfSquares(MeanOfSquares {
                axes: vec![1],
                scale: 4,
                count: 3,
                padded_count: 4,
            }),
            vec![1, 4],
            vec![1, 1],
            vec![16, 32, -16, 0],
        ),
    ] {
        parity(&unary(op, &shape, &output), &[tensor(&values, &shape)]);
    }
}

#[test]
fn layouts_match_atlas_with_scalars_and_negative_concat_axis() {
    for (op, shape, output, values) in [
        (Operator::Identity(Identity), vec![], vec![], vec![-9]),
        (
            Operator::Broadcast(Broadcast { shape: vec![2, 4] }),
            vec![1, 4],
            vec![2, 4],
            vec![1, -2, 3, -4],
        ),
        (
            Operator::Reshape(Reshape { shape: vec![2, 2] }),
            vec![4],
            vec![2, 2],
            vec![1, 2, 3, 4],
        ),
        (
            Operator::MoveAxis(MoveAxis {
                source: 0,
                destination: 1,
            }),
            vec![2, 4],
            vec![4, 2],
            (0..8).collect(),
        ),
        (
            Operator::Slice(Slice {
                axis: 1,
                start: 2,
                end: 4,
            }),
            vec![2, 4],
            vec![2, 2],
            (0..8).collect(),
        ),
    ] {
        parity(&unary(op, &shape, &output), &[tensor(&values, &shape)]);
    }
    let mut builder = ModelBuilder::new();
    let a = builder.input(vec![2, 2]);
    let b = builder.input(vec![2, 2]);
    let out = builder.concat(&[a, b], -1);
    builder.mark_output(out);
    let model = builder.build();
    parity(
        &model,
        &[
            tensor(&[1, 2, 3, 4], &[2, 2]),
            tensor(&[5, 6, 7, 8], &[2, 2]),
        ],
    );
}

#[test]
fn normalization_graph_preserves_all_intermediates() {
    let model = normalization();
    parity(&model, &[tensor(&[16, -32, 48, 0], &[1, 4])]);
}

fn normalization() -> Model {
    let mut model = unary(
        Operator::MeanOfSquares(MeanOfSquares {
            axes: vec![1],
            scale: 4,
            count: 3,
            padded_count: 4,
        }),
        &[1, 4],
        &[1, 1],
    );
    for node in [
        ComputationNode::new(
            8,
            Operator::Constant(Constant(tensor(&[1], &[1, 1]))),
            vec![],
            vec![1, 1],
        ),
        ComputationNode::new(9, Operator::Add(Add), vec![7, 8], vec![1, 1]),
        ComputationNode::new(10, Operator::Rsqrt(Rsqrt { scale: 4 }), vec![9], vec![1, 1]),
        ComputationNode::new(
            11,
            Operator::Broadcast(Broadcast { shape: vec![1, 4] }),
            vec![10],
            vec![1, 4],
        ),
        ComputationNode::new(12, Operator::Mul(Mul { scale: 4 }), vec![3, 11], vec![1, 4]),
    ] {
        model.graph.nodes.insert(node.idx, node);
    }
    model.graph.outputs = vec![12];
    model
}

#[test]
fn inlined_softmax_and_periodic_functions_match_atlas() {
    let mut model = unary(Operator::Cos(Cos { scale: 4 }), &[1, 4], &[1, 4]);
    model.graph.nodes.insert(
        8,
        ComputationNode::new(8, Operator::Sin(Sin { scale: 4 }), vec![7], vec![1, 4]),
    );
    model.graph.nodes.insert(
        9,
        ComputationNode::new(
            9,
            Operator::SoftmaxLastAxis(SoftmaxLastAxis { scale: 4 }),
            vec![8],
            vec![1, 4],
        ),
    );
    model.graph.outputs = vec![9];
    parity(&model, &[tensor(&[-1000, -1, 0, 1000], &[1, 4])]);
    // The centering checks appended after the numerical softmax output must
    // remain in the translated graph. An overflowing row difference rejects.
    let compiled = NativeModel::new(
        &unary(
            Operator::SoftmaxLastAxis(SoftmaxLastAxis { scale: 4 }),
            &[1, 2],
            &[1, 2],
        ),
        b"overflow".to_vec(),
    )
    .unwrap();
    assert!(
        NativeGraphWitness::uncommitted(compiled.graph(), vec![vec![i32::MIN, i32::MAX]]).is_err()
    );
}

#[test]
fn input_order_is_preserved_and_bound_in_registration() {
    let mut builder = ModelBuilder::new();
    let a = builder.input(vec![2]);
    let b = builder.input(vec![2]);
    let out = builder.sub(a, b);
    builder.mark_output(out);
    let mut model = builder.build();
    let first = NativeModel::new(&model, b"ordered inputs".to_vec()).unwrap();
    model.graph.inputs.reverse();
    parity(&model, &[tensor(&[3, 5], &[2]), tensor(&[11, 19], &[2])]);
    let second = NativeModel::new(&model, b"ordered inputs".to_vec()).unwrap();
    assert_ne!(first.graph.context, second.graph.context);
    let cache = second.preprocess(&SETUP).unwrap();
    let (_, witness) = cache
        .commit(&[tensor(&[3, 5], &[2]), tensor(&[11, 19], &[2])], &SETUP)
        .unwrap();
    assert_eq!(witness.outputs(), &[vec![8, 14]]);
}

#[test]
fn actual_normalization_proof_roundtrips_and_rejects_substitutions() {
    let model = normalization();
    let compiled = NativeModel::new(&model, b"normalization receipt".to_vec()).unwrap();
    let constant = compiled.tensor_id(8).unwrap();
    let input = compiled.input_tensor_ids()[0];
    let cache = compiled.preprocess(&SETUP).unwrap();
    let values = [tensor(&[16, -32, 48, 0], &[1, 4])];
    let (statement, witness) = cache.commit(&values, &SETUP).unwrap();
    assert_eq!(
        witness.outputs(),
        &model
            .forward(&values)
            .iter()
            .map(|t| t.inner.clone())
            .collect::<Vec<_>>()
    );
    let generators = DoryScheme::pedersen_generators(&SETUP, 64);
    let proof = NativeGraphProof::prove(&statement, witness, &SETUP, &generators).unwrap();
    let registered: NativeRegisteredGraph = roundtrip(cache.registered());
    let proof = roundtrip(&proof);
    let statement = roundtrip(&statement);
    registered.verify(&proof, &statement, &generators).unwrap();
    let (fresh, _) = cache.commit(&values, &SETUP).unwrap();
    assert_eq!(
        statement.input_commitment(constant),
        fresh.input_commitment(constant)
    );
    assert_ne!(
        statement.input_commitment(input),
        fresh.input_commitment(input)
    );
    assert!(registered.verify(&proof, &fresh, &generators).is_err());
    let mut wrong = statement.clone();
    wrong.graph.context.push(1);
    assert!(registered.verify(&proof, &wrong, &generators).is_err());
    let mut wrong = statement.clone();
    wrong.graph.outputs[0] = input;
    assert!(registered.verify(&proof, &wrong, &generators).is_err());
    let mut wrong = statement.clone();
    let output = *wrong.graph.outputs.first().unwrap();
    wrong.commitments.insert(
        common::CommittedPoly::DivNodeQuotient(output),
        *fresh.output_commitment(0).unwrap(),
    );
    assert!(registered.verify(&proof, &wrong, &generators).is_err());
    let mut other_model = model;
    other_model.graph.nodes.get_mut(&8).unwrap().operator =
        Operator::Constant(Constant(tensor(&[2], &[1, 1])));
    let other = NativeModel::new(&other_model, b"normalization receipt".to_vec())
        .unwrap()
        .preprocess(&SETUP)
        .unwrap();
    // The same valid proof must not authenticate against different weights.
    assert!(other
        .registered()
        .verify(&proof, &statement, &generators)
        .is_err());
}

#[test]
fn actual_softmax_proof_retains_auxiliary_constraints() {
    let model = unary(
        Operator::SoftmaxLastAxis(SoftmaxLastAxis { scale: 4 }),
        &[1, 2],
        &[1, 2],
    );
    let cache = NativeModel::new(&model, b"softmax receipt".to_vec())
        .unwrap()
        .preprocess(&SETUP)
        .unwrap();
    let values = [tensor(&[-4, 12], &[1, 2])];
    let (statement, witness) = cache.commit(&values, &SETUP).unwrap();
    assert_eq!(witness.outputs()[0], model.forward(&values)[0].inner);
    let generators = DoryScheme::pedersen_generators(&SETUP, 64);
    let proof = NativeGraphProof::prove(&statement, witness, &SETUP, &generators).unwrap();
    cache
        .registered()
        .verify(&proof, &statement, &generators)
        .unwrap();
    let mut missing = statement.clone();
    missing.graph.nodes.truncate(8);
    assert!(cache
        .registered()
        .verify(&proof, &missing, &generators)
        .is_err());
}

#[test]
fn malformed_models_fail_without_a_private_witness() {
    let good = unary(Operator::Identity(Identity), &[2], &[2]);
    let mut variants = vec![];
    let mut m = good.clone();
    m.graph.inputs.push(3);
    variants.push(m);
    let mut m = good.clone();
    m.graph.inputs.clear();
    variants.push(m);
    let mut m = good.clone();
    m.graph.nodes.get_mut(&7).unwrap().inputs = vec![99];
    variants.push(m);
    let mut m = good.clone();
    m.graph.nodes.get_mut(&7).unwrap().inputs = vec![7];
    variants.push(m);
    let mut m = good.clone();
    m.graph.nodes.get_mut(&7).unwrap().inputs.clear();
    variants.push(m);
    let mut m = good.clone();
    m.graph.nodes.get_mut(&7).unwrap().idx = 8;
    variants.push(m);
    let mut m = good.clone();
    m.graph.nodes.get_mut(&7).unwrap().output_dims = vec![4];
    variants.push(m);
    let mut m = good.clone();
    m.graph.nodes.get_mut(&3).unwrap().output_dims = vec![0];
    variants.push(m);
    let mut m = good.clone();
    m.graph.outputs = vec![99];
    variants.push(m);
    let mut m = good.clone();
    m.graph.original_input_dims.insert(3, vec![1]);
    variants.push(m);
    let mut m = good.clone();
    m.graph.original_output_dims.insert(7, vec![1]);
    variants.push(m);
    let mut m = good.clone();
    m.graph.nodes.get_mut(&7).unwrap().operator = Operator::MoveAxis(MoveAxis {
        source: 9,
        destination: 0,
    });
    variants.push(m);
    let mut m = good.clone();
    m.graph.nodes.get_mut(&7).unwrap().operator = Operator::Slice(Slice {
        axis: 0,
        start: 2,
        end: 1,
    });
    variants.push(m);
    let mut m = good.clone();
    m.graph.nodes.get_mut(&7).unwrap().operator = Operator::ReLU(ReLU);
    variants.push(m);
    for m in variants {
        assert!(NativeModel::new(&m, b"invalid".to_vec()).is_err());
    }
    assert!(NativeModel::new(&good, vec![]).is_err());
}

#[test]
fn unsupported_scales_and_reduction_counts_are_rejected() {
    for value in [-1, 0, 31, 256] {
        assert!(NativeModel::new(
            &unary(Operator::Square(Square { scale: value }), &[2], &[2]),
            b"scale".to_vec()
        )
        .is_err());
    }
    for (axes, count, padded) in [
        (vec![1], 0, 4),
        (vec![1], 5, 4),
        (vec![1], 3, 3),
        (vec![2], 3, 4),
        (vec![1, 1], 3, 16),
    ] {
        assert!(NativeModel::new(
            &unary(
                Operator::MeanOfSquares(MeanOfSquares {
                    axes,
                    scale: 4,
                    count,
                    padded_count: padded
                }),
                &[1, 4],
                &[1, 1]
            ),
            b"mean".to_vec()
        )
        .is_err());
    }
}

#[test]
fn commit_rejects_wrong_input_count_shape_and_storage() {
    let model = unary(Operator::Identity(Identity), &[2], &[2]);
    let cache = NativeModel::new(&model, b"inputs".to_vec())
        .unwrap()
        .preprocess(&SETUP)
        .unwrap();
    assert!(cache.commit(&[], &SETUP).is_err());
    assert!(cache.commit(&[tensor(&[1, 2], &[1, 2])], &SETUP).is_err());
    let mut malformed = tensor(&[1, 2], &[2]);
    malformed.inner.pop();
    assert!(cache.commit(&[malformed], &SETUP).is_err());
}
