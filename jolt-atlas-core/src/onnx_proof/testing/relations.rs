//! Relation-defined ops ([`crate::onnx_proof::relation`]).
//!
//! * Op binding: every relation the prover runs is recorded with the node it ran on, and
//!   [`relation_specs_match_their_dispatched_op_kind`] checks each relation's `Spec` against the
//!   [`OpKind`] it was dispatched under, over the probe models and the op gauntlet.
//! * Prover equivalence: every prover of a relation, the engine's and the specialized ones,
//!   produces the relation's reference round messages (`relation_prover_equiv_*`), computed
//!   from its expressions alone by summing the output expression over the hypercube.
//! * Claim flows: the `relation_flows` probe model proves, verifies and is swept with the other
//!   probe models; the tests below pin the structure that makes it exercise each flow, so it
//!   cannot stop doing so silently.

use ark_bn254::Bn254;
use atlas_trace::Op;
use common::VirtualPoly;
use jolt_claims::{Source, SymbolicSumcheck};
use jolt_field::Fr;
use joltworks::poly::commitment::hyperkzg::HyperKZG;
use joltworks::poly::opening_proof::{OpeningId, SumcheckId};
use std::sync::Mutex;

use super::{
    harness::prove_verify_roundtrip,
    model_builder::ModelBuilder,
    models::probe_models,
    op_kind::OpKind,
    zoo::{op_gauntlet_inputs, zoo},
};
use crate::{
    node_view::{ComputationNode, ModelView},
    onnx_proof::{
        fused_rebase,
        ops::{
            cube::CubeRelation,
            einsum::{dot::EinsumDotProver, ContractionRelation},
            mul::MulRelation,
            rsqrt::RsqrtRelation,
            square::SquareRelation,
        },
        relation::{
            AtlasRelation, EngineRelation, RelationChallenge, RelationExpr, RelationOpening,
            RelationParams, RelationProver, RelationSetup,
        },
        trace_view::TraceView,
    },
};
use atlas_trace::{trace::LayerData, Model, Tensor, Trace};
use jolt_field::Ring;
use jolt_sumcheck::ProveRounds;
use joltworks::{
    field::JoltField,
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    utils::math::Math,
};
use rand::{rngs::StdRng, SeedableRng};

/// `(op kind of the node, the relation's Spec type)` for every relation the prover has run.
static DISPATCHES: Mutex<Vec<(OpKind, &'static str)>> = Mutex::new(Vec::new());

/// Record that relation `R` was dispatched for `node`.
pub(crate) fn record_dispatch<R: AtlasRelation>(node: &ComputationNode) {
    DISPATCHES
        .lock()
        .expect("dispatch log")
        .push((OpKind::of(&node.operator), std::any::type_name::<R::Spec>()));
}

#[test]
fn relation_specs_match_their_dispatched_op_kind() {
    for probe in probe_models() {
        prove_verify_roundtrip::<HyperKZG<Bn254>>(probe.model, &probe.inputs, Default::default());
    }
    let gauntlet = zoo("op_gauntlet", 16);
    let inputs = op_gauntlet_inputs(&gauntlet);
    prove_verify_roundtrip::<HyperKZG<Bn254>>(gauntlet.model, &inputs, Default::default());

    let dispatches = DISPATCHES.lock().expect("dispatch log");
    for &(kind, spec) in dispatches.iter() {
        let spec_name = spec.rsplit("::").next().expect("a type name");
        assert_eq!(spec_name, kind.to_string(), "{spec} dispatched for {kind}");
    }
    for kind in [
        OpKind::Mul,
        OpKind::And,
        OpKind::Square,
        OpKind::Cube,
        OpKind::Rsqrt,
        OpKind::Einsum,
    ] {
        assert!(
            dispatches.iter().any(|&(k, _)| k == kind),
            "no {kind} node ran through its relation"
        );
    }
}

fn mul_nodes() -> (Vec<ComputationNode>, Vec<ComputationNode>) {
    let probe = probe_models()
        .into_iter()
        .find(|p| p.name == "relation_flows")
        .expect("relation_flows probe model");
    let nodes = probe.model.views();
    let muls = nodes
        .iter()
        .filter(|n| matches!(n.operator, Op::Mul(_)))
        .cloned()
        .collect();
    (nodes, muls)
}

fn openings(expr: &RelationExpr<Fr>) -> Vec<RelationOpening> {
    expr.terms
        .iter()
        .flat_map(|term| &term.factors)
        .filter_map(|factor| match factor {
            Source::Opening(opening) => Some(*opening),
            _ => None,
        })
        .collect()
}

fn consumers(nodes: &[ComputationNode], node: usize) -> Vec<&ComputationNode> {
    nodes.iter().filter(|n| n.inputs.contains(&node)).collect()
}

#[test]
fn fused_product_reads_its_advice_by_node_id() {
    let (_, muls) = mul_nodes();
    let fused = muls
        .iter()
        .find(|n| fused_rebase::fuses_rebase(n))
        .expect("a fused product");
    let advice = |poly: fn(usize) -> VirtualPoly| {
        RelationOpening::Id(OpeningId::new(
            poly(fused.idx),
            SumcheckId::NodeExecution(fused.idx),
        ))
    };
    let input = openings(&MulRelation::new(fused.clone()).input_expression::<Fr>());
    assert_eq!(
        input,
        vec![
            advice(VirtualPoly::ClampAcc),
            advice(VirtualPoly::RescaleRemainder)
        ]
    );
}

#[test]
fn raw_product_reads_a_multi_consumer_reduction() {
    let (nodes, muls) = mul_nodes();
    let raw = muls
        .iter()
        .find(|n| !fused_rebase::fuses_rebase(n))
        .expect("a raw product");
    assert!(
        consumers(&nodes, raw.idx).len() >= 2,
        "the reduction folds several claims"
    );
    let input = openings(&MulRelation::new(raw.clone()).input_expression::<Fr>());
    assert_eq!(input, vec![RelationOpening::ReducedNodeOutput(raw.idx)]);
}

#[test]
fn produced_opening_joins_a_shared_operand_reduction() {
    let (nodes, muls) = mul_nodes();
    let shared = muls
        .iter()
        .flat_map(|m| m.inputs.iter().copied())
        .find(|&operand| {
            let users = consumers(&nodes, operand);
            users.iter().any(|n| matches!(n.operator, Op::Mul(_)))
                && users.iter().any(|n| !matches!(n.operator, Op::Mul(_)))
        })
        .expect("an operand shared by a product and another op");
    for mul in muls.iter().filter(|m| m.inputs.contains(&shared)) {
        let produced = openings(&MulRelation::new(mul.clone()).output_expression::<Fr>());
        let operand = RelationOpening::Id(OpeningId::new(
            VirtualPoly::NodeOutput(shared),
            SumcheckId::NodeExecution(mul.idx),
        ));
        assert!(produced.contains(&operand), "{produced:?}");
    }
}

/// A relation's round messages computed from its expressions alone, as evaluations at
/// `0..=degree`: round `j`'s message at `x` sums the output expression over every point whose
/// first `j` coordinates are `challenges[..j]`, whose `j`-th is `x`, and whose rest is Boolean,
/// with each produced opening read through `open(position, point)`.
fn reference_messages<R: AtlasRelation>(
    relation: &R,
    setup: &RelationSetup<Fr>,
    open: &dyn Fn(usize, &[Fr]) -> Fr,
    challenges: &[Fr],
) -> Vec<Vec<Fr>> {
    let produced = relation.produced();
    let output = relation.output_expression::<Fr>();
    let summand = |point: &[Fr]| {
        let points = relation.opening_points(setup, point);
        let values: Vec<Fr> = (0..produced.len()).map(|i| open(i, &points[i])).collect();
        output.evaluate(
            |opening| match opening {
                RelationOpening::Id(id) => values[produced.iter().position(|p| p == id).unwrap()],
                RelationOpening::ReducedNodeOutput(node) => {
                    panic!("node {node}'s reduced output is not produced")
                }
            },
            |challenge| setup.challenge(challenge),
            |derived| relation.derive(derived, setup, point),
        )
    };
    let rounds = relation.rounds();
    (0..rounds)
        .map(|round| {
            let free = rounds - round - 1;
            (0..=relation.degree() as u64)
                .map(|x| {
                    (0..1usize << free)
                        .map(|bits| {
                            let mut point = challenges[..round].to_vec();
                            point.push(Fr::from_u64(x));
                            point.extend(
                                (0..free)
                                    .map(|i| Fr::from_u64(((bits >> (free - 1 - i)) & 1) as u64)),
                            );
                            summand(&point)
                        })
                        .sum()
                })
                .collect()
        })
        .collect()
}

/// Drive `member` on `challenges` and check every round message against the relation's
/// reference messages.
fn assert_matches_reference<R: AtlasRelation>(
    member: &mut dyn ProveRounds<Fr>,
    relation: &R,
    setup: &RelationSetup<Fr>,
    open: &dyn Fn(usize, &[Fr]) -> Fr,
    challenges: &[Fr],
) {
    let reference = reference_messages(relation, setup, open, challenges);
    let mut claim = reference[0][0] + reference[0][1];
    let mut bind = None;
    for (round, expected) in reference.iter().enumerate() {
        let message = member
            .prove_round(bind, round, claim)
            .unwrap_or_else(|e| panic!("round {round}: {e:?}"));
        for (x, value) in expected.iter().enumerate() {
            assert_eq!(
                message.evaluate(Fr::from_u64(x as u64)),
                *value,
                "round {round}, x = {x}"
            );
        }
        claim = message.evaluate(challenges[round]);
        bind = Some(challenges[round]);
    }
}

/// A one-node model: its input dims and how to build the node over the inputs.
type SingleNodeCase = (Vec<Vec<usize>>, fn(&mut ModelBuilder, &[usize]) -> usize);

fn random_point(rng: &mut StdRng, len: usize) -> Vec<Fr> {
    (0..len)
        .map(|_| <Fr as JoltField>::Challenge::random(rng).into())
        .collect()
}

/// One node of `op` built by `build` over inputs of the given dims, traced on random inputs.
fn single_node(
    seed: u64,
    input_dims: &[Vec<usize>],
    build: impl Fn(&mut ModelBuilder, &[usize]) -> usize,
) -> (Model, Trace, ComputationNode, StdRng) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut b = ModelBuilder::new();
    let inputs: Vec<usize> = input_dims.iter().map(|d| b.input(d.clone())).collect();
    let node = build(&mut b, &inputs);
    b.mark_output(node);
    let model = b.build();
    let values: Vec<Tensor<i32>> = input_dims
        .iter()
        .map(|d| Tensor::random_range(&mut rng, d, 1..(1 << 7)))
        .collect();
    let trace = model.trace(&values).expect("honest execution");
    let node = model.view(node);
    (model, trace, node, rng)
}

#[test]
fn relation_prover_equiv_engine() {
    let cases: Vec<SingleNodeCase> = vec![
        (vec![vec![8], vec![8]], |b, i| b.mul(i[0], i[1])),
        (vec![vec![8]], |b, i| b.square(i[0])),
        (vec![vec![8]], |b, i| b.cube(i[0], 0)),
        (vec![vec![8]], |b, i| b.rsqrt(i[0])),
    ];
    for (seed, (dims, build)) in (0xA17_0601..).zip(cases) {
        let (_, trace, node, mut rng) = single_node(seed, &dims, build);
        let rng = &mut rng;
        match node.operator {
            Op::Mul(_) => check_engine(|| MulRelation::new(node.clone()), &trace, rng),
            Op::Square(_) => check_engine(|| SquareRelation::new(node.clone()), &trace, rng),
            Op::Cube(_) => check_engine(|| CubeRelation::new(node.clone()), &trace, rng),
            Op::Rsqrt(_) => check_engine(|| RsqrtRelation::new(node.clone()), &trace, rng),
            ref other => panic!("no engine relation for {other}"),
        }
    }
}

/// The engine's messages for `relation()` match its reference messages.
fn check_engine<R: EngineRelation>(relation: impl Fn() -> R, trace: &Trace, rng: &mut StdRng) {
    let rounds = relation().rounds();
    let setup = RelationSetup {
        incoming_point: random_point(rng, rounds),
        challenges: vec![(RelationChallenge::Gamma, random_point(rng, 1)[0])],
    };
    let challenges = random_point(rng, rounds);
    let witnesses = relation().witnesses::<Fr>(trace);
    let open = |i: usize, point: &[Fr]| witnesses[i].evaluate(point);
    let mut prover =
        RelationProver::initialize(trace, RelationParams::at(relation(), setup.clone()));
    assert_matches_reference(&mut prover, &relation(), &setup, &open, &challenges);
}

#[test]
fn relation_prover_equiv_contraction() {
    let cases: Vec<SingleNodeCase> = vec![
        (vec![vec![4, 8], vec![8, 2]], |b, i| {
            b.einsum("mk,kn->mn", vec![i[0], i[1]], vec![4, 2])
        }),
        (vec![vec![8], vec![4, 8]], |b, i| {
            b.einsum("k,nk->n", vec![i[0], i[1]], vec![4])
        }),
        (vec![vec![2, 2, 2, 4], vec![2, 2, 2, 4]], |b, i| {
            b.einsum("abmk,abnk->abmn", vec![i[0], i[1]], vec![2, 2, 2, 2])
        }),
    ];
    for (seed, (dims, build)) in (0xA17_0611..).zip(cases) {
        let (model, trace, node, mut rng) = single_node(seed, &dims, build);
        let relation = || ContractionRelation::for_node(&model, node.clone());
        let setup = RelationSetup {
            incoming_point: random_point(&mut rng, node.pow2_padded_num_output_elements().log_2()),
            challenges: Vec::new(),
        };
        let challenges = random_point(&mut rng, relation().rounds());
        let LayerData { operands, .. } = TraceView::layer_data(&trace, &node);
        let operands: Vec<MultilinearPolynomial<Fr>> = operands
            .iter()
            .map(|t| MultilinearPolynomial::from(t.padded_pow2()))
            .collect();
        let open = |i: usize, point: &[Fr]| operands[i].evaluate(point);
        let mut prover =
            EinsumDotProver::initialize(&trace, RelationParams::at(relation(), setup.clone()));
        assert_matches_reference(&mut prover, &relation(), &setup, &open, &challenges);
    }
}
