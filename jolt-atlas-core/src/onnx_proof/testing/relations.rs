//! Relation-defined ops ([`crate::onnx_proof::relation`]).
//!
//! * Op binding and completeness: every relation the prover runs is recorded with the node it
//!   ran on, and [`relation_specs_match_their_dispatched_op_kind`] checks each relation's `Spec`
//!   against the [`OpKind`] it was dispatched under, over the probe models and the op gauntlet.
//!   Every op has a [`disposition`]; the ops marked [`Disposition::Relation`] are exactly the
//!   ones that ran through relations.
//! * Prover equivalence: every prover of a relation, the engine's and the specialized ones,
//!   produces the relation's reference round messages (`relation_prover_equiv_*`), computed
//!   from its expressions alone by summing the output expression over the hypercube.
//! * Claim flows: the `relation_flows` probe model proves, verifies and is swept with the other
//!   probe models; the tests below pin the structure that makes it exercise each flow, so it
//!   cannot stop doing so silently.

use atlas_algebra::field::fp128::Fp128 as Fr;
use atlas_claims::{Source, SymbolicSumcheck};
use atlas_pcs::akita::AkitaScheme;
use atlas_sumcheck::opening_proof::{OpeningId, SumcheckId};
use atlas_trace::Op;
use common::VirtualPoly;
use std::{ops::Range, sync::Mutex};
use strum::IntoEnumIterator;

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
            div::DivRelation,
            einsum::{dot::EinsumDotProver, ContractionRelation},
            iff::IffRelation,
            mean_of_squares::{MeanOfSquaresReductionProver, MeanOfSquaresRelation},
            mul::MulRelation,
            rsqrt::RsqrtRelation,
            scalar_const_div::ConstantDivisionRelation,
            square::SquareRelation,
            sum::axis::SumAxisRelation,
        },
        relation::{
            AtlasRelation, Relation, RelationChallenge, RelationExpr, RelationOpening,
            RelationParams, RelationProver, RelationSetup, RelationWitnesses,
        },
        trace_view::TraceView,
    },
};
use atlas_algebra::field::Ring;
use atlas_algebra::{
    poly::multilinear::{MultilinearPolynomial, PolynomialEvaluation},
    utils::math::Math,
};
use atlas_sumcheck::engine::prover::ProveRounds;
use atlas_trace::{trace::LayerData, Model, Tensor, Trace};
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
        prove_verify_roundtrip::<AkitaScheme>(probe.model, &probe.inputs, Default::default());
    }
    let gauntlet = zoo("op_gauntlet", 16);
    let inputs = op_gauntlet_inputs(&gauntlet);
    prove_verify_roundtrip::<AkitaScheme>(gauntlet.model, &inputs, Default::default());

    let dispatches = DISPATCHES.lock().expect("dispatch log");
    for &(kind, spec) in dispatches.iter() {
        let spec_name = spec.rsplit("::").next().expect("a type name");
        assert_eq!(spec_name, kind.to_string(), "{spec} dispatched for {kind}");
    }
    let relation_kinds: Vec<OpKind> = OpKind::iter()
        .filter(|&kind| disposition(kind) == Disposition::Relation)
        .collect();
    for &(kind, spec) in dispatches.iter() {
        assert!(
            relation_kinds.contains(&kind),
            "{kind} ran through {spec} but its disposition is {:?}",
            disposition(kind)
        );
    }
    for kind in relation_kinds {
        assert!(
            dispatches.iter().any(|&(k, _)| k == kind),
            "no {kind} node ran through its relation"
        );
    }
}

/// How an op is proven once stage 6 of `specs/adoption-program.md` closes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Disposition {
    /// Its algebraic protocol is a relation (`onnx_proof::relation`); the verifier is derived
    /// from it. Any lookup or range check it also needs stays with the lookup family.
    Relation,
    /// A lookup-family protocol (read-raf lookups over one-hot addresses, clamp and activation
    /// tables): kept as code and moved as code at stage 8.
    LookupFamily,
}

/// Every op's disposition. Exhaustive on purpose: a new op variant does not compile until it
/// has one.
fn disposition(kind: OpKind) -> Disposition {
    use Disposition::*;
    match kind {
        OpKind::Input
        | OpKind::Constant
        | OpKind::Identity
        | OpKind::Add
        | OpKind::Sub
        | OpKind::Neg
        | OpKind::Mul
        | OpKind::Square
        | OpKind::Cube
        | OpKind::Div
        | OpKind::ScalarConstDiv
        | OpKind::Einsum
        | OpKind::Sum
        | OpKind::MeanOfSquares
        | OpKind::Rsqrt
        | OpKind::Iff
        | OpKind::And
        | OpKind::IsNan
        | OpKind::Slice
        | OpKind::Concat
        | OpKind::Reshape
        | OpKind::Broadcast
        | OpKind::MoveAxis
        | OpKind::SoftmaxLastAxis => Relation,
        OpKind::Tanh
        | OpKind::Sigmoid
        | OpKind::Erf
        | OpKind::Sin
        | OpKind::Cos
        | OpKind::ReLU
        | OpKind::SmoothRelu
        | OpKind::Clamp
        | OpKind::GatherSmall
        | OpKind::GatherLarge => LookupFamily,
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
fn fused_product_reads_its_output_and_remainder() {
    let (_, muls) = mul_nodes();
    let fused = muls
        .iter()
        .find(|n| fused_rebase::fuses_rebase(n))
        .expect("a fused product");
    let remainder = RelationOpening::Id(OpeningId::new(
        VirtualPoly::RescaleRemainder(fused.idx),
        SumcheckId::NodeExecution(fused.idx),
    ));
    let input = openings(&MulRelation::new(fused.clone()).input_expression::<Fr>());
    assert_eq!(
        input,
        vec![RelationOpening::ReducedNodeOutput(fused.idx), remainder]
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
fn reference_messages<R: AtlasRelation + Relation<Fr>>(
    relation: &R,
    setup: &RelationSetup<Fr>,
    open: &dyn Fn(usize, &[Fr]) -> Fr,
    challenges: &[Fr],
) -> Vec<Vec<Fr>> {
    let produced = relation.produced();
    let output = relation.output_expression::<Fr>();
    let summand = |point: &[Fr]| {
        let points = relation.opening_points(setup, point).unwrap();
        let values: Vec<Fr> = (0..produced.len()).map(|i| open(i, &points[i])).collect();
        output.evaluate(
            |opening| match opening {
                RelationOpening::Id(id) => values[produced.iter().position(|p| p == id).unwrap()],
                RelationOpening::ReducedNodeOutput(node) => {
                    panic!("node {node}'s reduced output is not produced")
                }
            },
            |challenge| setup.challenge(challenge),
            |derived| relation.derive(derived, setup, point).unwrap(),
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
fn assert_matches_reference<R: AtlasRelation + Relation<Fr>>(
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

/// A one-node model: its inputs' dims and value ranges, and how to build the node over them.
type SingleNodeCase = (
    Vec<(Vec<usize>, Range<i32>)>,
    fn(&mut ModelBuilder, &[usize]) -> usize,
);

/// Positive values small enough for any op's arithmetic.
const SMALL: Range<i32> = 1..(1 << 7);

fn random_point(rng: &mut StdRng, len: usize) -> Vec<Fr> {
    (0..len)
        .map(|_| <Fr as atlas_algebra::field::Field>::random(rng))
        .collect()
}

/// One node of `op` built by `build` over inputs of the given dims, traced on random inputs.
fn single_node(
    seed: u64,
    inputs: &[(Vec<usize>, Range<i32>)],
    build: impl Fn(&mut ModelBuilder, &[usize]) -> usize,
) -> (Model, Trace, ComputationNode, StdRng) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut b = ModelBuilder::new();
    let wires: Vec<usize> = inputs.iter().map(|(d, _)| b.input(d.clone())).collect();
    let node = build(&mut b, &wires);
    b.mark_output(node);
    let model = b.build();
    let values: Vec<Tensor<i32>> = inputs
        .iter()
        .map(|(d, range)| Tensor::random_range(&mut rng, d, range.clone()))
        .collect();
    let trace = model.trace(&values).expect("honest execution");
    let node = model.view(node);
    (model, trace, node, rng)
}

#[test]
fn relation_prover_equiv_engine() {
    let cases: Vec<SingleNodeCase> = vec![
        (vec![(vec![8], SMALL), (vec![8], SMALL)], |b, i| {
            b.mul(i[0], i[1])
        }),
        (vec![(vec![8], SMALL)], |b, i| b.square(i[0])),
        (vec![(vec![8], SMALL)], |b, i| b.cube(i[0], 0)),
        (vec![(vec![8], SMALL)], |b, i| b.rsqrt(i[0])),
        (
            vec![(vec![8], -(1 << 7)..(1 << 7)), (vec![8], SMALL)],
            |b, i| b.div(i[0], i[1]),
        ),
        (
            vec![(vec![8], 0..2), (vec![8], SMALL), (vec![8], SMALL)],
            |b, i| b.iff(i[0], i[1], i[2]),
        ),
        (vec![(vec![8], -(1 << 7)..(1 << 7))], |b, i| {
            b.scalar_const_div(i[0], 7)
        }),
    ];
    for (seed, (dims, build)) in (0xA17_0601..).zip(cases) {
        let (_, trace, node, mut rng) = single_node(seed, &dims, build);
        let rng = &mut rng;
        match node.operator {
            Op::Mul(_) => check_engine(|| MulRelation::new(node.clone()), &trace, rng),
            Op::Square(_) => check_engine(|| SquareRelation::new(node.clone()), &trace, rng),
            Op::Cube(_) => check_engine(|| CubeRelation::new(node.clone()), &trace, rng),
            Op::Rsqrt(_) => check_engine(|| RsqrtRelation::new(node.clone()), &trace, rng),
            Op::Div(_) => check_engine(|| DivRelation::new(node.clone()), &trace, rng),
            Op::Iff(_) => check_engine(|| IffRelation::new(node.clone()), &trace, rng),
            Op::ScalarConstDiv(_) => {
                check_engine(|| ConstantDivisionRelation::new(node.clone()), &trace, rng)
            }
            ref other => panic!("no engine relation for {other}"),
        }
    }
}

/// The engine's messages for `relation()` match its reference messages.
fn check_engine<R: RelationWitnesses<Fr>>(
    relation: impl Fn() -> R,
    trace: &Trace,
    rng: &mut StdRng,
) {
    let rounds = relation().rounds();
    let setup = RelationSetup {
        incoming_point: random_point(rng, rounds),
        challenges: vec![(RelationChallenge::Gamma, random_point(rng, 1)[0])],
    };
    let challenges = random_point(rng, rounds);
    let witnesses = relation().witnesses(trace, &setup);
    let open = |i: usize, point: &[Fr]| witnesses[i].evaluate(point);
    let mut prover =
        RelationProver::initialize(trace, RelationParams::at(relation(), setup.clone()));
    assert_matches_reference(&mut prover, &relation(), &setup, &open, &challenges);
}

#[test]
fn relation_prover_equiv_dense() {
    let cases: Vec<SingleNodeCase> = vec![
        (vec![(vec![4, 8], SMALL)], |b, i| {
            b.sum(i[0], vec![1], vec![4])
        }),
        (vec![(vec![4, 8], SMALL)], |b, i| {
            b.sum(i[0], vec![0], vec![8])
        }),
    ];
    for (seed, (dims, build)) in (0xA17_0621..).zip(cases) {
        let (model, trace, node, mut rng) = single_node(seed, &dims, build);
        let rng = &mut rng;
        // Produced openings are operands, each read through its own MLE at its own point.
        let LayerData { operands, .. } = TraceView::layer_data(&trace, &node);
        let operands: Vec<MultilinearPolynomial<Fr>> = operands
            .iter()
            .map(|t| MultilinearPolynomial::from(t.padded_pow2()))
            .collect();
        let open = |i: usize, point: &[Fr]| operands[i].evaluate(point);
        match node.operator {
            Op::Sum(_) => check_dense(
                || {
                    SumAxisRelation::for_node(&model, node.clone())
                        .expect("the model\'s nodes have the shapes their ops prove")
                },
                &trace,
                &node,
                &open,
                rng,
            ),
            ref other => panic!("no dense relation for {other}"),
        }
    }
}

/// The engine's messages for a dense `relation()` match its reference messages.
fn check_dense<R: RelationWitnesses<Fr>>(
    relation: impl Fn() -> R,
    trace: &Trace,
    node: &ComputationNode,
    open: &dyn Fn(usize, &[Fr]) -> Fr,
    rng: &mut StdRng,
) {
    let setup = RelationSetup {
        incoming_point: random_point(rng, node.pow2_padded_num_output_elements().log_2()),
        challenges: Vec::new(),
    };
    let challenges = random_point(rng, relation().rounds());
    let mut prover =
        RelationProver::initialize(trace, RelationParams::at(relation(), setup.clone()));
    assert_matches_reference(&mut prover, &relation(), &setup, open, &challenges);
}

#[test]
fn relation_prover_equiv_mean_of_squares() {
    let (_, trace, node, mut rng) = single_node(0xA17_0631, &[(vec![8, 4], SMALL)], |b, i| {
        b.mean_of_squares(i[0], vec![1], vec![8, 1])
    });
    let relation = || MeanOfSquaresRelation::new(node.clone());
    let setup = RelationSetup {
        incoming_point: random_point(&mut rng, node.pow2_padded_num_output_elements().log_2()),
        challenges: Vec::new(),
    };
    let challenges = random_point(&mut rng, relation().rounds());
    let LayerData { operands, .. } = TraceView::layer_data(&trace, &node);
    let operand = MultilinearPolynomial::<Fr>::from(operands[0].padded_pow2());
    let open = |_: usize, point: &[Fr]| operand.evaluate(point);
    let mut prover = MeanOfSquaresReductionProver::initialize(
        &trace,
        RelationParams::at(relation(), setup.clone()),
    );
    assert_matches_reference(&mut prover, &relation(), &setup, &open, &challenges);
}

#[test]
fn relation_prover_equiv_contraction() {
    let cases: Vec<SingleNodeCase> = vec![
        (vec![(vec![4, 8], SMALL), (vec![8, 2], SMALL)], |b, i| {
            b.einsum("mk,kn->mn", vec![i[0], i[1]], vec![4, 2])
        }),
        (vec![(vec![8], SMALL), (vec![4, 8], SMALL)], |b, i| {
            b.einsum("k,nk->n", vec![i[0], i[1]], vec![4])
        }),
        (
            vec![(vec![2, 2, 2, 4], SMALL), (vec![2, 2, 2, 4], SMALL)],
            |b, i| b.einsum("abmk,abnk->abmn", vec![i[0], i[1]], vec![2, 2, 2, 2]),
        ),
    ];
    for (seed, (dims, build)) in (0xA17_0611..).zip(cases) {
        let (model, trace, node, mut rng) = single_node(seed, &dims, build);
        let relation = || {
            ContractionRelation::for_node(&model, node.clone())
                .expect("the model\'s nodes have the shapes their ops prove")
        };
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

/// The three layout ties hold on models compiled with padding: for a Reshape, an aligned Slice
/// and an aligned Concat over dims that are not powers of two, the output's extension at a
/// random point is the tie's combination of the operands' extensions at its opening points.
#[test]
fn layout_ties_hold_on_padded_models() {
    use crate::onnx_proof::ops::{
        concat::SubcubeTilingRelation, reshape::FlatCopyRelation, slice::SubcubeSelectRelation,
    };
    use atlas_algebra::poly::multilinear::EvaluateMle;
    use atlas_sumcheck::relation::RelationDerived;
    use rand::Rng;

    type Build = Box<dyn Fn(&mut ModelBuilder, &[usize]) -> usize>;
    let mut rng = StdRng::seed_from_u64(0x7_1E5);
    for _ in 0..60 {
        let lead = rng.gen_range(1..7usize);
        let other = rng.gen_range(1..7usize);
        let (inputs, build): (Vec<Vec<usize>>, Build) = match rng.gen_range(0..3) {
            // `[lead, 2^k] -> [lead·2^j, 2^(k−j)]`: the inner axes are powers of two.
            0 => {
                let k = rng.gen_range(1..4u32);
                let j = rng.gen_range(0..=k);
                let shape = vec![lead << j, 1 << (k - j)];
                (
                    vec![vec![lead, 1 << k]],
                    Box::new(move |b, i| b.reshape(i[0], shape.clone())),
                )
            }
            // A power-of-two extent at a multiple of itself, on a random axis.
            1 => {
                let size = rng.gen_range(2..10usize);
                let extent = 1 << rng.gen_range(0..=size.ilog2());
                let start = extent * rng.gen_range(0..size / extent);
                let axis = rng.gen_range(0..2usize);
                let mut dims = vec![other, other];
                dims[axis] = size;
                (
                    vec![dims],
                    Box::new(move |b, i| b.slice(i[0], axis, start, start + extent)),
                )
            }
            // Operands whose padded extents tile the padded axis.
            _ => {
                let layouts: [&[usize]; 6] = [
                    &[4, 4],
                    &[4, 3],
                    &[8, 4, 4],
                    &[8, 4, 3],
                    &[2, 2, 2, 2],
                    &[4, 2, 1, 1],
                ];
                let extents = layouts[rng.gen_range(0..layouts.len())];
                let axis = rng.gen_range(0..2usize);
                let dims = extents
                    .iter()
                    .map(|&extent| {
                        let mut dims = vec![other, other];
                        dims[axis] = extent;
                        dims
                    })
                    .collect();
                (dims, Box::new(move |b, i| b.concat(i, axis as isize)))
            }
        };
        let mut b = ModelBuilder::new();
        let wires: Vec<usize> = inputs.iter().map(|d| b.input(d.clone())).collect();
        let node = build(&mut b, &wires);
        b.mark_output(node);
        let model = b.build_padded();
        let values: Vec<Tensor<i32>> = inputs
            .iter()
            .map(|d| Tensor::random_range(&mut rng, d, SMALL))
            .collect();
        let trace = model.trace(&values).expect("honest execution");
        let node = model.view(node);
        let LayerData { operands, output } = TraceView::layer_data(&trace, &node);

        let r = random_point(&mut rng, node.pow2_padded_num_output_elements().log_2());
        let setup = RelationSetup {
            incoming_point: r.clone(),
            challenges: Vec::new(),
        };
        let expect = "a model compiled with padding has the dims the ties hold for";
        let (points, coefficients): (Vec<Vec<Fr>>, Vec<Fr>) = match &node.operator {
            Op::Reshape(_) => {
                FlatCopyRelation::for_node(&model, node.clone()).expect(expect);
                (vec![r.clone()], vec![Fr::from_u64(1)])
            }
            Op::Slice(op) => {
                let relation =
                    SubcubeSelectRelation::for_node(&model, node.clone(), op).expect(expect);
                (
                    vec![relation.operand_point(&r).expect(expect)],
                    vec![Fr::from_u64(1)],
                )
            }
            Op::Concat(op) => {
                let relation =
                    SubcubeTilingRelation::for_node(&model, node.clone(), op).expect(expect);
                let coefficients = (0..operands.len())
                    .map(|i| {
                        relation
                            .derive(&RelationDerived::BlockEq(i), &setup, &[])
                            .expect(expect)
                    })
                    .collect();
                (relation.operand_points(&r).expect(expect), coefficients)
            }
            other => panic!("not a layout op: {other}"),
        };
        let combined: Fr = operands
            .iter()
            .zip(&points)
            .zip(&coefficients)
            .map(|((operand, point), coefficient)| *coefficient * operand.evaluate_mle(point))
            .sum();
        assert_eq!(
            output.evaluate_mle(&r),
            combined,
            "{} over {inputs:?}",
            node.operator
        );
    }
}

/// The verifier rejects, on its own, a model whose layout ops do not have the dims and the
/// alignment their ties hold for: such a model was not compiled with padding.
#[test]
fn layout_ops_outside_their_ties_are_rejected() {
    use crate::node_view::check_model_shape;

    let cases: Vec<(&str, SingleNodeCase)> = vec![
        (
            "is not a power of two",
            (vec![(vec![10, 10], SMALL)], |b, i| {
                b.reshape(i[0], vec![20, 5])
            }),
        ),
        (
            "start 8 is not a multiple of the extent 32",
            (vec![(vec![64], SMALL)], |b, i| b.slice(i[0], 0, 8, 40)),
        ),
        (
            "offset 4 is not a multiple of the operand extent 8",
            (
                vec![(vec![4], SMALL), (vec![8], SMALL), (vec![4], SMALL)],
                |b, i| b.concat(i, 0),
            ),
        ),
    ];
    for (reason, (dims, build)) in cases {
        let (model, ..) = single_node(0x7_1E6, &dims, build);
        let err = check_model_shape(&model).expect_err("the model is outside the ties");
        assert!(err.to_string().contains(reason), "{err}");
    }
}

/// The verifier's relations are those of `SemVersion::V3`: a model compiled under an earlier
/// version is rejected before any proof is read.
#[test]
fn an_earlier_version_model_is_rejected() {
    use crate::node_view::check_model_shape;
    use atlas_trace::{CompileOptions, GraphBuilder, Scale, SemVersion};

    let build = |version| {
        let mut b = GraphBuilder::new();
        let x = b.input(&[4]);
        let y = b.push(Op::Neg(atlas_trace::op::Neg), &[x]);
        b.output(y);
        b.build(CompileOptions {
            scale: Scale::new(14).unwrap(),
            pad_pow2: true,
            version,
        })
        .expect("the graph compiles")
    };
    check_model_shape(&build(SemVersion::V3)).expect("the current version is accepted");
    for version in [SemVersion::V1, SemVersion::V2] {
        let err = check_model_shape(&build(version)).expect_err("an earlier version");
        assert!(err.to_string().contains("semantics version"), "{err}");
    }
}

/// A softmax proof carries nothing per row: its opening claims are as many for sixteen rows as
/// for two.
#[test]
fn softmax_proof_has_no_per_row_claims() {
    use crate::onnx_proof::{prove, AtlasProverPreprocessing, AtlasSharedPreprocessing};
    use atlas_transcript::Blake2bProver;

    let claims = |rows: usize| {
        let mut rng = StdRng::seed_from_u64(0x50F7);
        let mut b = ModelBuilder::new();
        let x = b.input(vec![rows, 8]);
        let out = b.softmax_last_axis(x);
        b.mark_output(out);
        let inputs = [Tensor::random_range(&mut rng, &[rows, 8], SMALL)];
        let pp = AtlasProverPreprocessing::<Fr, AkitaScheme>::new(
            AtlasSharedPreprocessing::preprocess(b.build()),
        );
        let (proof, _, _) = prove::<Fr, Blake2bProver, AkitaScheme>(&pp, &inputs);
        proof.opening_claims().openings().len()
    };
    assert_eq!(claims(2), claims(16));
}
