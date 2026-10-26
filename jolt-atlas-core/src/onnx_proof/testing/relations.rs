//! Relation-defined ops ([`crate::onnx_proof::relation`]).
//!
//! * Op binding: every relation the prover runs is recorded with the node it ran on, and
//!   [`relation_specs_match_their_dispatched_op_kind`] checks each relation's `Spec` against the
//!   [`OpKind`] it was dispatched under, over the probe models and the op gauntlet.
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
    models::probe_models,
    op_kind::OpKind,
    zoo::{op_gauntlet_inputs, zoo},
};
use crate::{
    node_view::{ComputationNode, ModelView},
    onnx_proof::{
        fused_rebase,
        ops::mul::MulRelation,
        relation::{AtlasRelation, RelationExpr, RelationOpening},
    },
};

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
    for kind in [OpKind::Mul, OpKind::And] {
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
