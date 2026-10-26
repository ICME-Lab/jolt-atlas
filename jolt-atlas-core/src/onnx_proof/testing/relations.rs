//! Relation-defined ops ([`crate::onnx_proof::relation`]) in the claim flows of the
//! `relation_flows` probe model. The model proves, verifies and is swept with the other probe
//! models; these tests pin the structure that makes it exercise each flow, so it cannot stop
//! doing so silently.

use atlas_trace::Op;
use common::VirtualPoly;
use jolt_claims::{Source, SymbolicSumcheck};
use jolt_field::Fr;
use joltworks::poly::opening_proof::{OpeningId, SumcheckId};

use super::models::probe_models;
use crate::{
    node_view::{ComputationNode, ModelView},
    onnx_proof::{
        fused_rebase,
        ops::mul::MulRelation,
        relation::{RelationExpr, RelationOpening},
    },
};

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
