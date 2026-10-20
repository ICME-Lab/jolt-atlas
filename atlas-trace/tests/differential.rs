//! Differential tests against `atlas-onnx-tracer`: the old crate is the
//! behavioral reference.
//!
//! - **Exact class** (structural ops, saturating arithmetic, fused rescale,
//!   gather, iff/and, div, rsqrt): outputs and fused intermediates must be
//!   byte-identical.
//! - **Table class** (tanh/sigmoid/erf/sin/cos, softmax): the new crate
//!   intentionally generates its lookup tables with deterministic integer
//!   arithmetic, so entries may differ from the old f64-generated tables by
//!   one ulp on rounding ties; outputs are compared within that tolerance.

use atlas_onnx_tracer::{
    model::{ComputationGraph, Model as OldModel},
    node::ComputationNode,
    ops as old_ops,
    ops::Operator,
    tensor::Tensor as OldTensor,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::{BTreeMap, HashMap};

use atlas_trace::{CompileOptions, Graph, Model, Node, NodeId, Op, Scale, SemVersion, Tensor, op};

const SCALE: u8 = 14;

/// Minimal builder over the old crate's public graph types.
struct OldBuilder {
    nodes: BTreeMap<usize, ComputationNode>,
    inputs: Vec<usize>,
}

impl OldBuilder {
    fn new() -> Self {
        OldBuilder {
            nodes: BTreeMap::new(),
            inputs: vec![],
        }
    }

    fn push(&mut self, operator: Operator, inputs: Vec<usize>, dims: Vec<usize>) -> usize {
        let idx = self.nodes.len();
        if matches!(operator, Operator::Input(_)) {
            self.inputs.push(idx);
        }
        self.nodes
            .insert(idx, ComputationNode::new(idx, operator, inputs, dims));
        idx
    }

    fn input(&mut self, dims: &[usize]) -> usize {
        self.push(Operator::Input(old_ops::Input), vec![], dims.to_vec())
    }

    fn constant(&mut self, data: &[i32], dims: &[usize]) -> usize {
        let t = OldTensor::new(Some(data), dims).unwrap();
        self.push(
            Operator::Constant(old_ops::Constant(t)),
            vec![],
            dims.to_vec(),
        )
    }

    fn build(self, outputs: Vec<usize>) -> OldModel {
        let mut model = OldModel {
            graph: ComputationGraph {
                nodes: self.nodes,
                inputs: self.inputs,
                outputs,
                original_input_dims: HashMap::new(),
                original_output_dims: HashMap::new(),
            },
            scale: SCALE as i32,
        };
        model.annotate_clamp_widths();
        model
    }
}

/// Convert an old model's graph to the new IR (also exercises the mapping a
/// real migration would use).
fn convert(old: &OldModel) -> Graph {
    let nodes = old
        .graph
        .nodes
        .values()
        .map(|n| Node {
            op: convert_op(&n.operator, &n.output_dims),
            inputs: n.inputs.iter().map(|&i| NodeId(i as u32)).collect(),
            output_dims: n.output_dims.clone(),
        })
        .collect();
    Graph {
        nodes,
        inputs: old.graph.inputs.iter().map(|&i| NodeId(i as u32)).collect(),
        outputs: old
            .graph
            .outputs
            .iter()
            .map(|&i| NodeId(i as u32))
            .collect(),
    }
}

fn convert_op(operator: &Operator, output_dims: &[usize]) -> Op {
    match operator {
        Operator::Input(_) => Op::Input(op::Input),
        Operator::Constant(c) => Op::Constant(op::Constant(
            Tensor::new(c.0.data().to_vec(), c.0.dims()).unwrap(),
        )),
        Operator::Identity(_) => Op::Identity(op::Identity),
        Operator::Add(_) => Op::Add(op::Add),
        Operator::Sub(_) => Op::Sub(op::Sub),
        Operator::Neg(_) => Op::Neg(op::Neg),
        Operator::Mul(m) => Op::Mul(op::Mul {
            rebase: m.scale != 0,
        }),
        Operator::Square(s) => Op::Square(op::Square {
            rebase: s.scale != 0,
        }),
        Operator::Cube(c) => Op::Cube(op::Cube {
            rebase: c.scale != 0,
        }),
        Operator::Div(_) => Op::Div(op::Div),
        Operator::ScalarConstDiv(d) => {
            Op::ScalarConstDiv(op::ScalarConstDiv { divisor: d.divisor })
        }
        Operator::Einsum(e) => Op::Einsum(op::Einsum {
            equation: e.equation.parse().unwrap(),
        }),
        Operator::Sum(s) => Op::Sum(op::Sum {
            axes: s.axes.clone(),
        }),
        Operator::MeanOfSquares(m) => Op::MeanOfSquares(op::MeanOfSquares {
            axes: m.axes.clone(),
            count: m.count,
            padded_count: m.padded_count,
        }),
        Operator::SoftmaxLastAxis(_) => {
            Op::SoftmaxLastAxis(op::SoftmaxLastAxis { active_len: None })
        }
        Operator::Rsqrt(_) => Op::Rsqrt(op::Rsqrt),
        Operator::Tanh(_) => Op::Tanh(op::Tanh),
        Operator::Sigmoid(_) => Op::Sigmoid(op::Sigmoid),
        Operator::Erf(_) => Op::Erf(op::Erf),
        Operator::Sin(_) => Op::Sin(op::Sin),
        Operator::Cos(_) => Op::Cos(op::Cos),
        Operator::ReLU(_) => Op::ReLU(op::ReLU),
        Operator::Clamp(c) => Op::Clamp(op::Clamp {
            bound_log: c.bound_log as u32,
        }),
        Operator::Iff(_) => Op::Iff(op::Iff),
        Operator::And(_) => Op::And(op::And),
        Operator::IsNan(_) => Op::IsNan(op::IsNan),
        Operator::GatherSmall(g) => Op::GatherSmall(op::GatherSmall {
            axis: g.axis,
            dict_len: g.dict_len,
        }),
        Operator::GatherLarge(g) => Op::GatherLarge(op::GatherLarge {
            axis: g.axis,
            dict_len: g.dict_len,
        }),
        Operator::Slice(s) => Op::Slice(op::Slice {
            axis: s.axis,
            start: s.start,
            end: s.end,
        }),
        Operator::Concat(c) => Op::Concat(op::Concat {
            axis: if c.axis < 0 {
                (c.axis + output_dims.len() as isize) as usize
            } else {
                c.axis as usize
            },
        }),
        Operator::Reshape(r) => Op::Reshape(op::Reshape {
            shape: r.shape.clone(),
        }),
        Operator::Broadcast(b) => Op::Broadcast(op::Broadcast {
            shape: b.shape.clone(),
        }),
        Operator::MoveAxis(m) => Op::MoveAxis(op::MoveAxis {
            source: m.source,
            destination: m.destination,
        }),
    }
}

fn compile_new(old: &OldModel) -> Model {
    Model::compile(
        convert(old),
        CompileOptions {
            scale: Scale::new(SCALE).unwrap(),
            pad_pow2: false,
            version: SemVersion::V1,
        },
    )
    .unwrap()
}

fn random_tensor(rng: &mut StdRng, dims: &[usize], bound: i32) -> Vec<i32> {
    (0..dims.iter().product::<usize>())
        .map(|_| rng.gen_range(-bound..bound))
        .collect()
}

/// Run both crates on the same graph+inputs; return the traces.
fn run_both(
    old: &OldModel,
    new: &Model,
    inputs: &[(Vec<i32>, Vec<usize>)],
) -> (atlas_onnx_tracer::model::trace::Trace, atlas_trace::Trace) {
    let old_inputs: Vec<OldTensor<i32>> = inputs
        .iter()
        .map(|(d, dims)| OldTensor::new(Some(d), dims).unwrap())
        .collect();
    let new_inputs: Vec<Tensor<i32>> = inputs
        .iter()
        .map(|(d, dims)| Tensor::new(d.clone(), dims).unwrap())
        .collect();
    (old.trace(&old_inputs), new.trace(&new_inputs).unwrap())
}

fn assert_nodes_equal(old: &atlas_onnx_tracer::model::trace::Trace, new: &atlas_trace::Trace) {
    for (&idx, old_out) in &old.node_outputs {
        let new_out = new.output(NodeId(idx as u32));
        assert_eq!(old_out.dims(), new_out.dims(), "node {idx} dims");
        assert_eq!(old_out.data(), new_out.data(), "node {idx} values");
    }
}

fn assert_fused_equal(old: &atlas_onnx_tracer::model::trace::Trace, new: &atlas_trace::Trace) {
    for (&idx, old_f) in &old.fused_intermediates {
        let new_f = new
            .fused(NodeId(idx as u32))
            .unwrap_or_else(|| panic!("node {idx} missing fused intermediates"));
        assert_eq!(old_f.quotient.data(), new_f.quotient.data(), "node {idx} q");
        assert_eq!(
            old_f.remainder.data(),
            new_f.remainder.data(),
            "node {idx} r"
        );
    }
}

/// Clamp widths must agree exactly (soundness-critical analysis).
fn assert_widths_equal(old: &OldModel, new: &Model) {
    for (id, _node, meta) in new.nodes() {
        let old_bits = old.graph.nodes[&id.index()].sat_clamp_bits;
        match meta.sat_clamp_bits {
            Some(w) => assert_eq!(w.bits(), old_bits, "node {id} clamp width"),
            None => assert_eq!(old_bits, 64, "node {id} should be unclamped (default 64)"),
        }
    }
}

#[test]
fn exact_class_elementwise_and_fused() {
    let mut b = OldBuilder::new();
    let dims = vec![4, 8];
    let x = b.input(&dims);
    let y = b.input(&dims);

    let mut rng = StdRng::seed_from_u64(0xA71A5);
    let divisor_data: Vec<i32> = (0..32).map(|_| rng.gen_range(1..5000) * 3 - 7000).collect();
    let divisor_data: Vec<i32> = divisor_data
        .into_iter()
        .map(|v| if v == 0 { 7 } else { v })
        .collect();
    let mask_data: Vec<i32> = (0..32).map(|_| rng.r#gen::<bool>() as i32).collect();
    let bool_data: Vec<i32> = (0..32).map(|_| rng.r#gen::<bool>() as i32).collect();

    let divisor = b.constant(&divisor_data, &dims);
    let mask = b.constant(&mask_data, &dims);
    let boolean = b.constant(&bool_data, &dims);

    let add = b.push(Operator::Add(old_ops::Add), vec![x, y], dims.clone());
    let sub = b.push(Operator::Sub(old_ops::Sub), vec![add, y], dims.clone());
    let neg = b.push(Operator::Neg(old_ops::Neg), vec![sub], dims.clone());
    let relu = b.push(Operator::ReLU(old_ops::ReLU), vec![neg], dims.clone());
    let mul = b.push(
        Operator::Mul(old_ops::Mul {
            scale: SCALE as i32,
        }),
        vec![relu, y],
        dims.clone(),
    );
    let square = b.push(
        Operator::Square(old_ops::Square {
            scale: SCALE as i32,
        }),
        vec![mul],
        dims.clone(),
    );
    let cube = b.push(
        Operator::Cube(old_ops::Cube {
            scale: SCALE as i32,
        }),
        vec![square],
        dims.clone(),
    );
    let div = b.push(
        Operator::Div(old_ops::Div),
        vec![cube, divisor],
        dims.clone(),
    );
    let scd = b.push(
        Operator::ScalarConstDiv(old_ops::ScalarConstDiv { divisor: -3 }),
        vec![div],
        dims.clone(),
    );
    let clamp = b.push(
        Operator::Clamp(old_ops::Clamp { bound_log: 16 }),
        vec![scd],
        dims.clone(),
    );
    let iff = b.push(
        Operator::Iff(old_ops::Iff),
        vec![mask, clamp, x],
        dims.clone(),
    );
    let and = b.push(
        Operator::And(old_ops::And),
        vec![mask, boolean],
        dims.clone(),
    );
    let isnan = b.push(
        Operator::IsNan(old_ops::IsNan {
            out_dims: dims.clone(),
        }),
        vec![iff],
        dims.clone(),
    );
    let rsqrt = b.push(
        Operator::Rsqrt(old_ops::Rsqrt {
            scale: SCALE as i32,
        }),
        vec![iff],
        dims.clone(),
    );
    let sum = b.push(
        Operator::Sum(old_ops::Sum { axes: vec![1] }),
        vec![rsqrt],
        vec![4, 1],
    );
    let identity = b.push(Operator::Identity(old_ops::Identity), vec![sum], vec![4, 1]);
    let _ = (and, isnan);

    let old = b.build(vec![identity]);
    let new = compile_new(&old);
    assert_widths_equal(&old, &new);

    let inputs = vec![
        (random_tensor(&mut rng, &[4, 8], 1 << 16), vec![4, 8]),
        (random_tensor(&mut rng, &[4, 8], 1 << 16), vec![4, 8]),
    ];
    let (old_trace, new_trace) = run_both(&old, &new, &inputs);
    assert_nodes_equal(&old_trace, &new_trace);
    assert_fused_equal(&old_trace, &new_trace);
}

#[test]
fn exact_class_structural_and_einsum() {
    let mut rng = StdRng::seed_from_u64(0x5EED);
    let mut b = OldBuilder::new();
    let x = b.input(&[4, 8]);

    let w_data = random_tensor(&mut rng, &[8, 4], 1 << 14);
    let w = b.constant(&w_data, &[8, 4]);
    let matmul = b.push(
        Operator::Einsum(old_ops::Einsum {
            equation: "mk,kn->mn".into(),
            scale: SCALE as i32,
        }),
        vec![x, w],
        vec![4, 4],
    );
    let mos = b.push(
        Operator::MeanOfSquares(old_ops::MeanOfSquares {
            axes: vec![1],
            scale: SCALE as i32,
            count: 4,
            padded_count: 4,
        }),
        vec![matmul],
        vec![4, 1],
    );
    let bcast = b.push(
        Operator::Broadcast(old_ops::Broadcast { shape: vec![4, 4] }),
        vec![mos],
        vec![4, 4],
    );
    let reshape = b.push(
        Operator::Reshape(old_ops::Reshape { shape: vec![2, 8] }),
        vec![bcast],
        vec![2, 8],
    );
    let moveaxis = b.push(
        Operator::MoveAxis(old_ops::MoveAxis {
            source: 0,
            destination: 1,
        }),
        vec![reshape],
        vec![8, 2],
    );
    let slice = b.push(
        Operator::Slice(old_ops::Slice {
            axis: 0,
            start: 2,
            end: 6,
        }),
        vec![moveaxis],
        vec![4, 2],
    );
    let concat = b.push(
        Operator::Concat(old_ops::Concat { axis: 0 }),
        vec![slice, slice],
        vec![8, 2],
    );

    // Gather from an embedding dict.
    let dict_data = random_tensor(&mut rng, &[16, 4], 1 << 14);
    let dict = b.constant(&dict_data, &[16, 4]);
    let idx_data: Vec<i32> = (0..6).map(|_| rng.gen_range(0..16)).collect();
    let indices = b.constant(&idx_data, &[6]);
    let gather = b.push(
        Operator::GatherSmall(old_ops::GatherSmall {
            axis: 0,
            dict_len: 16,
        }),
        vec![dict, indices],
        vec![6, 4],
    );

    // A dynamic×dynamic einsum (no constant operand) for the width analysis.
    let q = b.push(
        Operator::Einsum(old_ops::Einsum {
            equation: "mk,nk->mn".into(),
            scale: SCALE as i32,
        }),
        vec![gather, gather],
        vec![6, 6],
    );

    let old = b.build(vec![concat, q]);
    let new = compile_new(&old);
    assert_widths_equal(&old, &new);

    let inputs = vec![(random_tensor(&mut rng, &[4, 8], 1 << 15), vec![4, 8])];
    let (old_trace, new_trace) = run_both(&old, &new, &inputs);
    assert_nodes_equal(&old_trace, &new_trace);
    assert_fused_equal(&old_trace, &new_trace);
}

#[test]
fn table_class_within_one_ulp() {
    let mut rng = StdRng::seed_from_u64(0x7AB7E);
    let mut b = OldBuilder::new();
    let dims = vec![64];
    let x = b.input(&dims);
    let s = SCALE as i32;
    let tanh = b.push(
        Operator::Tanh(old_ops::Tanh { scale: s }),
        vec![x],
        dims.clone(),
    );
    let sigmoid = b.push(
        Operator::Sigmoid(old_ops::Sigmoid { scale: s }),
        vec![x],
        dims.clone(),
    );
    let erf = b.push(
        Operator::Erf(old_ops::Erf { scale: s }),
        vec![x],
        dims.clone(),
    );
    let sin = b.push(
        Operator::Sin(old_ops::Sin { scale: s }),
        vec![x],
        dims.clone(),
    );
    let cos = b.push(
        Operator::Cos(old_ops::Cos { scale: s }),
        vec![x],
        dims.clone(),
    );
    let old = b.build(vec![tanh, sigmoid, erf, sin, cos]);
    let new = compile_new(&old);

    let inputs = vec![(random_tensor(&mut rng, &[64], 1 << 18), vec![64])];
    let (old_trace, new_trace) = run_both(&old, &new, &inputs);

    let trig_tol = 1i32 << atlas_trace::semantics::fixed::trig_downscale_bits(SCALE as u32);
    for (idx, tol) in [
        (tanh, 1),
        (sigmoid, 1),
        (erf, 1),
        (sin, trig_tol),
        (cos, trig_tol),
    ] {
        let old_out = &old_trace.node_outputs[&idx];
        let new_out = new_trace.output(NodeId(idx as u32));
        for (i, (&a, &b)) in old_out.data().iter().zip(new_out.data()).enumerate() {
            assert!(
                (a - b).abs() <= tol,
                "node {idx} element {i}: old {a}, new {b} (tol {tol})"
            );
        }
    }
}

#[test]
fn softmax_matches_within_tolerance() {
    let mut rng = StdRng::seed_from_u64(0x50F7);
    let mut b = OldBuilder::new();
    let dims = vec![4, 8];
    let x = b.input(&dims);
    let softmax = b.push(
        Operator::SoftmaxLastAxis(old_ops::SoftmaxLastAxis {
            scale: SCALE as i32,
        }),
        vec![x],
        dims.clone(),
    );
    let old = b.build(vec![softmax]);
    let new = compile_new(&old);

    let inputs = vec![(random_tensor(&mut rng, &[4, 8], 1 << 15), vec![4, 8])];
    let (old_trace, new_trace) = run_both(&old, &new, &inputs);

    let old_out = &old_trace.node_outputs[&softmax];
    let new_out = new_trace.output(NodeId(softmax as u32));
    for (i, (&a, &bv)) in old_out.data().iter().zip(new_out.data()).enumerate() {
        assert!(
            (a - bv).abs() <= 8,
            "softmax element {i}: old {a}, new {bv}"
        );
    }
    // The new crate captures the witness in the trace.
    let w = new_trace.softmax(NodeId(softmax as u32)).unwrap();
    assert_eq!(w.active_len, 8);
}
