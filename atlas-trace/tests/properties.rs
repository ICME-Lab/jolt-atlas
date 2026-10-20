//! Structural property tests over randomly generated graphs:
//!
//! 1. **Padding invariance** — `crop(trace(pad(g), x)) == trace(g, x)` on
//!    every live lane of every node. This single property subsumes the
//!    softmax padding fix and guards every future op.
//! 2. **Clamp-width soundness** — every annotated width really contains the
//!    pre-clamp value (via the captured fused quotients).
//! 3. **Determinism** — identical runs produce byte-identical serialized
//!    traces.

use proptest::prelude::*;

use atlas_trace::{
    CompileOptions, Graph, Model, Node, NodeId, Op, Scale, SemVersion, Tensor, op,
};

const SCALE: u8 = 14;

/// A small random graph: a chain of unary/binary ops over two inputs with
/// deliberately non-power-of-two dims, ending in one output.
#[derive(Debug, Clone)]
struct GraphSpec {
    dims: Vec<usize>,
    ops: Vec<u8>,
    input_a: Vec<i32>,
    input_b: Vec<i32>,
}

fn graph_spec() -> impl Strategy<Value = GraphSpec> {
    // Non-pow2 dims make padding do real work.
    let dims = prop_oneof![
        Just(vec![3usize, 5]),
        Just(vec![2usize, 7]),
        Just(vec![6usize]),
        Just(vec![2usize, 3, 3]),
    ];
    (dims, proptest::collection::vec(0u8..12, 1..6)).prop_flat_map(|(dims, ops)| {
        let len = dims.iter().product::<usize>();
        (
            Just(dims),
            Just(ops),
            proptest::collection::vec(-(1 << 15)..(1 << 15), len),
            proptest::collection::vec(-(1 << 15)..(1 << 15), len),
        )
            .prop_map(|(dims, ops, input_a, input_b)| GraphSpec {
                dims,
                ops,
                input_a,
                input_b,
            })
    })
}

/// Materialize the spec as a graph. Each op code appends one node reading
/// the previous tip (and input b where binary).
fn build_graph(spec: &GraphSpec) -> Graph {
    let dims = spec.dims.clone();
    let mut nodes = vec![
        Node {
            op: Op::Input(op::Input),
            inputs: vec![],
            output_dims: dims.clone(),
        },
        Node {
            op: Op::Input(op::Input),
            inputs: vec![],
            output_dims: dims.clone(),
        },
    ];
    let b = NodeId(1);
    let mut tip = NodeId(0);
    for &code in &spec.ops {
        let id = NodeId(nodes.len() as u32);
        let (op, inputs, out_dims): (Op, Vec<NodeId>, Vec<usize>) = match code {
            0 => (Op::Add(op::Add), vec![tip, b], dims.clone()),
            1 => (Op::Sub(op::Sub), vec![tip, b], dims.clone()),
            2 => (Op::Neg(op::Neg), vec![tip], dims.clone()),
            3 => (Op::ReLU(op::ReLU), vec![tip], dims.clone()),
            4 => (
                Op::Mul(op::Mul { rebase: true }),
                vec![tip, b],
                dims.clone(),
            ),
            5 => (
                Op::Square(op::Square { rebase: true }),
                vec![tip],
                dims.clone(),
            ),
            6 => (
                Op::SoftmaxLastAxis(op::SoftmaxLastAxis { active_len: None }),
                vec![tip],
                dims.clone(),
            ),
            7 => (Op::Tanh(op::Tanh), vec![tip], dims.clone()),
            8 => (Op::Erf(op::Erf), vec![tip], dims.clone()),
            9 => (Op::Sin(op::Sin), vec![tip], dims.clone()),
            10 => {
                // Sum over the last axis.
                let axis = dims.len() - 1;
                let mut od = dims.clone();
                od[axis] = 1;
                (Op::Sum(op::Sum { axes: vec![axis] }), vec![tip], od)
            }
            _ => {
                let axis = dims.len() - 1;
                let count = dims[axis];
                let mut od = dims.clone();
                od[axis] = 1;
                (
                    Op::MeanOfSquares(op::MeanOfSquares {
                        axes: vec![axis],
                        count,
                        padded_count: count.next_power_of_two(),
                    }),
                    vec![tip],
                    od,
                )
            }
        };
        // Reductions change dims; chain only while dims match for binaries.
        nodes.push(Node {
            op,
            inputs,
            output_dims: out_dims.clone(),
        });
        if out_dims == dims {
            tip = id;
        } else {
            // Stop chaining after a reduction: it becomes the output.
            tip = id;
            break;
        }
    }
    Graph {
        inputs: vec![NodeId(0), NodeId(1)],
        outputs: vec![tip],
        nodes,
    }
}

fn compile(graph: Graph, pad: bool) -> Model {
    Model::compile(
        graph,
        CompileOptions {
            scale: Scale::new(SCALE).unwrap(),
            pad_pow2: pad,
            version: SemVersion::V1,
        },
    )
    .unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn padding_invariance(spec in graph_spec()) {
        let graph = build_graph(&spec);
        let inputs = vec![
            Tensor::new(spec.input_a.clone(), &spec.dims).unwrap(),
            Tensor::new(spec.input_b.clone(), &spec.dims).unwrap(),
        ];

        let unpadded = compile(graph.clone(), false);
        let padded = compile(graph.clone(), true);
        let t_un = unpadded.trace(&inputs).unwrap();
        let t_pad = padded.trace(&inputs).unwrap();

        // Every node's padded output, cropped to the unpadded dims, must
        // equal the unpadded output.
        for (id, node, _) in unpadded.nodes() {
            let logical = &node.output_dims;
            let cropped = t_pad.output(id).crop_to(logical).unwrap();
            prop_assert_eq!(
                cropped.data(),
                t_un.output(id).data(),
                "node {} ({}) diverges under padding",
                id,
                node.op
            );
        }
    }

    #[test]
    fn clamp_width_soundness(spec in graph_spec()) {
        let graph = build_graph(&spec);
        let inputs = vec![
            Tensor::new(spec.input_a.clone(), &spec.dims).unwrap(),
            Tensor::new(spec.input_b.clone(), &spec.dims).unwrap(),
        ];
        let model = compile(graph, true);
        let trace = model.trace(&inputs).unwrap();

        for (id, node, meta) in model.nodes() {
            let Some(width) = meta.sat_clamp_bits else { continue };
            // For fused ops the pre-clamp value is the captured quotient.
            if let Some(f) = trace.fused(id) {
                let bound = 1i128 << (width.bits() - 1);
                for &q in f.quotient.data() {
                    prop_assert!(
                        (q as i128) < bound && (q as i128) >= -bound,
                        "node {} ({}): quotient {} escapes {}-bit width",
                        id, node.op, q, width.bits()
                    );
                }
            }
        }
    }

    #[test]
    fn determinism(spec in graph_spec()) {
        let graph = build_graph(&spec);
        let inputs = vec![
            Tensor::new(spec.input_a.clone(), &spec.dims).unwrap(),
            Tensor::new(spec.input_b.clone(), &spec.dims).unwrap(),
        ];
        let model = compile(graph, true);
        let a = serde_json::to_vec(&model.trace(&inputs).unwrap()).unwrap();
        let b = serde_json::to_vec(&model.trace(&inputs).unwrap()).unwrap();
        prop_assert_eq!(a, b);
    }
}
