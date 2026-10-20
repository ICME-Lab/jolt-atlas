//! Transitional facade over [`atlas_trace`], shaped like the retired
//! `atlas_onnx_tracer` API.
//!
//! The proof layer was written against the old tracer's types
//! (`ComputationNode` with a usize `idx`, a `BTreeMap`-backed graph, a
//! `Trace` keyed by node index). This module rebuilds that surface as a thin
//! view over an [`atlas_trace::Model`], so the ~80 proof-layer files migrate
//! with import changes instead of a big-bang rewrite. It is a stepping
//! stone: call sites should move to the `atlas_trace` types directly over
//! time, and this module shrinks as they do.
//!
//! Semantic upgrades that land through this facade:
//! - traces always carry the fused-rescale intermediates and the full
//!   softmax witness (no prover-side re-execution);
//! - lookup tables come from [`atlas_trace::semantics`] — deterministic
//!   integer generation, never platform floats;
//! - einsum equations are parsed [`atlas_trace::op::EinsumEq`] values.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub use atlas_trace::{NodeId, Scale, SemVersion};

/// Node types, old-shaped.
pub mod node {
    use super::ops::Operator;
    use serde::{Deserialize, Serialize};

    /// A node viewed the way the proof layer has always consumed it: with
    /// its index, operator, usize operand indices, dims, and clamp width.
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    pub struct ComputationNode {
        /// Dense node index (topological order).
        pub idx: usize,
        /// The operator (constants are `Arc`'d, so cloning is cheap).
        pub operator: Operator,
        /// Operand node indices.
        pub inputs: Vec<usize>,
        /// Output dims (padded when the model was compiled with padding).
        pub output_dims: Vec<usize>,
        /// Saturating-clamp lookup width in bits (64 when unclamped).
        pub sat_clamp_bits: usize,
        /// The model's global fixed-point scale (log2). Carried per node so
        /// operator-level helpers (`rebase_bits`, table sizing) don't need a
        /// model reference.
        pub scale: i32,
    }

    impl ComputationNode {
        /// Construct a node view at the default model scale.
        pub fn new(
            idx: usize,
            operator: Operator,
            inputs: Vec<usize>,
            output_dims: Vec<usize>,
        ) -> Self {
            Self {
                idx,
                operator,
                inputs,
                output_dims,
                sat_clamp_bits: 64,
                scale: common::consts::MODEL_SCALE as i32,
            }
        }

        /// Total output elements at the declared dims.
        pub fn num_output_elements(&self) -> usize {
            self.output_dims.iter().product()
        }

        /// Total output elements after per-dimension power-of-two padding.
        pub fn pow2_padded_num_output_elements(&self) -> usize {
            self.output_dims
                .iter()
                .map(|d| d.next_power_of_two())
                .product()
        }

        /// Whether the node produces a single element.
        pub fn is_scalar(&self) -> bool {
            self.num_output_elements() == 1
        }
    }
}

/// Operator re-exports, old-shaped.
pub mod ops {
    pub use atlas_trace::op::*;
    pub use atlas_trace::trace::FusedIntermediates;

    /// The operator enum (the atlas-trace one, under its old name).
    pub type Operator = atlas_trace::Op;
}

/// Tensor re-export (old path shape).
pub mod tensor {
    pub use atlas_trace::tensor::Element;
    pub use atlas_trace::Tensor;
}

/// Model, graph, trace, and analysis views, old-shaped.
pub mod model {
    use super::*;
    use atlas_trace::{NodeAux, Tensor};

    pub use trace::{LayerData, ModelExecutionIO, Trace};

    /// Runtime arguments for loading a model (old surface).
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct RunArgs {
        /// Symbolic dim bindings (`batch_size`, `sequence_length`, …).
        pub variables: std::collections::HashMap<String, usize>,
        /// Global fixed-point scale (log2).
        pub scale: i32,
        /// Pad all dims to powers of two.
        pub pad_to_power_of_2: bool,
    }

    impl Default for RunArgs {
        fn default() -> Self {
            let mut variables = std::collections::HashMap::new();
            variables.insert("batch_size".to_string(), 1);
            RunArgs {
                variables,
                scale: common::consts::MODEL_SCALE as i32,
                pad_to_power_of_2: true,
            }
        }
    }

    impl RunArgs {
        /// RunArgs with the given symbolic dim bindings.
        pub fn new<K: Into<String>>(bindings: impl IntoIterator<Item = (K, usize)>) -> Self {
            let mut args = RunArgs::default();
            args.variables
                .extend(bindings.into_iter().map(|(k, v)| (k.into(), v)));
            args
        }

        /// Toggle power-of-two padding.
        pub fn with_padding(mut self, pad: bool) -> Self {
            self.pad_to_power_of_2 = pad;
            self
        }

        /// Set the fixed-point scale (log2).
        pub fn set_scale(mut self, scale: i32) -> Self {
            self.scale = scale;
            self
        }
    }

    /// The graph view: a dense index-keyed map of node views.
    #[derive(Clone, Debug, PartialEq, Default)]
    pub struct ComputationGraph {
        /// Node views by index (dense, topologically ordered).
        pub nodes: BTreeMap<usize, node::ComputationNode>,
        /// Input node indices.
        pub inputs: Vec<usize>,
        /// Output node indices.
        pub outputs: Vec<usize>,
    }

    impl ComputationGraph {
        /// A node by index.
        pub fn get_node(&self, idx: usize) -> Option<&node::ComputationNode> {
            self.nodes.get(&idx)
        }

        /// The operand node views of `node`.
        pub fn get_input_nodes(&self, n: &node::ComputationNode) -> Vec<&node::ComputationNode> {
            n.inputs
                .iter()
                .map(|idx| self.nodes.get(idx).expect("operand present"))
                .collect()
        }
    }

    /// The model: an [`atlas_trace::Model`] plus its old-shaped graph view.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    #[serde(from = "ModelSerde", into = "ModelSerde")]
    pub struct Model {
        /// Old-shaped graph view (derived from `inner`).
        pub graph: ComputationGraph,
        /// Global fixed-point scale (log2).
        pub scale: i32,
        /// `None` only for hand-assembled view-only test models
        /// ([`Model::default`] + manual graph edits), which cannot trace.
        inner: Option<atlas_trace::Model>,
    }

    impl Default for Model {
        fn default() -> Self {
            Model {
                graph: ComputationGraph::default(),
                scale: common::consts::MODEL_SCALE as i32,
                inner: None,
            }
        }
    }

    /// Serialized form: only the compiled model (the view is derived).
    #[derive(Serialize, Deserialize)]
    struct ModelSerde {
        inner: Option<atlas_trace::Model>,
    }

    impl From<ModelSerde> for Model {
        fn from(s: ModelSerde) -> Self {
            match s.inner {
                Some(inner) => Model::from_atlas(inner),
                None => Model::default(),
            }
        }
    }

    impl From<Model> for ModelSerde {
        fn from(m: Model) -> Self {
            ModelSerde { inner: m.inner }
        }
    }

    impl PartialEq for Model {
        fn eq(&self, other: &Self) -> bool {
            self.graph == other.graph && self.scale == other.scale
        }
    }
    impl Eq for Model {}

    impl Model {
        /// Build the old-shaped view over a compiled model.
        pub fn from_atlas(inner: atlas_trace::Model) -> Self {
            let scale = inner.scale().log2() as i32;
            let nodes = inner
                .nodes()
                .map(|(id, n, meta)| {
                    (
                        id.index(),
                        node::ComputationNode {
                            idx: id.index(),
                            operator: n.op.clone(),
                            inputs: n.inputs.iter().map(|i| i.index()).collect(),
                            output_dims: n.output_dims.clone(),
                            sat_clamp_bits: meta.sat_clamp_bits.map(|w| w.bits()).unwrap_or(64),
                            scale,
                        },
                    )
                })
                .collect();
            let graph = ComputationGraph {
                nodes,
                inputs: inner.inputs().iter().map(|i| i.index()).collect(),
                outputs: inner.outputs().iter().map(|i| i.index()).collect(),
            };
            Model {
                graph,
                scale,
                inner: Some(inner),
            }
        }

        /// Load and compile an ONNX model (the old `Model::load` surface over
        /// `atlas_onnx_import`).
        pub fn load(path: &str, run_args: &RunArgs) -> Model {
            let imported = atlas_onnx_import::load(
                std::path::Path::new(path),
                &atlas_onnx_import::LoadOptions {
                    bindings: run_args.variables.clone(),
                    scale: Scale::new(run_args.scale as u8).expect("scale in range"),
                    pad_pow2: run_args.pad_to_power_of_2,
                },
            )
            .unwrap_or_else(|e| panic!("Model::load({path}): {e}"));
            Model::from_atlas(imported.model)
        }

        /// One line per node (the old `pretty_print` surface).
        pub fn pretty_print(&self) -> String {
            use std::fmt::Write;
            let mut out = String::new();
            for node in self.graph.nodes.values() {
                let _ = writeln!(
                    out,
                    "{:>5}  {:<40} inputs {:?} dims {:?} clamp {}",
                    node.idx,
                    node.operator.to_string(),
                    node.inputs,
                    node.output_dims,
                    node.sat_clamp_bits,
                );
            }
            out
        }

        /// The underlying compiled model.
        pub fn inner(&self) -> &atlas_trace::Model {
            self.inner
                .as_ref()
                .expect("view-only model (Model::default) cannot be executed")
        }

        /// Execute and capture the full trace (panics on malformed inputs,
        /// matching the old API; use `inner().trace()` for typed errors).
        ///
        /// A view-only model ([`Model::default`]) yields an empty trace that
        /// tests populate by hand.
        pub fn trace(&self, inputs: &[Tensor<i32>]) -> Trace {
            let Some(inner) = &self.inner else {
                assert!(inputs.is_empty(), "view-only model cannot be executed");
                return Trace::new(BTreeMap::new());
            };
            let t = inner.trace(inputs).expect("model execution failed");
            Trace::from_atlas(t)
        }

        /// The node views, by index.
        pub fn nodes(&self) -> &BTreeMap<usize, node::ComputationNode> {
            &self.graph.nodes
        }

        /// Input node indices.
        pub fn inputs(&self) -> &[usize] {
            &self.graph.inputs
        }

        /// Output node indices.
        pub fn outputs(&self) -> &[usize] {
            &self.graph.outputs
        }

        /// Operand node views of `node`.
        pub fn get_input_nodes(&self, n: &node::ComputationNode) -> Vec<&node::ComputationNode> {
            self.graph.get_input_nodes(n)
        }

        /// Max committed-poly variable count over all nodes (SRS sizing).
        pub fn max_num_vars(&self) -> usize {
            self.graph
                .nodes
                .values()
                .map(|node| self.node_committed_poly_num_vars(node))
                .max()
                .unwrap_or(1)
        }

        /// Number of variables of the largest committed polynomial this node
        /// contributes (1 when it commits nothing). Prover sizing logic that
        /// used to live in the tracer.
        pub fn node_committed_poly_num_vars(&self, node: &node::ComputationNode) -> usize {
            use common::consts::LOG_K_CHUNK;
            let log_2 = |x: usize| {
                assert_ne!(x, 0);
                if x.is_power_of_two() {
                    x.trailing_zeros() as usize
                } else {
                    (usize::BITS - x.leading_zeros()) as usize
                }
            };
            use ops::Operator;
            match &node.operator {
                Operator::Tanh(_)
                | Operator::Cos(_)
                | Operator::Div(_)
                | Operator::Erf(_)
                | Operator::Clamp(_)
                | Operator::ReLU(_)
                | Operator::Rsqrt(_)
                | Operator::Sigmoid(_)
                | Operator::Sin(_)
                | Operator::Add(_)
                | Operator::Sub(_)
                | Operator::Sum(_)
                | Operator::Einsum(_)
                | Operator::MeanOfSquares(_)
                | Operator::SoftmaxLastAxis(_) => {
                    LOG_K_CHUNK + log_2(node.pow2_padded_num_output_elements())
                }
                Operator::Mul(op) if op.rebase => {
                    LOG_K_CHUNK + log_2(node.pow2_padded_num_output_elements())
                }
                Operator::Square(op) if op.rebase => {
                    LOG_K_CHUNK + log_2(node.pow2_padded_num_output_elements())
                }
                Operator::Cube(op) if op.rebase => {
                    LOG_K_CHUNK + log_2(node.pow2_padded_num_output_elements())
                }
                Operator::ScalarConstDiv(_) => log_2(node.pow2_padded_num_output_elements()),
                Operator::GatherSmall(_) => {
                    let input_nodes = self.get_input_nodes(node);
                    let num_words = input_nodes[0].output_dims[0];
                    let num_indices = input_nodes[1].pow2_padded_num_output_elements();
                    log_2(num_words) + log_2(num_indices)
                }
                Operator::GatherLarge(_) => {
                    let input_nodes = self.get_input_nodes(node);
                    let num_words = input_nodes[0].output_dims[0].next_power_of_two();
                    let num_indices = input_nodes[1].pow2_padded_num_output_elements();
                    log_2(num_words) + log_2(num_indices)
                }
                _ => 1,
            }
        }
    }

    impl std::ops::Index<usize> for Model {
        type Output = node::ComputationNode;

        fn index(&self, index: usize) -> &Self::Output {
            &self.graph.nodes[&index]
        }
    }

    /// Clamp-width analysis surface, old-shaped.
    pub mod clamp_width {
        use super::*;
        use atlas_trace::op::BoundCtx;

        /// Widest saturating-clamp lookup.
        pub const CLAMP_WIDTH_MAX: usize = 64;
        /// Supported clamp lookup widths.
        pub const CLAMP_WIDTHS: [usize; 4] = [40, 48, 56, 64];

        /// Sound magnitude bound on a node's pre-clamp value, or `None` for
        /// unclamped nodes. Delegates to the op's own `clamp_bound`.
        pub fn clamp_value_bound(
            node: &node::ComputationNode,
            nodes: &BTreeMap<usize, node::ComputationNode>,
        ) -> Option<u128> {
            let input_dims: Vec<&[usize]> = node
                .inputs
                .iter()
                .map(|idx| nodes[idx].output_dims.as_slice())
                .collect();
            let input_consts: Vec<Option<&Tensor<i32>>> = node
                .inputs
                .iter()
                .map(|idx| match &nodes[idx].operator {
                    ops::Operator::Constant(c) => Some(c.0.as_ref()),
                    _ => None,
                })
                .collect();
            let ctx = BoundCtx {
                scale: Scale::new(node.scale as u8).expect("scale in range"),
                input_dims,
                input_consts,
            };
            node.operator.spec().clamp_bound(&ctx)
        }
    }

    /// Programmatic model construction for tests (the old `ModelBuilder`
    /// surface over the new compiler).
    pub mod test {
        use super::*;
        use atlas_trace::{op, CompileOptions, Graph, Node, NodeId, Op, SemVersion};

        /// A connection to a node's output.
        pub type Wire = usize;

        const DEFAULT_SCALE: u32 = common::consts::MODEL_SCALE as u32;

        /// Builds a [`Model`] node by node (old test-builder API).
        pub struct ModelBuilder {
            nodes: Vec<Node>,
            inputs: Vec<NodeId>,
            outputs: Vec<NodeId>,
            scale: u32,
        }

        impl Default for ModelBuilder {
            fn default() -> Self {
                Self::new()
            }
        }

        impl ModelBuilder {
            /// Builder at the default model scale.
            pub fn new() -> Self {
                Self::with_scale(DEFAULT_SCALE)
            }

            /// Builder at a custom log2 scale.
            pub fn with_scale(scale: u32) -> Self {
                Self {
                    nodes: Vec::new(),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    scale,
                }
            }

            fn dims(&self, w: Wire) -> Vec<usize> {
                self.nodes[w].output_dims.clone()
            }

            fn push(&mut self, op: Op, inputs: Vec<Wire>, output_dims: Vec<usize>) -> Wire {
                let id = self.nodes.len();
                self.nodes.push(Node {
                    op,
                    inputs: inputs.into_iter().map(|i| NodeId(i as u32)).collect(),
                    output_dims,
                });
                id
            }

            /// Add a graph input.
            pub fn input(&mut self, dims: Vec<usize>) -> Wire {
                let id = self.push(Op::Input(op::Input), vec![], dims);
                self.inputs.push(NodeId(id as u32));
                id
            }

            /// Add a constant tensor.
            pub fn constant(&mut self, tensor: Tensor<i32>) -> Wire {
                let dims = tensor.dims().to_vec();
                self.push(
                    Op::Constant(op::Constant(std::sync::Arc::new(tensor))),
                    vec![],
                    dims,
                )
            }

            /// Pass-through node.
            pub fn identity(&mut self, input: Wire) -> Wire {
                let d = self.dims(input);
                self.push(Op::Identity(op::Identity), vec![input], d)
            }

            /// ReLU activation.
            pub fn relu(&mut self, input: Wire) -> Wire {
                let d = self.dims(input);
                self.push(Op::ReLU(op::ReLU), vec![input], d)
            }

            /// Saturating addition.
            pub fn add(&mut self, a: Wire, b: Wire) -> Wire {
                let d = self.dims(a);
                self.push(Op::Add(op::Add), vec![a, b], d)
            }

            /// Saturating subtraction.
            pub fn sub(&mut self, a: Wire, b: Wire) -> Wire {
                let d = self.dims(a);
                self.push(Op::Sub(op::Sub), vec![a, b], d)
            }

            /// Negation.
            pub fn neg(&mut self, input: Wire) -> Wire {
                let d = self.dims(input);
                self.push(Op::Neg(op::Neg), vec![input], d)
            }

            /// Boolean product.
            pub fn and(&mut self, a: Wire, b: Wire) -> Wire {
                let d = self.dims(a);
                self.push(Op::And(op::And), vec![a, b], d)
            }

            /// Symmetric clamp into `[-2^bound_log, 2^bound_log)`.
            pub fn clamp(&mut self, input: Wire, bound_log: usize) -> Wire {
                let d = self.dims(input);
                self.push(
                    Op::Clamp(op::Clamp {
                        bound_log: bound_log as u32,
                    }),
                    vec![input],
                    d,
                )
            }

            /// Fused rescaling multiplication.
            pub fn mul(&mut self, a: Wire, b: Wire) -> Wire {
                let d = self.dims(a);
                self.push(Op::Mul(op::Mul { rebase: true }), vec![a, b], d)
            }

            /// Floor division.
            pub fn div(&mut self, a: Wire, b: Wire) -> Wire {
                let d = self.dims(a);
                self.push(Op::Div(op::Div), vec![a, b], d)
            }

            /// Floor division by a constant.
            pub fn scalar_const_div(&mut self, a: Wire, divisor: i32) -> Wire {
                let d = self.dims(a);
                self.push(
                    Op::ScalarConstDiv(op::ScalarConstDiv { divisor }),
                    vec![a],
                    d,
                )
            }

            /// Reciprocal square root.
            pub fn rsqrt(&mut self, input: Wire) -> Wire {
                let d = self.dims(input);
                self.push(Op::Rsqrt(op::Rsqrt), vec![input], d)
            }

            /// Select `mask·a + (1-mask)·b`.
            pub fn iff(&mut self, condition: Wire, true_branch: Wire, false_branch: Wire) -> Wire {
                let d = self.dims(true_branch);
                self.push(
                    Op::Iff(op::Iff),
                    vec![condition, true_branch, false_branch],
                    d,
                )
            }

            /// Broadcast-expand to `target_dims`.
            pub fn broadcast(&mut self, input: Wire, target_dims: Vec<usize>) -> Wire {
                self.push(
                    Op::Broadcast(op::Broadcast {
                        shape: target_dims.clone(),
                    }),
                    vec![input],
                    target_dims,
                )
            }

            /// Move one axis.
            pub fn moveaxis(&mut self, input: Wire, source: usize, destination: usize) -> Wire {
                let mut d = self.dims(input);
                let dim = d.remove(source);
                d.insert(destination, dim);
                self.push(
                    Op::MoveAxis(op::MoveAxis {
                        source,
                        destination,
                    }),
                    vec![input],
                    d,
                )
            }

            /// Fused einsum contraction.
            pub fn einsum(
                &mut self,
                equation: &str,
                inputs: Vec<Wire>,
                output_dims: Vec<usize>,
            ) -> Wire {
                let equation = equation
                    .parse()
                    .unwrap_or_else(|e| panic!("einsum {equation:?}: {e}"));
                self.push(Op::Einsum(op::Einsum { equation }), inputs, output_dims)
            }

            /// Reshape.
            pub fn reshape(&mut self, input: Wire, new_shape: Vec<usize>) -> Wire {
                self.push(
                    Op::Reshape(op::Reshape {
                        shape: new_shape.clone(),
                    }),
                    vec![input],
                    new_shape,
                )
            }

            /// Sum over axes. The node's dims follow keep-dims semantics
            /// (reduced axes become 1); a caller passing squeezed dims gets
            /// the keep-dims shape, which holds the same elements.
            pub fn sum(&mut self, input: Wire, axes: Vec<usize>, output_dims: Vec<usize>) -> Wire {
                let input_dims = self.dims(input);
                let keepdims: Vec<usize> = input_dims
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| if axes.contains(&i) { 1 } else { d })
                    .collect();
                debug_assert_eq!(
                    keepdims.iter().product::<usize>(),
                    output_dims.iter().product::<usize>().max(1),
                    "sum output dims disagree with the reduction"
                );
                self.push(Op::Sum(op::Sum { axes }), vec![input], keepdims)
            }

            /// Fused mean of squares over axes.
            pub fn mean_of_squares(
                &mut self,
                input: Wire,
                axes: Vec<usize>,
                output_dims: Vec<usize>,
            ) -> Wire {
                let input_dims = self.dims(input);
                let count: usize = axes.iter().map(|&ax| input_dims[ax]).product();
                let padded_count: usize = axes
                    .iter()
                    .map(|&ax| input_dims[ax].next_power_of_two())
                    .product();
                self.push(
                    Op::MeanOfSquares(op::MeanOfSquares {
                        axes,
                        count,
                        padded_count,
                    }),
                    vec![input],
                    output_dims,
                )
            }

            /// Fused rescaling square.
            pub fn square(&mut self, input: Wire) -> Wire {
                let d = self.dims(input);
                self.push(Op::Square(op::Square { rebase: true }), vec![input], d)
            }

            /// Cube; `scale == 0` is the raw building-block variant.
            pub fn cube(&mut self, input: Wire, scale: i32) -> Wire {
                let d = self.dims(input);
                self.push(Op::Cube(op::Cube { rebase: scale != 0 }), vec![input], d)
            }

            /// Embedding gather on axis 0.
            pub fn gather(
                &mut self,
                data: Wire,
                indices: Wire,
                axis: usize,
                output_dims: Vec<usize>,
            ) -> Wire {
                let dict_len = self.dims(data)[axis];
                let operator = if dict_len.next_power_of_two() <= 65536 {
                    Op::GatherSmall(op::GatherSmall { axis, dict_len })
                } else {
                    Op::GatherLarge(op::GatherLarge { axis, dict_len })
                };
                self.push(operator, vec![data, indices], output_dims)
            }

            /// Concatenate along `axis` (may be negative).
            pub fn concat(&mut self, inputs: &[Wire], axis: isize) -> Wire {
                assert!(!inputs.is_empty(), "Concat expects at least one input");
                let first_dims = self.dims(inputs[0]);
                let rank = first_dims.len();
                let axis_norm = if axis < 0 {
                    (axis + rank as isize) as usize
                } else {
                    axis as usize
                };
                let mut output_dims = first_dims;
                for &input in inputs.iter().skip(1) {
                    output_dims[axis_norm] += self.dims(input)[axis_norm];
                }
                self.push(
                    Op::Concat(op::Concat { axis: axis_norm }),
                    inputs.to_vec(),
                    output_dims,
                )
            }

            /// Slice along one axis.
            pub fn slice(&mut self, input: Wire, axis: usize, start: usize, end: usize) -> Wire {
                let mut d = self.dims(input);
                d[axis] = end - start;
                self.push(Op::Slice(op::Slice { axis, start, end }), vec![input], d)
            }

            /// Softmax over the last axis.
            pub fn softmax_last_axis(&mut self, input: Wire) -> Wire {
                let d = self.dims(input);
                self.push(
                    Op::SoftmaxLastAxis(op::SoftmaxLastAxis { active_len: None }),
                    vec![input],
                    d,
                )
            }

            /// Tanh activation.
            pub fn tanh(&mut self, input: Wire) -> Wire {
                let d = self.dims(input);
                self.push(Op::Tanh(op::Tanh), vec![input], d)
            }

            /// Erf activation.
            pub fn erf(&mut self, input: Wire) -> Wire {
                let d = self.dims(input);
                self.push(Op::Erf(op::Erf), vec![input], d)
            }

            /// Sigmoid activation.
            pub fn sigmoid(&mut self, input: Wire) -> Wire {
                let d = self.dims(input);
                self.push(Op::Sigmoid(op::Sigmoid), vec![input], d)
            }

            /// Sine.
            pub fn sin(&mut self, input: Wire) -> Wire {
                let d = self.dims(input);
                self.push(Op::Sin(op::Sin), vec![input], d)
            }

            /// Cosine.
            pub fn cos(&mut self, input: Wire) -> Wire {
                let d = self.dims(input);
                self.push(Op::Cos(op::Cos), vec![input], d)
            }

            /// Declare a graph output.
            pub fn mark_output(&mut self, w: Wire) {
                self.outputs.push(NodeId(w as u32));
            }

            /// Compile into a [`Model`] (dims are taken as given — no padding
            /// pass — matching the old builder; clamp widths are annotated).
            pub fn build(self) -> Model {
                let graph = Graph {
                    nodes: self.nodes,
                    inputs: self.inputs,
                    outputs: self.outputs,
                };
                let inner = atlas_trace::Model::compile(
                    graph,
                    CompileOptions {
                        scale: atlas_trace::Scale::new(self.scale as u8).expect("scale in range"),
                        pad_pow2: false,
                        version: SemVersion::V1,
                    },
                )
                .unwrap_or_else(|e| panic!("ModelBuilder::build: {e}"));
                Model::from_atlas(inner)
            }
        }
    }

    /// Trace types, old-shaped.
    pub mod trace {
        use super::*;
        use atlas_trace::trace::FusedIntermediates;
        use atlas_trace::SoftmaxWitness;

        pub use atlas_trace::trace::ModelExecutionIO;

        /// A captured execution: node outputs plus per-node witnesses.
        #[derive(Clone, Debug, PartialEq)]
        pub struct Trace {
            /// Output tensor per node index.
            pub node_outputs: BTreeMap<usize, Tensor<i32>>,
            /// Fused-rescale quotient/remainder per fused node.
            pub fused_intermediates: BTreeMap<usize, FusedIntermediates>,
            /// Full softmax witness per softmax node.
            pub softmax_witnesses: BTreeMap<usize, SoftmaxWitness>,
        }

        /// Operands and output of one node.
        #[derive(Debug, Clone)]
        pub struct LayerData<'a> {
            /// The node's output tensor.
            pub output: &'a Tensor<i32>,
            /// The node's operand tensors.
            pub operands: Vec<&'a Tensor<i32>>,
        }

        impl Trace {
            /// Convert from an [`atlas_trace::Trace`].
            pub fn from_atlas(t: atlas_trace::Trace) -> Self {
                let node_outputs = t.node_outputs.into_iter().enumerate().collect();
                let mut fused_intermediates = BTreeMap::new();
                let mut softmax_witnesses = BTreeMap::new();
                for (id, aux) in t.aux {
                    match aux {
                        NodeAux::Fused(f) => {
                            fused_intermediates.insert(id.index(), f);
                        }
                        NodeAux::Softmax(w) => {
                            softmax_witnesses.insert(id.index(), *w);
                        }
                    }
                }
                Trace {
                    node_outputs,
                    fused_intermediates,
                    softmax_witnesses,
                }
            }

            /// A trace from bare node outputs (tests / tampering harnesses).
            pub fn new(node_outputs: BTreeMap<usize, Tensor<i32>>) -> Self {
                Trace {
                    node_outputs,
                    fused_intermediates: BTreeMap::new(),
                    softmax_witnesses: BTreeMap::new(),
                }
            }

            /// Cached fused intermediates of a node, if any.
            pub fn fused_intermediates(&self, node_idx: usize) -> Option<&FusedIntermediates> {
                self.fused_intermediates.get(&node_idx)
            }

            /// The softmax witness of a node, if any.
            pub fn softmax_witness(&self, node_idx: usize) -> Option<&SoftmaxWitness> {
                self.softmax_witnesses.get(&node_idx)
            }

            /// Operands and output of a node.
            pub fn layer_data<'a>(&'a self, n: &node::ComputationNode) -> LayerData<'a> {
                LayerData {
                    output: &self[n.idx],
                    operands: self.operand_tensors(n),
                }
            }

            /// Operand tensors of a node.
            pub fn operand_tensors(&self, n: &node::ComputationNode) -> Vec<&Tensor<i32>> {
                n.inputs
                    .iter()
                    .map(|idx| self.node_outputs.get(idx).expect("operand traced"))
                    .collect()
            }

            /// Operand tensors with power-of-two padding applied.
            pub fn padded_operand_tensors(&self, n: &node::ComputationNode) -> Vec<Tensor<i32>> {
                self.operand_tensors(n)
                    .into_iter()
                    .map(Tensor::padded_pow2)
                    .collect()
            }

            /// The public IO of this execution.
            pub fn io(&self, model: &Model) -> ModelExecutionIO {
                ModelExecutionIO {
                    inputs: model
                        .inputs()
                        .iter()
                        .map(|idx| self.node_outputs[idx].clone())
                        .collect(),
                    outputs: model
                        .outputs()
                        .iter()
                        .map(|idx| self.node_outputs[idx].clone())
                        .collect(),
                    input_indices: model.inputs().to_vec(),
                    output_indices: model.outputs().to_vec(),
                }
            }
        }

        impl std::ops::Index<usize> for Trace {
            type Output = Tensor<i32>;

            fn index(&self, index: usize) -> &Self::Output {
                &self.node_outputs[&index]
            }
        }
    }
}
