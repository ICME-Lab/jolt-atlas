//! Programmatic [`atlas_trace::Model`] construction for tests (explicit dims,
//! no padding pass unless [`ModelBuilder::build_padded`] is used).

use atlas_trace::{op, CompileOptions, Graph, Node, NodeId, Op, SemVersion, Tensor};

/// A connection to a node's output.
pub type Wire = usize;

const DEFAULT_SCALE: u8 = common::consts::MODEL_SCALE as u8;

/// Builds a model node by node.
pub struct ModelBuilder {
    nodes: Vec<Node>,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
    scale: u8,
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
    pub fn with_scale(scale: u8) -> Self {
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

    /// Raw (un-rescaled) product, the building block of decompositions.
    pub fn mul_raw(&mut self, a: Wire, b: Wire) -> Wire {
        let d = self.dims(a);
        self.push(Op::Mul(op::Mul { rebase: false }), vec![a, b], d)
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
    pub fn einsum(&mut self, equation: &str, inputs: Vec<Wire>, output_dims: Vec<usize>) -> Wire {
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
    pub fn cube(&mut self, input: Wire, scale: u8) -> Wire {
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

    /// Softmax over the last axis with only the first `active_len` columns live
    /// (the masked form the proof layer rejects at preprocessing).
    pub fn softmax_last_axis_masked(&mut self, input: Wire, active_len: usize) -> Wire {
        let d = self.dims(input);
        self.push(
            Op::SoftmaxLastAxis(op::SoftmaxLastAxis {
                active_len: Some(active_len),
            }),
            vec![input],
            d,
        )
    }

    /// Always-zero NaN check.
    pub fn is_nan(&mut self, input: Wire) -> Wire {
        let d = self.dims(input);
        self.push(Op::IsNan(op::IsNan), vec![input], d)
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

    /// Compile into an [`atlas_trace::Model`] (dims are taken as given —
    /// no padding pass; clamp widths are annotated).
    pub fn build(self) -> atlas_trace::Model {
        self.build_with(false)
    }

    /// Compile with the production power-of-two padding pass, as the ONNX
    /// importer does (`pad_pow2: true`); inputs are then supplied unpadded
    /// and padded by the executor.
    pub fn build_padded(self) -> atlas_trace::Model {
        self.build_with(true)
    }

    fn build_with(self, pad_pow2: bool) -> atlas_trace::Model {
        let graph = Graph {
            nodes: self.nodes,
            inputs: self.inputs,
            outputs: self.outputs,
        };
        atlas_trace::Model::compile(
            graph,
            CompileOptions {
                scale: atlas_trace::Scale::new(self.scale).expect("scale in range"),
                pad_pow2,
                version: SemVersion::V1,
            },
        )
        .unwrap_or_else(|e| panic!("ModelBuilder::build: {e}"))
    }
}
