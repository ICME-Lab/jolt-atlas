//! Test utilities for building `Model` instances programmatically.
//!
//! This module provides a builder pattern for constructing computation graphs
//! without needing to load from ONNX files.

use rand::rngs::StdRng;

use crate::{
    node::{ComputationNode, handlers::activation::NEURAL_TELEPORT_LOG_TABLE_SIZE},
    ops::*,
    tensor::Tensor,
    utils::f32::F32,
};
use std::collections::{BTreeMap, HashMap};

use super::{ComputationGraph, Model};

/// A wire represents a connection to a node's output in the computation graph.
pub type Wire = usize;

// number of bits dedicated to the fractional part in fixed-point representation
const SCALE: u32 = 7;

/// Builder for constructing `Model` instances programmatically.
///
/// This is useful for testing and creating simple computation graphs
/// without loading from ONNX files.
pub struct ModelBuilder {
    nodes: BTreeMap<usize, ComputationNode>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    next_id: usize,
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelBuilder {
    /// Create a new empty model builder.
    pub fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            next_id: 0,
        }
    }

    /// Allocate a new node ID.
    fn alloc(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Insert a node into the graph and return its wire.
    fn insert_node(&mut self, node: ComputationNode) -> Wire {
        let id = node.idx;
        self.nodes.insert(id, node);
        id
    }

    /// Add an input node with the given output dimensions.
    pub fn input(&mut self, dims: Vec<usize>) -> Wire {
        let id = self.alloc();
        let node = ComputationNode::new(id, Operator::Input(Input), vec![], dims);
        self.inputs.push(id);
        self.insert_node(node)
    }

    /// Add a constant tensor node.
    pub fn constant(&mut self, tensor: Tensor<i32>) -> Wire {
        let id = self.alloc();
        let dims = tensor.dims().to_vec();
        let node = ComputationNode::new(id, Operator::Constant(Constant(tensor)), vec![], dims);
        self.insert_node(node)
    }

    /// Add an identity (passthrough) node.
    pub fn identity(&mut self, input: Wire) -> Wire {
        let id = self.alloc();
        let output_dims = self.nodes[&input].output_dims.clone();
        let node = ComputationNode::new(id, Operator::Identity(Identity), vec![input], output_dims);
        self.insert_node(node)
    }

    /// Add a ReLU activation node.
    pub fn relu(&mut self, input: Wire) -> Wire {
        let id = self.alloc();
        let output_dims = self.nodes[&input].output_dims.clone();
        let node = ComputationNode::new(id, Operator::ReLU(ReLU), vec![input], output_dims);
        self.insert_node(node)
    }

    /// Add an addition node.
    pub fn add(&mut self, a: Wire, b: Wire) -> Wire {
        let id = self.alloc();
        let output_dims = self.nodes[&a].output_dims.clone();
        let node = ComputationNode::new(id, Operator::Add(Add), vec![a, b], output_dims);
        self.insert_node(node)
    }

    /// Add a subtraction node.
    pub fn sub(&mut self, a: Wire, b: Wire) -> Wire {
        let id = self.alloc();
        let output_dims = self.nodes[&a].output_dims.clone();
        let node = ComputationNode::new(id, Operator::Sub(Sub), vec![a, b], output_dims);
        self.insert_node(node)
    }

    /// Add a multiplication node.
    pub fn mul(&mut self, a: Wire, b: Wire) -> Wire {
        let id = self.alloc();
        let output_dims = self.nodes[&a].output_dims.clone();
        let node = ComputationNode::new(id, Operator::Mul(Mul), vec![a, b], output_dims);
        self.insert_node(node)
    }

    /// Add a division node.
    pub fn div(&mut self, a: Wire, b: Wire) -> Wire {
        let id = self.alloc();
        let output_dims = self.nodes[&a].output_dims.clone();
        let node = ComputationNode::new(id, Operator::Div(Div), vec![a, b], output_dims);
        self.insert_node(node)
    }

    /// Add a scalar constant division node.
    pub fn scalar_const_div(&mut self, a: Wire, divisor: i32) -> Wire {
        let id = self.alloc();
        let output_dims = self.nodes[&a].output_dims.clone();
        let node = ComputationNode::new(
            id,
            Operator::ScalarConstDiv(ScalarConstDiv { divisor }),
            vec![a],
            output_dims,
        );
        self.insert_node(node)
    }

    /// Add a reciprocal square root (rsqrt) node.
    pub fn rsqrt(&mut self, input: Wire) -> Wire {
        let id = self.alloc();
        let output_dims = self.nodes[&input].output_dims.clone();
        let node = ComputationNode::new(
            id,
            Operator::Rsqrt(Rsqrt {
                scale: F32(SCALE as f32),
            }),
            vec![input],
            output_dims,
        );
        self.insert_node(node)
    }

    /// Add a Iff node.
    pub fn iff(&mut self, condition: Wire, true_branch: Wire, false_branch: Wire) -> Wire {
        let id = self.alloc();
        let output_dims = self.nodes[&true_branch].output_dims.clone();
        let node = ComputationNode::new(
            id,
            Operator::Iff(Iff),
            vec![condition, true_branch, false_branch],
            output_dims,
        );
        self.insert_node(node)
    }

    /// Add a broadcast node (expand dimensions to match target shape).
    pub fn broadcast(&mut self, input: Wire, target_dims: Vec<usize>) -> Wire {
        let id = self.alloc();
        let node = ComputationNode::new(
            id,
            Operator::Broadcast(Broadcast {
                shape: target_dims.clone(),
            }),
            vec![input],
            target_dims,
        );
        self.insert_node(node)
    }

    /// Add a moveaxis node (transpose an axis from source to destination position).
    pub fn moveaxis(&mut self, input: Wire, source: usize, destination: usize) -> Wire {
        let id = self.alloc();
        let input_dims = self.nodes[&input].output_dims.clone();
        let mut output_dims = input_dims.clone();

        let dim = output_dims.remove(source);
        output_dims.insert(destination, dim);
        let node = ComputationNode::new(
            id,
            Operator::MoveAxis(MoveAxis {
                source,
                destination,
            }),
            vec![input],
            output_dims,
        );
        self.insert_node(node)
    }

    /// Add an einsum node (general tensor contraction).
    pub fn einsum(&mut self, equation: &str, inputs: Vec<Wire>, output_dims: Vec<usize>) -> Wire {
        let id = self.alloc();
        let node = ComputationNode::new(
            id,
            Operator::Einsum(Einsum {
                equation: equation.to_string(),
            }),
            inputs,
            output_dims,
        );
        self.insert_node(node)
    }

    /// Add a reshape node.
    pub fn reshape(&mut self, input: Wire, new_shape: Vec<usize>) -> Wire {
        let id = self.alloc();
        let node = ComputationNode::new(
            id,
            Operator::Reshape(Reshape {
                shape: new_shape.clone(),
            }),
            vec![input],
            new_shape,
        );
        self.insert_node(node)
    }

    /// Add a sum reduction node.
    pub fn sum(&mut self, input: Wire, axes: Vec<usize>, output_dims: Vec<usize>) -> Wire {
        let id = self.alloc();
        let node = ComputationNode::new(id, Operator::Sum(Sum { axes }), vec![input], output_dims);
        self.insert_node(node)
    }

    /// Add a square node.
    pub fn square(&mut self, input: Wire) -> Wire {
        let id = self.alloc();
        let output_dims = self.nodes[&input].output_dims.clone();
        let node = ComputationNode::new(id, Operator::Square(Square), vec![input], output_dims);
        self.insert_node(node)
    }

    /// Add a cube node.
    pub fn cube(&mut self, input: Wire) -> Wire {
        let id = self.alloc();
        let output_dims = self.nodes[&input].output_dims.clone();
        let node = ComputationNode::new(id, Operator::Cube(Cube), vec![input], output_dims);
        self.insert_node(node)
    }

    /// Add a gather node.
    pub fn gather(
        &mut self,
        data: Wire,
        indices: Wire,
        axis: usize,
        output_dims: Vec<usize>,
    ) -> Wire {
        let id = self.alloc();
        let node = ComputationNode::new(
            id,
            Operator::Gather(Gather { axis }),
            vec![data, indices],
            output_dims,
        );
        self.insert_node(node)
    }

    /// Add a softmax node.
    pub fn softmax(&mut self, axes: usize, input: Wire) -> Wire {
        let id = self.alloc();
        let output_dims = self.nodes[&input].output_dims.clone();
        let node = ComputationNode::new(
            id,
            Operator::SoftmaxAxes(SoftmaxAxes {
                axes,
                scale: F32((1 << SCALE) as f32),
            }),
            vec![input],
            output_dims,
        );
        self.insert_node(node)
    }

    /// Add a hyperbolic tangent (tanh) activation node.
    pub fn tanh(&mut self, input: Wire) -> Wire {
        let id = self.alloc();
        let output_dims = self.nodes[&input].output_dims.clone();
        let tau = 2;
        let node = ComputationNode::new(
            id,
            Operator::Tanh(Tanh {
                scale: F32((1 << SCALE) as f32),
                tau,
                log_table: NEURAL_TELEPORT_LOG_TABLE_SIZE,
            }),
            vec![input],
            output_dims,
        );
        self.insert_node(node)
    }

    /// Mark a wire as an output of the model.
    pub fn mark_output(&mut self, wire: Wire) {
        self.outputs.push(wire);
    }

    /// Build and consume the builder, returning the constructed `Model`.
    pub fn build(self) -> Model {
        Model {
            graph: ComputationGraph {
                nodes: self.nodes,
                inputs: self.inputs,
                outputs: self.outputs,
                original_input_dims: HashMap::new(),
                original_output_dims: HashMap::new(),
            },
        }
    }
}

/// Create a model for element-wise multiplication of an input and constant tensor.
pub fn mul_model(rng: &mut StdRng, T: usize) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![T]);
    let c = b.constant(Tensor::random_small(rng, &[T]));
    let res = b.mul(i, c);
    b.mark_output(res);
    b.build()
}

/// Create a model for matrix multiplication (mk,kn->mn).
pub fn matmul_model(rng: &mut StdRng, m: usize, k: usize, n: usize) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![m, k]);
    let c = b.constant(Tensor::random_small(rng, &[k, n]));
    let res = b.einsum("mk,kn->mn", vec![i, c], vec![m, n]);
    b.mark_output(res);
    b.build()
}

/// Create a model for matrix-vector multiplication (k,nk->n).
pub fn matvec_model(rng: &mut StdRng, k: usize, n: usize) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![k]);
    let c = b.constant(Tensor::random_small(rng, &[n, k]));
    let res = b.einsum("k,nk->n", vec![i, c], vec![n]);
    b.mark_output(res);
    b.build()
}

/// Create a model for einsum contraction (mbk,nbk->bmn).
pub fn mbk_nbk_bmn_model(rng: &mut StdRng, m: usize, b: usize, k: usize, n: usize) -> Model {
    let mut builder = ModelBuilder::new();
    let i = builder.input(vec![m, b, k]);
    let c = builder.constant(Tensor::random_small(rng, &[n, b, k]));
    let res = builder.einsum("mbk,nbk->bmn", vec![i, c], vec![b, m, n]);
    builder.mark_output(res);
    builder.build()
}

/// Create a model for einsum contraction (mbk,bnk->bmn).
pub fn mbk_bnk_bmn_model(rng: &mut StdRng, m: usize, b: usize, k: usize, n: usize) -> Model {
    let mut builder = ModelBuilder::new();
    let i = builder.input(vec![m, b, k]);
    let c = builder.constant(Tensor::random_small(rng, &[b, n, k]));
    let res = builder.einsum("mbk,bnk->bmn", vec![i, c], vec![b, m, n]);
    builder.mark_output(res);
    builder.build()
}

/// Create a model for einsum contraction (bmk,kbn->mbn).
pub fn bmk_kbn_mbn_model(rng: &mut StdRng, b: usize, m: usize, k: usize, n: usize) -> Model {
    let mut builder = ModelBuilder::new();
    let i = builder.input(vec![b, m, k]);
    let c = builder.constant(Tensor::random_small(rng, &[k, b, n]));
    let res = builder.einsum("bmk,kbn->mbn", vec![i, c], vec![m, b, n]);
    builder.mark_output(res);
    builder.build()
}

/// Create a model for einsum contraction (bmk,bkn->mbn).
pub fn bmk_bkn_mbn_model(rng: &mut StdRng, b: usize, m: usize, k: usize, n: usize) -> Model {
    let mut builder = ModelBuilder::new();
    let i = builder.input(vec![b, m, k]);
    let c = builder.constant(Tensor::random_small(rng, &[b, k, n]));
    let res = builder.einsum("bmk,bkn->mbn", vec![i, c], vec![m, b, n]);
    builder.mark_output(res);
    builder.build()
}

/// Create a model for element-wise subtraction of an input and constant tensor.
pub fn sub_model(rng: &mut StdRng, T: usize) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![T]);
    let c = b.constant(Tensor::random_small(rng, &[T]));
    let res = b.sub(i, c);
    b.mark_output(res);
    b.build()
}

/// Create a model for element-wise squaring of an input tensor.
pub fn square_model(T: usize) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![T]);
    let res = b.square(i);
    b.mark_output(res);
    b.build()
}

/// Create a model for ReLU activation.
pub fn relu_model(T: usize) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![T]);
    let res = b.relu(i);
    b.mark_output(res);
    b.build()
}

/// Create a model for element-wise cubing of an input tensor.
pub fn cube_model(T: usize) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![T]);
    let res = b.cube(i);
    b.mark_output(res);
    b.build()
}

/// Create a model for element-wise division of an input by a constant tensor.
pub fn div_model(T: usize) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![T]);
    let c = b.constant(Tensor::construct(vec![128; T], vec![T]));
    let res = b.div(i, c);
    b.mark_output(res);
    b.build()
}

/// Create a model for element-wise division of an input by a scalar constant.
pub fn scalar_const_div_model(T: usize, divisor: i32) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![T]);
    let res = b.scalar_const_div(i, divisor);
    b.mark_output(res);
    b.build()
}

/// Create a model for element-wise reciprocal (1/x) of an input.
pub fn recip_model(T: usize) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![T]);
    let c = b.constant(Tensor::construct(vec![1 << (SCALE * 2); T], vec![T]));
    let res = b.div(c, i);
    b.mark_output(res);
    b.build()
}

/// Create a model for element-wise reciprocal square root (1/sqrt(x)).
pub fn rsqrt_model(T: usize) -> Model {
    let mut b = ModelBuilder::new();
    let input = b.input(vec![T]);
    let res = b.rsqrt(input);
    b.mark_output(res);
    b.build()
}

/// Create a model for conditional selection (if-then-else).
pub fn iff_model(rng: &mut StdRng, T: usize) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![T]);
    let mask = b.constant(Tensor::random_boolean(rng, &[T]));
    let c0 = b.constant(Tensor::random(rng, &[T]));
    let res = b.iff(mask, i, c0);
    b.mark_output(res);
    b.build()
}

/// Create a model for broadcasting an input to a target shape.
pub fn broadcast_model(input_shape: &[usize], output_shape: &[usize]) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(input_shape.to_vec());
    let res = b.broadcast(i, output_shape.to_vec());
    b.mark_output(res);
    b.build()
}

/// Create a model for moving an axis from one position to another.
pub fn moveaxis_model(input_shape: &[usize], source: usize, destination: usize) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(input_shape.to_vec());
    let res = b.moveaxis(i, source, destination);
    b.mark_output(res);
    b.build()
}

/// Create a model for reshaping a tensor to a new shape.
pub fn reshape_model(input_shape: &[usize], output_shape: &[usize]) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(input_shape.to_vec());
    let res = b.reshape(i, output_shape.to_vec());
    b.mark_output(res);
    b.build()
}

/// Create a model for softmax activation along specified axes.
pub fn softmax_axes_model(input_shape: &[usize], axes: usize) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(input_shape.to_vec());
    let res = b.softmax(axes, i);
    b.mark_output(res);
    b.build()
}

/// Creates a model featuring a gather operation (embedding lookup), with axis=0
pub fn gather_model(input_shape: &[usize], dictionnary_len: usize, word_dim: usize) -> Model {
    let mut b = ModelBuilder::new();
    let dictionnary = {
        let data = (0..dictionnary_len * word_dim)
            .map(|i| i as i32)
            .collect::<Vec<_>>();
        Tensor::construct(data, vec![dictionnary_len, word_dim])
    };
    let dict = b.constant(dictionnary);
    let indexes = b.input(input_shape.to_vec());

    let res = b.gather(
        dict,
        indexes,
        0,
        [input_shape.to_vec(), vec![word_dim]].concat(),
    );
    b.mark_output(res);
    b.build()
}

/// Create a model for summing a 2D tensor along a specified axis.
pub fn sum_model<const AXIS: usize>(m: usize, n: usize) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![m, n]);
    let res = b.sum(i, vec![AXIS], vec![m, 1]);
    b.mark_output(res);
    b.build()
}

/// Create a model for hyperbolic tangent (tanh) activation.
pub fn tanh_model(input_shape: &[usize]) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(input_shape.to_vec());
    let res = b.tanh(i);
    b.mark_output(res);
    b.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_add() {
        let mut b = ModelBuilder::new();

        let x = b.input(vec![2, 3]);
        let y = b.input(vec![2, 3]);
        let z = b.add(x, y);
        b.mark_output(z);

        let model = b.build();

        assert_eq!(model.graph.inputs.len(), 2);
        assert_eq!(model.graph.outputs.len(), 1);
        assert_eq!(model.graph.nodes.len(), 3);
    }

    #[test]
    fn test_mlp_like() {
        let mut b = ModelBuilder::new();

        // Input: [1, 4]
        let input = b.input(vec![1, 4]);

        // Weights: [4, 8]
        let w1 = b.constant(Tensor::new(None, &[4, 8]).unwrap());

        // MatMul via einsum: [1,4] x [4,8] -> [1,8]
        let h1 = b.einsum("mk,kn->mn", vec![input, w1], vec![1, 8]);

        // ReLU
        let h1_relu = b.relu(h1);

        // Weights: [8, 2]
        let w2 = b.constant(Tensor::new(None, &[8, 2]).unwrap());

        // MatMul via einsum: [1,8] x [8,2] -> [1,2]
        let output = b.einsum("mk,kn->mn", vec![h1_relu, w2], vec![1, 2]);

        b.mark_output(output);

        let model = b.build();

        assert_eq!(model.graph.inputs.len(), 1);
        assert_eq!(model.graph.outputs.len(), 1);
        assert_eq!(model.graph.nodes.len(), 6);
    }
}
