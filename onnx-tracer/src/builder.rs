//! Model builder utilities for creating ONNX computation graphs.
//!
//! This module provides a `ModelBuilder` struct that simplifies the creation of
//! neural network models for testing and development. It offers high-level methods
//! for common operations like matrix multiplication, element-wise operations, and more.
//!
//! It supposes that the model's input node is always the first node (idx 0),
//! and that the nodes are correctly broadcasted with broadcast nodes where necessary.
//!
//! # Example: Simple Matrix Multiplication
//!
//! ```ignore
//! use onnx_tracer::builder::simple_matmult_model;
//! use onnx_tracer::tensor::Tensor;
//!
//! // Create a simple matrix multiplication model
//! let model = simple_matmult_model();
//!
//! // Test with input [1, 2, 3, 4]
//! let /// Simple matrix multiplication model for testing ONNX MatMul semantics.
///
/// This model demonstrates ONNX matrix multiplication functionality:
/// 1. Takes an input tensor of shape [1, 4]
/// 2. Multiplies it with a constant weight matrix of shape [3, 4] (gets implicitly transposed)
/// 3. Outputs the result of shape [1, 3]
///
/// **ONNX MatMul Behavior**: The second matrix is implicitly transposed, so:
/// - Input: [1, 4]
/// - Weights: [3, 4] (stored as [3, 4], but acts like [4, 3] due to implicit transpose)
/// - Result: [1, 3]
///
/// The weight matrix contains simple values for easy verification:
/// ```ignore
/// weights = [[1, 4, 7, 10],    // First output neuron weights
///            [2, 5, 8, 11],    // Second output neuron weights  
///            [3, 6, 9, 12]]    // Third output neuron weights
/// ```
use crate::{
    graph::{
        model::Model,
        node::{RebaseScale, SupportedOp},
        utilities::{
            create_const_node, create_div_node, create_iff_node, create_input_node,
            create_matmul_node, create_node, create_polyop_node, create_relu_node,
        },
    },
    ops::{hybrid::HybridOp, poly::PolyOp},
    tensor::Tensor,
};

type Wire = (usize, usize); // (node_id, output_idx)
const O: usize = 0; // single-output nodes use 0

struct ModelBuilder {
    model: Model,
    next_id: usize,
    scale: i32,
}

impl ModelBuilder {
    fn new(scale: i32) -> Self {
        Self {
            model: Model::default(),
            next_id: 0,
            scale,
        }
    }

    fn take(self, inputs: Vec<usize>, outputs: Vec<Wire>) -> Model {
        let mut m = self.model;
        m.set_inputs(inputs);
        m.set_outputs(outputs);
        m
    }

    fn alloc(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn input(&mut self, dims: Vec<usize>, fanout_hint: usize) -> Wire {
        let id = self.alloc();
        let n = create_input_node(self.scale, dims, id, fanout_hint);
        self.model.insert_node(n);
        (id, O)
    }

    fn const_tensor(
        &mut self,
        tensor: Tensor<i32>,
        out_dims: Vec<usize>,
        fanout_hint: usize,
    ) -> Wire {
        let id = self.alloc();
        let raw = Tensor::new(Some(&[] as &[f32]), &[0]).unwrap();
        let n = create_const_node(tensor, raw, self.scale, out_dims, id, fanout_hint);
        self.model.insert_node(n);
        (id, O)
    }

    fn poly(
        &mut self,
        op: PolyOp<i32>,
        a: Wire,
        b: Wire,
        out_dims: Vec<usize>,
        fanout_hint: usize,
    ) -> Wire {
        let id = self.alloc();
        let n = create_polyop_node(op, self.scale, vec![a, b], out_dims, id, fanout_hint);
        self.model.insert_node(n);
        (id, O)
    }

    fn div(&mut self, divisor: i32, x: Wire, out_dims: Vec<usize>, fanout_hint: usize) -> Wire {
        let id = self.alloc();
        let n = create_div_node(divisor, self.scale, vec![x], out_dims, id, fanout_hint);
        self.model.insert_node(n);
        (id, O)
    }

    fn gather(
        &mut self,
        data: Wire,
        indices: Wire,
        dim: usize,
        out_dims: Vec<usize>,
        fanout_hint: usize,
    ) -> Wire {
        let id = self.alloc();
        let gather_op = HybridOp::Gather {
            dim,
            constant_idx: None,
        };
        let gather_node = create_node(
            SupportedOp::Hybrid(gather_op),
            self.scale,
            vec![data, indices],
            out_dims,
            id,
            fanout_hint,
        );
        self.model.insert_node(gather_node);
        (id, O)
    }

    fn reshape(
        &mut self,
        input: Wire,
        new_shape: Vec<usize>,
        out_dims: Vec<usize>,
        fanout_hint: usize,
    ) -> Wire {
        let id = self.alloc();
        let reshape_node = create_node(
            SupportedOp::Linear(PolyOp::Reshape(new_shape)),
            self.scale,
            vec![input],
            out_dims,
            id,
            fanout_hint,
        );
        self.model.insert_node(reshape_node);
        (id, O)
    }

    fn sum(
        &mut self,
        input: Wire,
        axes: Vec<usize>,
        out_dims: Vec<usize>,
        fanout_hint: usize,
    ) -> Wire {
        let id = self.alloc();
        let sum_node = create_node(
            SupportedOp::Linear(PolyOp::Sum { axes }),
            self.scale,
            vec![input],
            out_dims,
            id,
            fanout_hint,
        );
        self.model.insert_node(sum_node);
        (id, O)
    }

    fn greater_equal(
        &mut self,
        a: Wire,
        b: Wire,
        out_dims: Vec<usize>,
        fanout_hint: usize,
    ) -> Wire {
        let id = self.alloc();
        let gte_node = create_node(
            SupportedOp::Hybrid(HybridOp::GreaterEqual),
            0, // Binary output has scale 0
            vec![a, b],
            out_dims,
            id,
            fanout_hint,
        );
        self.model.insert_node(gte_node);
        (id, O)
    }

    fn iff(
        &mut self,
        condition: Wire,
        if_true: Wire,
        if_false: Wire,
        out_dims: Vec<usize>,
        fanout_hint: usize,
    ) -> Wire {
        let id = self.alloc();
        let iff_node = create_iff_node(
            self.scale,
            vec![condition, if_true, if_false],
            out_dims,
            id,
            fanout_hint,
        );
        self.model.insert_node(iff_node);
        (id, O)
    }

    fn const_tensor_with_scale(
        &mut self,
        tensor: Tensor<i32>,
        scale: i32,
        out_dims: Vec<usize>,
        fanout_hint: usize,
    ) -> Wire {
        let id = self.alloc();
        let raw = Tensor::new(Some(&[] as &[f32]), &[0]).unwrap();
        let n = create_const_node(tensor, raw, scale, out_dims, id, fanout_hint);
        self.model.insert_node(n);
        (id, O)
    }

    fn argmax(
        &mut self,
        input: Wire,
        dim: usize,
        out_dims: Vec<usize>,
        fanout_hint: usize,
    ) -> Wire {
        let id = self.alloc();
        let argmax_node = create_node(
            SupportedOp::Hybrid(HybridOp::ReduceArgMax { dim }),
            0, // ArgMax output has scale 0 (returns indices)
            vec![input],
            out_dims,
            id,
            fanout_hint,
        );
        self.model.insert_node(argmax_node);
        (id, O)
    }

    fn rebase_scale_mul(
        &mut self,
        a: Wire,
        b: Wire,
        out_dims: Vec<usize>,
        fanout_hint: usize,
    ) -> Wire {
        let id = self.alloc();
        let opkind = SupportedOp::RebaseScale(RebaseScale {
            inner: Box::new(SupportedOp::Linear(PolyOp::Mult)),
            multiplier: 2f64.powi(self.scale),
            target_scale: self.scale,
            original_scale: self.scale * 2,
        });
        let rebase_node = create_node(opkind, self.scale, vec![a, b], out_dims, id, fanout_hint);
        self.model.insert_node(rebase_node);
        (id, O)
    }

    /// Performs matrix multiplication wrapped in RebaseScale for ONNX binary compilation scenarios.
    ///
    /// This function wraps the MatMul operation in RebaseScale, which is commonly used
    /// when compiling ONNX models to binary format for handling quantization scaling.
    ///
    /// # Arguments
    /// * `a` - First input tensor (left operand)
    /// * `b` - Second input tensor (right operand)
    /// * `out_dims` - Expected output dimensions
    /// * `fanout_hint` - Hint for optimization purposes
    ///
    /// # Returns
    /// A `Wire` representing the RebaseScale-wrapped matrix multiplication result
    fn rebase_scale_matmult(
        &mut self,
        a: Wire,
        b: Wire,
        out_dims: Vec<usize>,
        fanout_hint: usize,
    ) -> Wire {
        let id = self.alloc();
        let matmul_op = SupportedOp::Linear(PolyOp::Einsum {
            equation: "mk,nk->mn".to_string(),
        });
        let opkind = SupportedOp::RebaseScale(RebaseScale {
            inner: Box::new(matmul_op),
            multiplier: 2f64.powi(self.scale),
            target_scale: self.scale,
            original_scale: self.scale, // Keep same scale to avoid division
        });
        let rebase_node = create_node(opkind, self.scale, vec![a, b], out_dims, id, fanout_hint);
        self.model.insert_node(rebase_node);
        (id, O)
    }

    fn broadcast(
        &mut self,
        input: Wire,
        target_shape: Vec<usize>,
        out_dims: Vec<usize>,
        fanout_hint: usize,
    ) -> Wire {
        let id = self.alloc();
        let broadcast_node = create_node(
            SupportedOp::Linear(PolyOp::MultiBroadcastTo {
                shape: target_shape,
            }),
            self.scale,
            vec![input],
            out_dims,
            id,
            fanout_hint,
        );
        self.model.insert_node(broadcast_node);
        (id, O)
    }

    /// Performs matrix multiplication between two input tensors.
    ///
    /// Matrix multiplication is implemented using the Einsum operation with the equation "ij,jk->ik".
    /// This computes the standard matrix product where the inner dimensions must match.
    ///
    /// # Arguments
    /// * `a` - First input tensor (left operand)
    /// * `b` - Second input tensor (right operand)
    /// * `out_dims` - Expected output dimensions
    /// * `fanout_hint` - Hint for optimization purposes
    ///
    /// # Returns
    /// A `Wire` representing the matrix multiplication result
    ///
    /// # Example
    /// For input tensors of shapes [m, n] and [n, p], the output will have shape [m, p].
    fn matmult(&mut self, a: Wire, b: Wire, out_dims: Vec<usize>, fanout_hint: usize) -> Wire {
        let id = self.alloc();
        let matmul_node = create_matmul_node(
            "mk,nk->mn".to_string(), // ONNX MatMul equation (implicitly transposes second matrix)
            self.scale,
            vec![a, b],
            out_dims,
            id,
            fanout_hint,
        );
        self.model.insert_node(matmul_node);
        (id, O)
    }

    /// Applies ReLU (Rectified Linear Unit) activation function.
    ///
    /// ReLU is defined as f(x) = max(0, x), which zeros out negative values
    /// while keeping positive values unchanged.
    ///
    /// # Arguments
    /// * `input` - Input tensor to apply ReLU to
    /// * `out_dims` - Expected output dimensions (same as input dimensions)
    /// * `fanout_hint` - Hint for optimization purposes
    ///
    /// # Returns
    /// A `Wire` representing the ReLU result
    fn relu(&mut self, input: Wire, out_dims: Vec<usize>, fanout_hint: usize) -> Wire {
        let id = self.alloc();
        let relu_node = create_relu_node(self.scale, vec![input], out_dims, id, fanout_hint);
        self.model.insert_node(relu_node);
        (id, O)
    }
}

/* ********************** Testing Model's ********************** */

/// Simplified multiclass classification model using only basic operations:
/// input, const, gather, matmult, add, sub, mul, div, reshape
///
/// This model:
/// 1. Takes embedding tensor and input indices
/// 2. Gathers embeddings based on input indices  
/// 3. Sums the gathered embeddings via matrix multiplication
/// 4. Performs linear transformation (matmult + add bias)
/// 5. Returns logits (no ArgMax - let caller handle classification)
pub fn multiclass1() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: Input indices (shape [1, 8])
    let input_indices = b.input(vec![1, 8], 1);

    // Node 1: Embedding matrix (shape [31, 1]) - same values as before
    let mut embedding: Tensor<i32> = Tensor::new(
        Some(&[
            -61, -287, -437, -294, -318, 345, 331, 330, -28, 337, 113, 111, 91, 103, -58, 85, 72,
            -463, -342, -345, -318, 355, 385, 376, 180, 125, 10, 143, 137, -45, 128,
        ]),
        &[31, 1],
    )
    .unwrap();
    embedding.set_scale(SCALE);
    let embedding_const = b.const_tensor_with_scale(embedding, SCALE, vec![31, 1], 1);

    // Node 2: Gather embeddings - results in [1, 8, 1]
    let gathered = b.gather(embedding_const, input_indices, 0, vec![1, 8, 1], 1);

    // Node 3: Reshape gathered embeddings to [1, 8] for matrix operations
    let reshaped_gathered = b.reshape(gathered, vec![1, 8], vec![1, 8], 1);

    // Node 4: Sum weights (all ones) to sum embeddings via matrix multiplication
    // Shape [8, 1] - each row is 1, so matmult will sum the 8 embeddings
    let mut sum_weights: Tensor<i32> = Tensor::new(Some(&[1; 8]), &[8, 1]).unwrap();
    sum_weights.set_scale(SCALE);
    let sum_weights_const = b.const_tensor_with_scale(sum_weights, SCALE, vec![8, 1], 1);

    // Node 5: Sum embeddings via matrix multiplication [1, 8] × [8, 1] → [1, 1]
    let summed = b.matmult(reshaped_gathered, sum_weights_const, vec![1, 1], 1);

    // Node 6: Weight matrix for classification (shape [1, 10])
    let mut weights: Tensor<i32> = Tensor::new(
        Some(&[388, 16, -93, 517, 208, 208, 208, 208, 208, 208]),
        &[1, 10],
    )
    .unwrap();
    weights.set_scale(SCALE);
    let weights_const = b.const_tensor_with_scale(weights, SCALE, vec![1, 10], 1);

    // Node 7: Linear transformation [1, 1] × [1, 10] → [1, 10]
    // This broadcasts the scalar across all 10 classes and applies weights
    let weighted = b.matmult(summed, weights_const, vec![1, 10], 1);

    // Node 8: Scale down by dividing by 128 (replacing the RebaseScale division)
    let scaled = b.div(128, weighted, vec![1, 10], 1);

    // Node 9: Bias vector (shape [1, 10])
    let mut bias: Tensor<i32> = Tensor::new(
        Some(&[449, 421, -137, -95, -155, -155, -155, -155, -155, -155]),
        &[1, 10],
    )
    .unwrap();
    bias.set_scale(SCALE);
    let bias_const = b.const_tensor_with_scale(bias, SCALE, vec![1, 10], 1);

    // Node 10: Add bias to get final logits
    let logits = b.poly(PolyOp::Add, scaled, bias_const, vec![1, 10], 1);

    // Return logits instead of argmax - let the caller handle classification
    // This is more flexible and potentially faster for training scenarios
    b.take(vec![input_indices.0], vec![logits])
}

/// Creates a model with 3 nodes
/// Has a trace lenght of 2^s - 1
pub fn add_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);
    let mut const_tensor: Tensor<i32> = Tensor::new(Some(&[50, 60, 70, 80]), &[1, 4]).unwrap();
    const_tensor.set_scale(SCALE);

    let x = b.input(vec![1, 4], 1);
    let a = b.const_tensor(const_tensor, vec![1, 4], 1);
    let y = b.poly(PolyOp::Add, x, a, vec![1, 4], 1);
    b.take(vec![x.0], vec![y])
}

/// Creates a model with 4 nodes
/// Has a trace lenght of 2^s
pub fn addmul_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);
    let mut const_tensor: Tensor<i32> = Tensor::new(Some(&[50, 60, 70, 80]), &[1, 4]).unwrap();
    const_tensor.set_scale(SCALE);

    let x = b.input(vec![1, 4], 2);
    let a = b.const_tensor(const_tensor, vec![1, 4], 1);
    let s = b.poly(PolyOp::Add, x, a, vec![1, 4], 1);
    let y = b.poly(PolyOp::Mult, x, s, vec![1, 4], 1);
    b.take(vec![x.0], vec![y])
}

/// [(0, input, []), (1, add, [0, 0]), (2, sub, [1, 0]), (3, mul, [1, 2]), (4, add, [2, 3]), (5, output, [4])]
pub fn addsubmul_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);
    let out_dims = vec![1, 4];

    let x = b.input(out_dims.clone(), 2);
    let a = b.poly(PolyOp::Add, x, x, out_dims.clone(), 2);
    let s = b.poly(PolyOp::Sub, a, x, out_dims.clone(), 1);
    let m = b.poly(PolyOp::Mult, a, s, out_dims.clone(), 1);
    let y = b.poly(PolyOp::Add, s, m, out_dims.clone(), 1);

    b.take(vec![x.0], vec![y])
}

/// [(0, input, []), (1, const, []), (2, add, [0, 1]), (3, sub, [0, 1]), (4, mul, [2, 3]), (5, output, [4])]
pub fn addsubmulconst_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    let x = b.input(vec![1, 4], 2);
    let mut c: Tensor<i32> = Tensor::new(Some(&[50, 60, 70, 80]), &[1, 4]).unwrap();
    c.set_scale(SCALE);
    let k = b.const_tensor(c, vec![1, 4], 2);

    let a = b.poly(PolyOp::Add, x, k, vec![1, 4], 1);
    let s = b.poly(PolyOp::Sub, x, k, vec![1, 4], 1);
    let y = b.poly(PolyOp::Mult, a, s, vec![1, 4], 1);

    b.take(vec![x.0], vec![y])
}

/// Creates a model with 15 nodes (a div op creates 9 nodes)
/// Has a trace lenght of 2^s - 1, finishing with a virtual instruction
pub fn addsubmuldiv15_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);
    let out_dims = vec![1, 4];

    let x = b.input(out_dims.clone(), 4);
    let a = b.poly(PolyOp::Add, x, x, out_dims.clone(), 1);
    let a = b.poly(PolyOp::Add, a, x, out_dims.clone(), 2);
    let s = b.poly(PolyOp::Sub, a, x, out_dims.clone(), 1);
    let m = b.poly(PolyOp::Mult, a, s, out_dims.clone(), 1);
    let t = b.poly(PolyOp::Add, s, m, out_dims.clone(), 1);
    let y = b.div(2, t, out_dims.clone(), 1);

    b.take(vec![x.0], vec![y])
}

/// Creates a model with 16 nodes (a div op creates 9 nodes)
/// Has a trace lenght of 2^s, finishing with a virtual instruction
pub fn addsubmuldiv_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);
    let out_dims = vec![1, 4];

    let x = b.input(out_dims.clone(), 4);
    let a = b.poly(PolyOp::Add, x, x, out_dims.clone(), 1);
    let a = b.poly(PolyOp::Add, a, x, out_dims.clone(), 1);
    let a = b.poly(PolyOp::Add, a, x, out_dims.clone(), 2);
    let s = b.poly(PolyOp::Sub, a, x, out_dims.clone(), 1);
    let m = b.poly(PolyOp::Mult, a, s, out_dims.clone(), 1);
    let t = b.poly(PolyOp::Add, s, m, out_dims.clone(), 1);
    let y = b.div(2, t, out_dims.clone(), 1);

    b.take(vec![x.0], vec![y])
}

/// Creates a model with add, sub, mul, div, add operations
/// Has a trace length of 2^s, finishing with a virtual instruction  
pub fn addsubmuldivadd_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);
    let out_dims = vec![1, 4];

    let x = b.input(out_dims.clone(), 4);
    let a = b.poly(PolyOp::Add, x, x, out_dims.clone(), 1);
    let a = b.poly(PolyOp::Add, a, x, out_dims.clone(), 1);
    let a = b.poly(PolyOp::Add, a, x, out_dims.clone(), 2);
    let s = b.poly(PolyOp::Sub, a, x, out_dims.clone(), 1);
    let m = b.poly(PolyOp::Mult, a, s, out_dims.clone(), 1);
    let d = b.div(2, m, out_dims.clone(), 1);
    let y = b.poly(PolyOp::Add, d, s, out_dims.clone(), 1);

    b.take(vec![x.0], vec![y])
}

/// [(0, input, []), (1, add, [0, 0]), (2, sub, [1, 0]), (3, mul, [1, 2]), (4, add, [2, 3]), (5, div, [4]), (6, div, [5]), (7, output, [6])]
pub fn addsubmuldivdiv_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);
    let out_dims = vec![1, 4];

    let x = b.input(out_dims.clone(), 2);
    let a = b.poly(PolyOp::Add, x, x, out_dims.clone(), 2);
    let s = b.poly(PolyOp::Sub, a, x, out_dims.clone(), 1);
    let m = b.poly(PolyOp::Mult, a, s, out_dims.clone(), 1);
    let t = b.poly(PolyOp::Add, s, m, out_dims.clone(), 1);
    let d1 = b.div(2, t, out_dims.clone(), 1);
    let y = b.div(5, d1, out_dims.clone(), 1);

    b.take(vec![x.0], vec![y])
}

/// [(0, input, []), (1, add, [0, 0]), (2, sub, [1, 0]), (3, mul, [1, 2]), (4, add, [2, 3]), (5, output, [4])]
pub fn scalar_addsubmul_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);
    let dims = vec![1];

    let x = b.input(dims.clone(), 2);
    let a = b.poly(PolyOp::Add, x, x, dims.clone(), 2);
    let s = b.poly(PolyOp::Sub, a, x, dims.clone(), 1);
    let m = b.poly(PolyOp::Mult, a, s, dims.clone(), 1);
    let y = b.poly(PolyOp::Add, s, m, dims.clone(), 1);

    b.take(vec![x.0], vec![y])
}

pub fn relu_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);
    let dims = vec![1, 4];

    let x = b.input(dims.clone(), 2);
    let r = b.relu(x, dims.clone(), 1);

    b.take(vec![x.0], vec![r])
}

/// Implements a simple embedding-based sentiment analysis model:
/// 1. Looks up embeddings for input word indices
/// 2. Sums the embeddings and normalizes (divides by -0.46149117, which we round up to -0.5, which is multiplying by -2)
/// 3. Adds a bias term (-54)
/// 4. Returns positive sentiment if result >= 0
///
/// # Note all magic values here like -54, or the embedding tensors are from the pre-trained model in /models/sentiment_sum
pub fn sentiment0() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: Input node for word indices (shape [1, 5])
    let input_indices = b.input(vec![1, 5], 1);

    // Node 1: Create the embedding tensor (shape [14, 1]) (embeddings taken from /models/sentiment_sum)
    let mut embedding: Tensor<i32> = Tensor::new(
        Some(&[
            139, -200, -331, -42, -260, -290, -166, -171, -481, -294, 210, 291, 2, 328,
        ]),
        &[14, 1],
    )
    .unwrap();
    embedding.set_scale(SCALE);
    let embedding_const = b.const_tensor(embedding, vec![14, 1], 1);

    // Node 2: Gather (lookup embeddings based on indices)
    let gathered = b.gather(embedding_const, input_indices, 0, vec![1, 5, 1], 1);

    // Node 3: Reshape (flatten the gathered embeddings)
    let reshaped = b.reshape(gathered, vec![1, 5], vec![1, 5], 1);

    // Node 4: Sum the embeddings along axis 1
    let summed = b.sum(reshaped, vec![1], vec![1, 1], 1);
    /*
       Node 6: Divide by constant with floating-point value
       Node 6: Multiply by constant (reciprocal of divisor)
       let divided = b.div_f64(-0.46149117, summed, vec![1, 1], 1);
       -1 / -0.46149117 ≈ -2.167
    */
    let mul_const: Tensor<i32> = Tensor::new(Some(&[-2]), &[1, 1]).unwrap();
    let mul_wire = b.const_tensor(mul_const, vec![1, 1], 1);
    // Multiplication instead of division
    let multiplied = b.poly(PolyOp::Mult, summed, mul_wire, vec![1, 1], 1);

    // Node 7: Create the bias constant (-54)
    let mut bias: Tensor<i32> = Tensor::new(Some(&[-54]), &[1, 1]).unwrap();
    bias.set_scale(SCALE);
    let bias_const = b.const_tensor(bias, vec![1, 1], 1);

    // Node 8: Add the bias
    let added = b.poly(PolyOp::Add, multiplied, bias_const, vec![1, 1], 1);

    // Node 9: Create the zero constant
    let mut zero: Tensor<i32> = Tensor::new(Some(&[0]), &[1, 1]).unwrap();
    zero.set_scale(SCALE);
    let zero_const = b.const_tensor(zero, vec![1, 1], 1);

    // Node 10: Greater than or equal comparison
    let result = b.greater_equal(added, zero_const, vec![1, 1], 1);

    b.take(vec![input_indices.0], vec![result])
}

/// Implements a sentiment selection model with embeddings and conditional logic:
/// 1. Looks up embeddings for input word indices
/// 2. Filters embeddings based on a threshold (64)
/// 3. Uses conditional (IFF) to select embeddings or zeros
/// 4. Sums the selected embeddings
/// 5. Applies scaling (multiply by 261, then divide by 128)
/// 6. Adds bias (-142) and compares with zero
pub fn sentiment_select() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: Input indices (shape [1, 5])
    let input_indices = b.input(vec![1, 5], 1);

    // Node 1: Embedding tensor (shape [14, 1])
    let mut embedding: Tensor<i32> = Tensor::new(
        Some(&[
            0, 45, -137, -14, -6, 454, -81, -92, -32, 421, -106, -16, -146, 18,
        ]),
        &[14, 1],
    )
    .unwrap();
    embedding.set_scale(SCALE);
    let embedding_const = b.const_tensor_with_scale(embedding, SCALE, vec![14, 1], 1);

    // Node 2: Gather embeddings
    let gathered = b.gather(embedding_const, input_indices, 0, vec![1, 5, 1], 1);

    // Node 3: Threshold constant (64)
    let mut threshold: Tensor<i32> = Tensor::new(Some(&[64; 5]), &[1, 5, 1]).unwrap();
    threshold.set_scale(SCALE);
    let threshold_const = b.const_tensor_with_scale(threshold, SCALE, vec![1, 5, 1], 1);

    // Node 4: Greater than or equal comparison (embeddings >= threshold)
    let condition = b.greater_equal(gathered, threshold_const, vec![1, 5, 1], 1);

    // Node 5: Zero tensor for false case
    let mut zeros: Tensor<i32> = Tensor::new(Some(&[0, 0, 0, 0, 0]), &[1, 5, 1]).unwrap();
    zeros.set_scale(SCALE);
    let zeros_const = b.const_tensor_with_scale(zeros, SCALE, vec![1, 5, 1], 1);

    // Node 6: IFF (conditional selection)
    let selected = b.iff(condition, gathered, zeros_const, vec![1, 5, 1], 1);

    // Node 7: Sum the selected embeddings
    let summed = b.sum(selected, vec![1, 2], vec![1, 1, 1], 1);

    // Node 8: Reshape to [1, 1]
    let reshaped = b.reshape(summed, vec![1, 1], vec![1, 1], 1);

    // Node 9: Scale factor constant (261)
    let mut scale_factor: Tensor<i32> = Tensor::new(Some(&[261]), &[1, 1]).unwrap();
    scale_factor.set_scale(SCALE);
    let scale_const = b.const_tensor_with_scale(scale_factor, SCALE, vec![1, 1], 1);

    // Node 10: Multiply by scale factor (replacing RebaseScale)
    let multiplied = b.poly(PolyOp::Mult, reshaped, scale_const, vec![1, 1], 1);

    // Node 10.5: Divide by 128 (replacing the rebase scale division)
    let scaled = b.div(128i32, multiplied, vec![1, 1], 1);

    // Node 11: Bias constant (-142)
    let mut bias: Tensor<i32> = Tensor::new(Some(&[-142]), &[1, 1]).unwrap();
    bias.set_scale(SCALE);
    let bias_const = b.const_tensor_with_scale(bias, SCALE, vec![1, 1], 1);

    // Node 12: Add bias
    let added = b.poly(PolyOp::Add, scaled, bias_const, vec![1, 1], 1);

    // Node 13: Zero constant for final comparison
    let mut zero: Tensor<i32> = Tensor::new(Some(&[0]), &[1, 1]).unwrap();
    zero.set_scale(SCALE);
    let zero_const = b.const_tensor_with_scale(zero, SCALE, vec![1, 1], 1);

    // Node 14: Final greater than or equal comparison
    let result = b.greater_equal(added, zero_const, vec![1, 1], 1);

    b.take(vec![input_indices.0], vec![result])
}

/// Simple ArgMax model:
/// 1. Takes a 1D vector input
/// 2. Returns the index of the maximum element
pub fn argmax_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: Input vector (1D)
    let input = b.input(vec![5], 1); // Example: vector of length 5

    // Node 1: ArgMax operation along dimension 0
    let argmax_result = b.argmax(input, 0, vec![1], 1); // Returns a scalar index

    b.take(vec![input.0], vec![argmax_result])
}

/// Simple RebaseScale model:
/// 1. Takes a 1D vector input
/// 2. Applies a multiplication of input to itself
/// 3. Rescale the output
pub fn rebase_scale_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: First input vector (1D)
    let input = b.input(vec![5], 1); // Example: vector of length 5

    // Node 2: RebaseScale multiplication of both inputs
    let rebase_result = b.rebase_scale_mul(input, input, vec![5], 1);

    b.take(vec![input.0], vec![rebase_result])
}

pub fn greater_equal_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: First input vector (1D)
    let input_a = b.input(vec![1, 5], 1); // Example: vector of length 5

    // Node 1: Const tensor (1D)
    let mut const_tensor: Tensor<i32> = Tensor::new(Some(&[64, 0, 0, 0, 0]), &[1, 5]).unwrap();
    const_tensor.set_scale(SCALE);
    let input_b_const = b.const_tensor_with_scale(const_tensor, SCALE, vec![1, 5], 1);

    // Node 2: Greater than or equal comparison
    let gte_result = b.greater_equal(input_a, input_b_const, vec![1, 5], 1);

    b.take(vec![input_a.0], vec![gte_result])
}

/// Analog to onnx-tracer/models/multiclass0/network.onnx
///
/// Multiclass classification model that:
/// 1. Takes embedding tensor and input indices
/// 2. Gathers embeddings based on input indices  
/// 3. Sums the gathered embeddings
/// 4. Broadcasts the sum across a weight matrix
/// 5. Multiplies by weights (replacing RebaseScale with mul + div)
/// 6. Adds bias vector
/// 7. Applies ArgMax to find predicted class
/// 8. Reshapes output to scalar
pub fn multiclass0() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: Input indices (shape [1, 8])
    let input_indices = b.input(vec![1, 8], 1);

    // Node 1: Embedding matrix (shape [31, 1]) - Updated size and values
    let mut embedding: Tensor<i32> = Tensor::new(
        Some(&[
            -61, -287, -437, -294, -318, 345, 331, 330, -28, 337, 113, 111, 91, 103, -58, 85, 72,
            -463, -342, -345, -318, 355, 385, 376, 180, 125, 10, 143, 137, -45, 128,
        ]),
        &[31, 1],
    )
    .unwrap();
    embedding.set_scale(SCALE);
    let embedding_const = b.const_tensor_with_scale(embedding, SCALE, vec![31, 1], 1);

    // Node 2: Gather embeddings
    let gathered = b.gather(embedding_const, input_indices, 0, vec![1, 8, 1], 1);

    // Node 3: Sum the gathered embeddings
    let summed = b.sum(gathered, vec![1, 2], vec![1, 1, 1], 1);

    // Node 4: Reshape to [1, 1]
    let reshaped = b.reshape(summed, vec![1, 1], vec![1, 1], 1);

    // Node 5: Weight matrix constants (shape [1, 10]) - Updated values
    let mut weights: Tensor<i32> = Tensor::new(
        Some(&[388, 16, -93, 517, 208, 208, 208, 208, 208, 208]),
        &[1, 10],
    )
    .unwrap();
    weights.set_scale(SCALE);
    let weights_const = b.const_tensor_with_scale(weights, SCALE, vec![1, 10], 1);

    // Node 5.5: Broadcast the scalar [1, 1] to [1, 10] shape
    let scalar_broadcasted = b.broadcast(reshaped, vec![1, 10], vec![1, 10], 1);

    // Node 6: Multiply the broadcasted scalar by the weight vector (replacing RebaseScale)
    let multiplied = b.poly(
        PolyOp::Mult,
        weights_const,
        scalar_broadcasted,
        vec![1, 10],
        1,
    );

    // Node 6.5: Divide by 128 (replacing the rebase scale division)
    let scaled = b.div(128, multiplied, vec![1, 10], 1);

    // Node 7: Bias vector (shape [1, 10]) - Updated values
    let mut bias: Tensor<i32> = Tensor::new(
        Some(&[449, 421, -137, -95, -155, -155, -155, -155, -155, -155]),
        &[1, 10],
    )
    .unwrap();
    bias.set_scale(SCALE);
    let bias_const = b.const_tensor_with_scale(bias, SCALE, vec![1, 10], 1);

    // Node 8: Add bias
    let added = b.poly(PolyOp::Add, scaled, bias_const, vec![1, 10], 1);

    // Node 9: ArgMax along dimension 1 to find predicted class
    let argmax_result = b.argmax(added, 1, vec![1, 1], 1);

    // Node 10: Reshape to scalar output [1]
    let final_result = b.reshape(argmax_result, vec![1], vec![1], 1);

    b.take(vec![input_indices.0], vec![final_result])
}

/// Simple matrix multiplication model for testing ONNX MatMul semantics.
///
/// This model demonstrates ONNX matrix multiplication functionality:
/// 1. Takes an input tensor of shape [1, 4]
/// 2. Multiplies it with a constant weight matrix of shape [3, 4] (gets implicitly transposed)
/// 3. Outputs the result of shape [1, 3]
///
/// **ONNX MatMul Behavior**: The second matrix is implicitly transposed, so:
/// - Input: [1, 4]
/// - Weights: [3, 4] (stored as [3, 4], but acts like [4, 3] due to implicit transpose)
/// - Result: [1, 3]
///
/// The weight matrix contains simple values for easy verification:
/// ```ignore
/// weights = [[1, 4, 7, 10],    // First output neuron weights
///            [2, 5, 8, 11],    // Second output neuron weights  
///            [3, 6, 9, 12]]    // Third output neuron weights
/// ```
///
/// For input [a, b, c, d], the output will be:
/// [a*1 + b*4 + c*7 + d*10, a*2 + b*5 + c*8 + d*11, a*3 + b*6 + c*9 + d*12]
///
/// # Returns
/// A `Model` representing the matrix multiplication computation graph
pub fn simple_matmult_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: Input tensor (shape [1, 4])
    let input = b.input(vec![1, 4], 1);

    // Node 1: Weight matrix constant (shape [3, 4] - ONNX format with implicit transpose)
    let mut weights: Tensor<i32> = Tensor::new(
        Some(&[
            1, 4, 7, 10, // First output neuron weights
            2, 5, 8, 11, // Second output neuron weights
            3, 6, 9, 12, // Third output neuron weights
        ]),
        &[3, 4],
    )
    .unwrap();
    weights.set_scale(SCALE);
    let weight_matrix = b.const_tensor(weights, vec![3, 4], 1);

    // Node 2: Matrix multiplication: [1, 4] × [3, 4] → [1, 3] (using ONNX semantics)
    let result = b.matmult(input, weight_matrix, vec![1, 3], 1);

    b.take(vec![input.0], vec![result])
}

/// Tiny MLP (Multi-Layer Perceptron) head for testing feed-forward neural networks.
///
/// This model demonstrates a simple 2-layer feed-forward neural network:
/// 1. Takes an input tensor of shape [1, 4]
/// 2. First linear layer: [1, 4] → [1, 8] with ReLU activation
/// 3. Second linear layer: [1, 8] → [1, 2] with ReLU activation
/// 4. Outputs the final result of shape [1, 2]
///
/// Architecture:
/// ```ignore
/// Input [1, 4] → Linear → ReLU → Linear → ReLU → Output [1, 2]
///                [1, 8]         [1, 2]
/// ```
///
/// The weight matrices contain simple incremental values for easy verification:
/// - First layer weights: 8x4 matrix with values 1-32
/// - Second layer weights: 2x8 matrix with values 1-16
///
/// # Returns
/// A `Model` representing the tiny MLP computation graph
pub fn tiny_mlp_head_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: Input tensor (shape [1, 4])
    let input = b.input(vec![1, 4], 1);

    // Node 1: First layer weight matrix (shape [8, 4] - will be transposed in matmult)
    let mut weights1: Tensor<i32> = Tensor::new(
        Some(&[
            1, 2, 3, 4, // First hidden neuron weights
            5, 6, 7, 8, // Second hidden neuron weights
            9, 10, 11, 12, // Third hidden neuron weights
            13, 14, 15, 16, // Fourth hidden neuron weights
            17, 18, 19, 20, // Fifth hidden neuron weights
            21, 22, 23, 24, // Sixth hidden neuron weights
            25, 26, 27, 28, // Seventh hidden neuron weights
            29, 30, 31, 32, // Eighth hidden neuron weights
        ]),
        &[8, 4],
    )
    .unwrap();
    weights1.set_scale(SCALE);
    let weight_matrix1 = b.const_tensor(weights1, vec![8, 4], 1);

    // Node 2: First matrix multiplication: [1, 4] × [8, 4] → [1, 8]
    let hidden1 = b.matmult(input, weight_matrix1, vec![1, 8], 1);

    // Node 3: First ReLU activation: [1, 8] → [1, 8]
    let relu1 = b.relu(hidden1, vec![1, 8], 1);

    // Node 4: Second layer weight matrix (shape [2, 8] - will be transposed in matmult)
    let mut weights2: Tensor<i32> = Tensor::new(
        Some(&[
            1, 2, 3, 4, 5, 6, 7, 8, // First output neuron weights
            9, 10, 11, 12, 13, 14, 15, 16, // Second output neuron weights
        ]),
        &[2, 8],
    )
    .unwrap();
    weights2.set_scale(SCALE);
    let weight_matrix2 = b.const_tensor(weights2, vec![2, 8], 1);

    // Node 5: Second matrix multiplication: [1, 8] × [2, 8] → [1, 2]
    let hidden2 = b.matmult(relu1, weight_matrix2, vec![1, 2], 1);

    // Node 6: Second ReLU activation: [1, 2] → [1, 2]
    let output = b.relu(hidden2, vec![1, 2], 1);

    b.take(vec![input.0], vec![output])
}

/// Matrix multiplication model with non-power-of-two dimensions for testing padding.
///
/// This model demonstrates matrix multiplication with dimensions that are NOT powers of two:
/// 1. Takes an input tensor of shape [3, 5] (neither 3 nor 5 are powers of two)
/// 2. Multiplies it with a constant weight matrix of shape [7, 5] (7 is not a power of two)
/// 3. Outputs the result of shape [3, 7]
///
/// **Power-of-Two Padding Behavior** (when feature enabled):
/// - Input [3, 5] gets padded to [4, 8] (next powers of two)
/// - Weights [7, 5] get padded to [8, 8] (next powers of two)
/// - Computation is performed as [4, 8] × [8, 8] → [4, 8]
/// - Result is cropped back to [3, 7]
///
/// **ONNX MatMul Semantics**: Uses "mk,nk->mn" einsum pattern:
/// - Input: [3, 5] (m=3, k=5)
/// - Weights: [7, 5] (n=7, k=5)
/// - Result: [3, 7] (m=3, n=7)
///
/// The weight matrix contains simple incremental values for easy verification:
/// ```ignore
/// weights = [[1, 2, 3, 4, 5],      // First output neuron weights (row 0)
///            [6, 7, 8, 9, 10],     // Second output neuron weights (row 1)
///            [11, 12, 13, 14, 15], // Third output neuron weights (row 2)
///            [...],                // Rows 3-6 continue the pattern
///            [31, 32, 33, 34, 35]] // Seventh output neuron weights (row 6)
/// ```
///
/// # Returns
/// A `Model` representing the non-power-of-two matrix multiplication computation graph
pub fn non_power_of_two_matmult_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: Input tensor (shape [3, 5] - non-power-of-two dimensions)
    let input = b.input(vec![3, 5], 1);

    // Node 1: Weight matrix constant (shape [7, 5] - non-power-of-two dimensions)
    // Using "mk,nk->mn" semantics where input is [m=3, k=5] and weights are [n=7, k=5]
    let mut weights: Tensor<i32> = Tensor::new(
        Some(&[
            1, 2, 3, 4, 5, // Row 0 (output neuron 0 weights)
            6, 7, 8, 9, 10, // Row 1 (output neuron 1 weights)
            11, 12, 13, 14, 15, // Row 2 (output neuron 2 weights)
            16, 17, 18, 19, 20, // Row 3 (output neuron 3 weights)
            21, 22, 23, 24, 25, // Row 4 (output neuron 4 weights)
            26, 27, 28, 29, 30, // Row 5 (output neuron 5 weights)
            31, 32, 33, 34, 35, // Row 6 (output neuron 6 weights)
        ]),
        &[7, 5],
    )
    .unwrap();
    weights.set_scale(SCALE);
    let weight_matrix = b.const_tensor(weights, vec![7, 5], 1);

    // Node 2: Matrix multiplication: [3, 5] × [7, 5] → [3, 7] (using mk,nk->mn pattern)
    // This will trigger power-of-two padding when the feature is enabled:
    // - [3, 5] → [4, 8] (padded)
    // - [7, 5] → [8, 8] (padded)
    // - Compute [4, 8] × [8, 8] → [4, 8]
    // - Crop to [3, 7] (final result)
    let result = b.matmult(input, weight_matrix, vec![3, 7], 1);

    b.take(vec![input.0], vec![result])
}

/// Simple linear head model using only basic operations: input, matmult, add, mul, const.
///
/// This model demonstrates a minimal linear transformation:
/// 1. Takes an input tensor of shape [1, 4]
/// 2. Multiplies it with a weight matrix [2, 4] → [1, 2]
/// 3. Adds a bias vector [1, 2]
/// 4. Multiplies by a scale factor [1, 2]
/// 5. Outputs the final result of shape [1, 2]
///
/// Operations used: Input → MatMult → Add → Mult → Output
/// Total nodes: 5 (input, 3 constants, matmult, add, mult)
///
/// # Returns
/// A `Model` representing the simple linear head computation graph
pub fn simple_linear_head_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: Input tensor (shape [1, 4])
    let input = b.input(vec![1, 4], 1);

    // Node 1: Weight matrix constant (shape [2, 4])
    let mut weights: Tensor<i32> = Tensor::new(
        Some(&[
            1, 2, 3, 4, // First output neuron weights
            5, 6, 7, 8, // Second output neuron weights
        ]),
        &[2, 4],
    )
    .unwrap();
    weights.set_scale(SCALE);
    let weight_matrix = b.const_tensor(weights, vec![2, 4], 1);

    // Node 2: Matrix multiplication: [1, 4] × [2, 4] → [1, 2]
    let linear_out = b.matmult(input, weight_matrix, vec![1, 2], 1);

    // Node 3: Bias vector constant (shape [1, 2])
    let mut bias: Tensor<i32> = Tensor::new(Some(&[10, 20]), &[1, 2]).unwrap();
    bias.set_scale(SCALE);
    let bias_vector = b.const_tensor(bias, vec![1, 2], 1);

    // Node 4: Add bias: [1, 2] + [1, 2] → [1, 2]
    let biased_out = b.poly(PolyOp::Add, linear_out, bias_vector, vec![1, 2], 1);

    // Node 5: Scale factor constant (shape [1, 2])
    let mut scale: Tensor<i32> = Tensor::new(Some(&[2, 3]), &[1, 2]).unwrap();
    scale.set_scale(SCALE);
    let scale_vector = b.const_tensor(scale, vec![1, 2], 1);

    // Node 6: Multiply by scale: [1, 2] * [1, 2] → [1, 2]
    let final_out = b.poly(PolyOp::Mult, biased_out, scale_vector, vec![1, 2], 1);

    b.take(vec![input.0], vec![final_out])
}

/// Simple linear head model with 4x4 matrix multiplication using only basic operations.
///
/// This model demonstrates a minimal linear transformation with square matrices:
/// 1. Takes an input tensor of shape [1, 4]
/// 2. Multiplies it with a 4x4 weight matrix → [1, 4]
/// 3. Adds a bias vector [1, 4]
/// 4. Multiplies by a scale factor [1, 4]
/// 5. Outputs the final result of shape [1, 4]
///
/// Operations used: Input → MatMult(4x4) → Add → Mult → Output
/// Total nodes: 7 (input, 3 constants, matmult, add, mult)
///
/// # Returns
/// A `Model` representing the 4x4 matrix linear head computation graph
pub fn simple_linear_head_4x4_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: Input tensor (shape [1, 4])
    let input = b.input(vec![1, 4], 1);

    // Node 1: 4x4 Weight matrix constant (shape [4, 4])
    let mut weights: Tensor<i32> = Tensor::new(
        Some(&[
            1, 2, 3, 4, // First output neuron weights
            5, 6, 7, 8, // Second output neuron weights
            9, 10, 11, 12, // Third output neuron weights
            13, 14, 15, 16, // Fourth output neuron weights
        ]),
        &[4, 4],
    )
    .unwrap();
    weights.set_scale(SCALE);
    let weight_matrix = b.const_tensor(weights, vec![4, 4], 1);

    // Node 2: Matrix multiplication: [1, 4] × [4, 4] → [1, 4]
    let linear_out = b.matmult(input, weight_matrix, vec![1, 4], 1);

    // Node 3: Bias vector constant (shape [1, 4])
    let mut bias: Tensor<i32> = Tensor::new(Some(&[10, 20, 30, 40]), &[1, 4]).unwrap();
    bias.set_scale(SCALE);
    let bias_vector = b.const_tensor(bias, vec![1, 4], 1);

    // Node 4: Add bias: [1, 4] + [1, 4] → [1, 4]
    let biased_out = b.poly(PolyOp::Add, linear_out, bias_vector, vec![1, 4], 1);

    // Node 5: Scale factor constant (shape [1, 4])
    let mut scale: Tensor<i32> = Tensor::new(Some(&[2, 3, 4, 5]), &[1, 4]).unwrap();
    scale.set_scale(SCALE);
    let scale_vector = b.const_tensor(scale, vec![1, 4], 1);

    // Node 6: Multiply by scale: [1, 4] * [1, 4] → [1, 4]
    let final_out = b.poly(PolyOp::Mult, biased_out, scale_vector, vec![1, 4], 1);

    b.take(vec![input.0], vec![final_out])
}

/// Minimal matrix multiplication model: input, const, matmult only.
///
/// This model demonstrates the simplest possible linear transformation:
/// 1. Takes an input tensor of shape [1, 4]
/// 2. Multiplies it with a weight matrix [2, 4] → [1, 2]
/// 3. Outputs the final result of shape [1, 2]
///
/// Operations used: Input → MatMult → Output
/// Total nodes: 3 (input, const, matmult)
///
/// # Returns
/// A `Model` representing the minimal matrix multiplication computation graph
pub fn minimal_matmult_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: Input tensor (shape [1, 4])
    let input = b.input(vec![1, 4], 1);

    // Node 1: Weight matrix constant (shape [2, 4])
    let mut weights: Tensor<i32> = Tensor::new(
        Some(&[
            1, 2, 3, 4, // First output neuron weights
            5, 6, 7, 8, // Second output neuron weights
        ]),
        &[2, 4],
    )
    .unwrap();
    weights.set_scale(SCALE);
    let weight_matrix = b.const_tensor(weights, vec![2, 4], 1);

    // Node 2: Matrix multiplication: [1, 4] × [2, 4] → [1, 2]
    let result = b.matmult(input, weight_matrix, vec![1, 2], 1);

    b.take(vec![input.0], vec![result])
}

/// Minimal matrix multiplication model with non-power-of-two dimensions: input, const, matmult only.
///
/// This model demonstrates the simplest possible linear transformation with non-power-of-two dimensions:
/// 1. Takes an input tensor of shape [1, 5] (5 is not a power of two)
/// 2. Multiplies it with a weight matrix [3, 5] → [1, 3] (3 is not a power of two)
/// 3. Outputs the final result of shape [1, 3]
///
/// **Power-of-Two Padding Behavior** (when feature enabled):
/// - Input [1, 5] gets padded to [1, 8] (next power of two for second dimension)
/// - Weights [3, 5] get padded to [4, 8] (next powers of two)
/// - Computation is performed as [1, 8] × [4, 8] → [1, 4]
/// - Result is cropped back to [1, 3]
///
/// Neither input dimension (5) nor output dimension (3) are powers of two
/// Operations used: Input → MatMult → Output
/// Total nodes: 3 (input, const, matmult)
///
/// # Returns
/// A `Model` representing the minimal matrix multiplication computation graph with non-power-of-two dimensions
pub fn minimal_matmult_non_power_of_two_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: Input tensor (shape [1, 5] - non-power-of-two)
    let input = b.input(vec![1, 5], 1);

    // Node 1: Weight matrix constant (shape [3, 5] - non-power-of-two dimensions)
    let mut weights: Tensor<i32> = Tensor::new(
        Some(&[
            1, 2, 3, 4, 5, // First output neuron weights
            6, 7, 8, 9, 10, // Second output neuron weights
            11, 12, 13, 14, 15, // Third output neuron weights
        ]),
        &[3, 5],
    )
    .unwrap();
    weights.set_scale(SCALE);
    let weight_matrix = b.const_tensor(weights, vec![3, 5], 1);

    // Node 2: Matrix multiplication: [1, 5] × [3, 5] → [1, 3]
    let result = b.matmult(input, weight_matrix, vec![1, 3], 1);

    b.take(vec![input.0], vec![result])
}

/// Dual matrix multiplication model with power-of-two dimensions and non-overflowing weights.
///
/// This model demonstrates a 2-layer neural network with power-of-two dimensions:
/// 1. Takes an input tensor of shape [1, 4] (4 = 2^2)
/// 2. First matmult: [1, 4] × [4, 4] → [1, 4] with weights (1-16)
/// 3. Second matmult: [1, 4] × [2, 4] → [1, 2] with weights (1-8)
/// 4. Outputs the final result of shape [1, 2] (2 = 2^1)
///
/// All dimensions are powers of two: 4=2^2, 2=2^1, 1=2^0
/// Weight values are chosen to avoid overflow while being reasonably sized
/// Operations used: Input → MatMult → MatMult → Output
/// Total nodes: 4 (input, 2 consts, 2 matmults)
///
/// # Returns
/// A `Model` representing the dual matrix multiplication computation graph
pub fn dual_matmult_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: Input tensor (shape [1, 4])
    let input = b.input(vec![1, 4], 1);

    // Node 1: First weight matrix constant (shape [4, 4] - square matrix)
    let mut weights1: Tensor<i32> = Tensor::new(
        Some(&[
            1, 2, 3, 4, // First output neuron weights
            5, 6, 7, 8, // Second output neuron weights
            9, 10, 11, 12, // Third output neuron weights
            13, 14, 15, 16, // Fourth output neuron weights
        ]),
        &[4, 4],
    )
    .unwrap();
    weights1.set_scale(SCALE);
    let weight_matrix1 = b.const_tensor(weights1, vec![4, 4], 1);

    // Node 2: First matrix multiplication: [1, 4] × [4, 4] → [1, 4]
    let first_result = b.matmult(input, weight_matrix1, vec![1, 4], 1);

    // Node 3: Second weight matrix constant (shape [2, 4])
    let mut weights2: Tensor<i32> = Tensor::new(
        Some(&[
            1, 2, 3, 4, // First output neuron weights
            5, 6, 7, 8, // Second output neuron weights
        ]),
        &[2, 4],
    )
    .unwrap();
    weights2.set_scale(SCALE);
    let weight_matrix2 = b.const_tensor(weights2, vec![2, 4], 1);

    // Node 4: Second matrix multiplication: [1, 4] × [2, 4] → [1, 2]
    let final_result = b.matmult(first_result, weight_matrix2, vec![1, 2], 1);

    b.take(vec![input.0], vec![final_result])
}

/// Dual matrix multiplication model with non-power-of-two dimensions for testing padding.
///
/// This model demonstrates a 2-layer neural network with non-power-of-two dimensions:
/// 1. Takes an input tensor of shape [1, 5] (5 is not a power of two)
/// 2. First matmult: [1, 5] × [6, 5] → [1, 6] with weights (1-30)
/// 3. Second matmult: [1, 6] × [3, 6] → [1, 3] with weights (1-18)
/// 4. Outputs the final result of shape [1, 3] (3 is not a power of two)
///
/// **Power-of-Two Padding Behavior** (when feature enabled):
/// - Input [1, 5] gets padded to [1, 8] (next power of two for second dimension)
/// - First weights [6, 5] get padded to [8, 8] (next powers of two)
/// - First result [1, 6] gets padded to [1, 8] (next power of two)
/// - Second weights [3, 6] get padded to [4, 8] (next powers of two)
/// - Final result is cropped back to [1, 3]
///
/// None of the dimensions are powers of two: 5, 6, 3
/// Weight values are chosen to avoid overflow while being reasonably sized
/// Operations used: Input → MatMult → MatMult → Output
/// Total nodes: 4 (input, 2 consts, 2 matmults)
///
/// # Returns
/// A `Model` representing the dual matrix multiplication computation graph with non-power-of-two dimensions
pub fn dual_matmult_non_power_of_two_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: Input tensor (shape [1, 5] - non-power-of-two)
    let input = b.input(vec![1, 5], 1);

    // Node 1: First weight matrix constant (shape [6, 5] - non-power-of-two dimensions)
    let mut weights1: Tensor<i32> = Tensor::new(
        Some(&[
            1, 2, 3, 4, 5, // First output neuron weights
            6, 7, 8, 9, 10, // Second output neuron weights
            11, 12, 13, 14, 15, // Third output neuron weights
            16, 17, 18, 19, 20, // Fourth output neuron weights
            21, 22, 23, 24, 25, // Fifth output neuron weights
            26, 27, 28, 29, 30, // Sixth output neuron weights
        ]),
        &[6, 5],
    )
    .unwrap();
    weights1.set_scale(SCALE);
    let weight_matrix1 = b.const_tensor(weights1, vec![6, 5], 1);

    // Node 2: First matrix multiplication: [1, 5] × [6, 5] → [1, 6]
    let first_result = b.matmult(input, weight_matrix1, vec![1, 6], 1);

    // Node 3: Second weight matrix constant (shape [3, 6] - non-power-of-two dimensions)
    let mut weights2: Tensor<i32> = Tensor::new(
        Some(&[
            1, 2, 3, 4, 5, 6, // First output neuron weights
            7, 8, 9, 10, 11, 12, // Second output neuron weights
            13, 14, 15, 16, 17, 18, // Third output neuron weights
        ]),
        &[3, 6],
    )
    .unwrap();
    weights2.set_scale(SCALE);
    let weight_matrix2 = b.const_tensor(weights2, vec![3, 6], 1);

    // Node 4: Second matrix multiplication: [1, 6] × [3, 6] → [1, 3]
    let final_result = b.matmult(first_result, weight_matrix2, vec![1, 3], 1);

    b.take(vec![input.0], vec![final_result])
}

/// Triple matrix multiplication model with power-of-two dimensions and non-overflowing weights.
///
/// This model demonstrates a 3-layer neural network with power-of-two dimensions:
/// 1. Takes an input tensor of shape [1, 8] (8 = 2^3)
/// 2. First matmult: [1, 8] × [8, 8] → [1, 8] with weights (1-64)
/// 3. Second matmult: [1, 8] × [4, 8] → [1, 4] with weights (1-32)
/// 4. Third matmult: [1, 4] × [2, 4] → [1, 2] with weights (1-8)
/// 5. Outputs the final result of shape [1, 2] (2 = 2^1)
///
/// All dimensions are powers of two: 8=2^3, 4=2^2, 2=2^1, 1=2^0
/// Weight values are chosen to avoid overflow while being reasonably sized
/// Operations used: Input → MatMult → MatMult → MatMult → Output
/// Total nodes: 6 (input, 3 consts, 3 matmults)
///
/// # Returns
/// A `Model` representing the triple matrix multiplication computation graph
pub fn triple_matmult_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: Input tensor (shape [1, 8])
    let input = b.input(vec![1, 8], 1);

    // Node 1: First weight matrix constant (shape [8, 8] - square matrix)
    let mut weights1: Tensor<i32> = Tensor::new(
        Some(&[
            1, 2, 3, 4, 5, 6, 7, 8, // First output neuron weights
            9, 10, 11, 12, 13, 14, 15, 16, // Second output neuron weights
            17, 18, 19, 20, 21, 22, 23, 24, // Third output neuron weights
            25, 26, 27, 28, 29, 30, 31, 32, // Fourth output neuron weights
            33, 34, 35, 36, 37, 38, 39, 40, // Fifth output neuron weights
            41, 42, 43, 44, 45, 46, 47, 48, // Sixth output neuron weights
            49, 50, 51, 52, 53, 54, 55, 56, // Seventh output neuron weights
            57, 58, 59, 60, 61, 62, 63, 64, // Eighth output neuron weights
        ]),
        &[8, 8],
    )
    .unwrap();
    weights1.set_scale(SCALE);
    let weight_matrix1 = b.const_tensor(weights1, vec![8, 8], 1);

    // Node 2: First matrix multiplication: [1, 8] × [8, 8] → [1, 8]
    let first_result = b.matmult(input, weight_matrix1, vec![1, 8], 1);

    // Node 3: Second weight matrix constant (shape [4, 8])
    let mut weights2: Tensor<i32> = Tensor::new(
        Some(&[
            1, 2, 3, 4, 5, 6, 7, 8, // First output neuron weights
            9, 10, 11, 12, 13, 14, 15, 16, // Second output neuron weights
            17, 18, 19, 20, 21, 22, 23, 24, // Third output neuron weights
            25, 26, 27, 28, 29, 30, 31, 32, // Fourth output neuron weights
        ]),
        &[4, 8],
    )
    .unwrap();
    weights2.set_scale(SCALE);
    let weight_matrix2 = b.const_tensor(weights2, vec![4, 8], 1);

    // Node 4: Second matrix multiplication: [1, 8] × [4, 8] → [1, 4]
    let second_result = b.matmult(first_result, weight_matrix2, vec![1, 4], 1);

    // Node 5: Third weight matrix constant (shape [2, 4])
    let mut weights3: Tensor<i32> = Tensor::new(
        Some(&[
            1, 2, 3, 4, // First output neuron weights
            5, 6, 7, 8, // Second output neuron weights
        ]),
        &[2, 4],
    )
    .unwrap();
    weights3.set_scale(SCALE);
    let weight_matrix3 = b.const_tensor(weights3, vec![2, 4], 1);

    // Node 6: Third matrix multiplication: [1, 4] × [2, 4] → [1, 2]
    let final_result = b.matmult(second_result, weight_matrix3, vec![1, 2], 1);

    b.take(vec![input.0], vec![final_result])
}

/// Simple MLP Small model that recreates the exact bytecode structure.
///
/// This model matches the bytecode from the test case:
/// 1. Takes an input tensor of shape [1, 4] with scale 7
/// 2. Matrix multiplies with a 4x4 weight matrix (Einsum "mk,nk->mn", scale 14)
/// 3. Divides by 128 to rescale back to scale 7
/// 4. Adds a bias vector [1, 4] with scale 7
/// 5. Applies ReLU activation with scale 7
/// 6. Outputs the result of shape [1, 4]
///
/// The exact values match those seen in the bytecode:
/// - Weight matrix: [16, -11, 8, 29, -14, 53, 3, -3, 8, -26, 100, 15, -15, 1, -17, 47]
/// - Bias vector: [82, 71, 30, 76]
///
/// Operations sequence: Input → MatMult → Div → Add → ReLU → Output
///
/// # Returns
/// A `Model` representing the simple MLP small computation graph
pub fn simple_mlp_small_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: Input tensor (shape [1, 4], scale 7)
    let input = b.input(vec![1, 4], 1);

    // Node 1: Weight matrix constant (shape [4, 4], scale 7)
    // Exact values from the bytecode
    let mut weights: Tensor<i32> = Tensor::new(
        Some(&[
            16, -11, 8, 29, // First row
            -14, 53, 3, -3, // Second row
            8, -26, 100, 15, // Third row
            -15, 1, -17, 47, // Fourth row
        ]),
        &[4, 4],
    )
    .unwrap();
    weights.set_scale(SCALE);
    let weight_matrix = b.const_tensor(weights, vec![4, 4], 1);

    // Node 2: Matrix multiplication (Einsum "mk,nk->mn"): [1, 4] × [4, 4] → [1, 4]
    // This creates scale 14 (7 + 7)
    let matmul_result = b.matmult(input, weight_matrix, vec![1, 4], 1);

    // Node 3: Division by 128 to rescale from 14 back to 7
    let rescaled = b.div(128, matmul_result, vec![1, 4], 1);

    // Node 4: Bias vector constant (shape [1, 4], scale 7)
    // Exact values from the bytecode
    let mut bias: Tensor<i32> = Tensor::new(Some(&[82, 71, 30, 76]), &[1, 4]).unwrap();
    bias.set_scale(SCALE);
    let bias_vector = b.const_tensor(bias, vec![1, 4], 1);

    // Node 5: Add bias: [1, 4] + [1, 4] → [1, 4] (scale 7)
    let biased = b.poly(PolyOp::Add, rescaled, bias_vector, vec![1, 4], 1);

    // Node 7: ReLU activation: [1, 4] → [1, 4] (scale 7)
    let output = b.relu(biased, vec![1, 4], 1);

    b.take(vec![input.0], vec![output])
}

/// Matrix multiplication model with RebaseScale wrapper for testing ONNX binary compilation scenarios.
///
/// This model demonstrates matrix multiplication wrapped in RebaseScale, which is common
/// when compiling ONNX models to binary format. The RebaseScale handles quantization scaling:
/// 1. Takes an input tensor of shape [3, 5] (non-power-of-two dimensions)
/// 2. Multiplies it with a constant weight matrix of shape [7, 5] (non-power-of-two dimensions)
/// 3. Wraps the MatMul directly in RebaseScale for quantization handling
/// 4. Outputs the result of shape [3, 7]
///
/// **RebaseScale Behavior**:
/// - Performs MatMul operation with RebaseScale wrapper
/// - Applies scaling: result = (matmul_result * multiplier) / divisor automatically
/// - Handles quantization effects from ONNX binary compilation
/// - No additional division step needed (handled internally by RebaseScale)
///
/// **Power-of-Two Padding Behavior** (when feature enabled):
/// - Input [3, 5] gets padded to [4, 8] (next powers of two)
/// - Weights [7, 5] get padded to [8, 8] (next powers of two)
/// - Computation is performed as [4, 8] × [8, 8] → [4, 8]
/// - Result is cropped back to [3, 7], then RebaseScale is applied
///
/// **ONNX MatMul Semantics**: Uses "mk,nk->mn" einsum pattern:
/// - Input: [3, 5] (m=3, k=5)
/// - Weights: [7, 5] (n=7, k=5)
/// - MatMul Result: [3, 7] (m=3, n=7)
/// - RebaseScale Applied: [3, 7] (scaled values)
///
/// The weight matrix contains simple incremental values for easy verification:
/// ```ignore
/// weights = [[1, 2, 3, 4, 5],      // First output neuron weights (row 0)
///            [6, 7, 8, 9, 10],     // Second output neuron weights (row 1)
///            [11, 12, 13, 14, 15], // Third output neuron weights (row 2)
///            [...],                // Rows 3-6 continue the pattern
///            [31, 32, 33, 34, 35]] // Seventh output neuron weights (row 6)
/// ```
///
/// # Returns
/// A `Model` representing the RebaseScale-wrapped matrix multiplication computation graph
pub fn non_power_of_two_matmult_rebase_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);

    // Node 0: Input tensor (shape [3, 5] - non-power-of-two dimensions)
    let input = b.input(vec![3, 5], 1);

    // Node 1: Weight matrix constant (shape [7, 5] - non-power-of-two dimensions)
    // Using "mk,nk->mn" semantics where input is [m=3, k=5] and weights are [n=7, k=5]
    let mut weights: Tensor<i32> = Tensor::new(
        Some(&[
            1, 2, 3, 4, 5, // Row 0 (output neuron 0 weights)
            6, 7, 8, 9, 10, // Row 1 (output neuron 1 weights)
            11, 12, 13, 14, 15, // Row 2 (output neuron 2 weights)
            16, 17, 18, 19, 20, // Row 3 (output neuron 3 weights)
            21, 22, 23, 24, 25, // Row 4 (output neuron 4 weights)
            26, 27, 28, 29, 30, // Row 5 (output neuron 5 weights)
            31, 32, 33, 34, 35, // Row 6 (output neuron 6 weights)
        ]),
        &[7, 5],
    )
    .unwrap();
    weights.set_scale(SCALE);
    let weight_matrix = b.const_tensor(weights, vec![7, 5], 1);

    // Node 2: RebaseScale-wrapped matrix multiplication: [3, 5] × [7, 5] → [3, 7]
    // This wraps the MatMul directly in RebaseScale as commonly done in ONNX compilation
    // The RebaseScale handles both the matrix multiplication and quantization scaling internally
    let result = b.rebase_scale_matmult(input, weight_matrix, vec![3, 7], 1);

    b.take(vec![input.0], vec![result])
}
