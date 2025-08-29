//! Model builder utilities for creating ONNX computation graphs.
//!
//! This module provides a `ModelBuilder` struct that simplifies the creation of
//! neural network models for testing and development. It offers high-level methods
//! for common operations like matrix multiplication, element-wise operations, and more.
//!
//! # Example: Simple Matrix Multiplication
//!
//! ```rust,ignore
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
/// ```
/// weights = [[1, 4, 7, 10],    // First output neuron weights
///            [2, 5, 8, 11],    // Second output neuron weights  
///            [3, 6, 9, 12]]    // Third output neuron weights
/// ```
use crate::{
    circuit::ops::{hybrid::HybridOp, poly::PolyOp},
    graph::{
        model::Model,
        node::{RebaseScale, SupportedOp},
        utilities::{
            create_const_node, create_div_node, create_iff_node, create_input_node,
            create_matmul_node, create_node, create_polyop_node,
        },
    },
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

    // TODO(AntoineF4C5): generic quantization
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

    // TODO(AntoineF4C5): generic quantization
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

    // TODO(AntoineF4C5): generic quantization
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
}

/* ********************** Testing Model's ********************** */

/// [(0, input, []), (1, add, [0, 0]), (2, sub, [1, 0]), (3, mul, [1, 2]), (4, add, [2, 3]), (5, output, [4])]
pub fn custom_addsubmul_model() -> Model {
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
pub fn custom_addsubmulconst_model() -> Model {
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

/// [(0, input, []), (1, add, [0, 0]), (2, sub, [1, 0]), (3, mul, [1, 2]), (4, add, [2, 3]), (5, div, [4]), (6, output, [5])]
pub fn custom_addsubmuldiv_model() -> Model {
    const SCALE: i32 = 7;
    let mut b = ModelBuilder::new(SCALE);
    let out_dims = vec![1, 4];

    let x = b.input(out_dims.clone(), 2);
    let a = b.poly(PolyOp::Add, x, x, out_dims.clone(), 2);
    let s = b.poly(PolyOp::Sub, a, x, out_dims.clone(), 1);
    let m = b.poly(PolyOp::Mult, a, s, out_dims.clone(), 1);
    let t = b.poly(PolyOp::Add, s, m, out_dims.clone(), 1);
    let y = b.div(2, t, out_dims.clone(), 1);

    b.take(vec![x.0], vec![y])
}

/// [(0, input, []), (1, add, [0, 0]), (2, sub, [1, 0]), (3, mul, [1, 2]), (4, add, [2, 3]), (5, div, [4]), (6, div, [5]), (7, output, [6])]
pub fn custom_addsubmuldivdiv_model() -> Model {
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

    // Node 0: Create the embedding tensor (shape [14, 1]) (embeddings taken from /models/sentiment_sum)
    let mut embedding: Tensor<i32> = Tensor::new(
        Some(&[
            139, -200, -331, -42, -260, -290, -166, -171, -481, -294, 210, 291, 2, 328,
        ]),
        &[14, 1],
    )
    .unwrap();
    embedding.set_scale(SCALE);
    let embedding_const = b.const_tensor(embedding, vec![14, 1], 1);

    // Node 1: Input node for word indices (shape [1, 5])
    let input_indices = b.input(vec![1, 5], 1);

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

    // Node 0: Embedding tensor (shape [14, 1])
    let mut embedding: Tensor<i32> = Tensor::new(
        Some(&[
            0, 45, -137, -14, -6, 454, -81, -92, -32, 421, -106, -16, -146, 18,
        ]),
        &[14, 1],
    )
    .unwrap();
    embedding.set_scale(SCALE);
    let embedding_const = b.const_tensor_with_scale(embedding, SCALE, vec![14, 1], 1);

    // Node 1: Input indices (shape [1, 5])
    let input_indices = b.input(vec![1, 5], 1);

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

    // Node 0: Embedding matrix (shape [31, 1]) - Updated size and values
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

    // Node 1: Input indices (shape [1, 8])
    let input_indices = b.input(vec![1, 8], 1);

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
/// ```
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
