//! Node representation and helpers used by the ONNX tracer.
//! A `ComputationNode` models a single operation in the graph.
use crate::ops::Operator;
use serde::{Deserialize, Serialize};

/// Node-specific handler functions and utilities.
pub mod handlers;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Represents a single computation node in the computation graph.
///
/// Nodes carry their operator, input dependencies (by index), and the
/// output tensor dimensions produced by the operator.
pub struct ComputationNode {
    /// Stable node index within the graph (0-based).
    pub idx: usize,
    /// The operation executed by this node.
    pub operator: Operator,
    /// Indices of upstream nodes whose outputs feed this node.
    pub inputs: Vec<usize>,
    /// Dimensions (shape) of the tensor produced by this node.
    pub output_dims: Vec<usize>,
}

impl Default for ComputationNode {
    /// Creates a default no-op node with no inputs and no output dims.
    fn default() -> Self {
        Self {
            idx: 0,
            operator: Operator::Noop(Default::default()),
            inputs: vec![],
            output_dims: vec![],
        }
    }
}

impl ComputationNode {
    /// Construct a new computation node.
    ///
    /// - `idx`: Stable index of the node within the graph.
    /// - `operator`: The operator this node performs.
    /// - `inputs`: Indices of nodes providing inputs to this node.
    /// - `output_dims`: Shape of the output tensor produced.
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
        }
    }

    pub fn num_output_elements(&self) -> usize {
        self.output_dims.iter().product()
    }

    pub fn is_scalar(&self) -> bool {
        self.num_output_elements() == 1
    }
}
