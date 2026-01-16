//! Fluent builder for constructing handler decompositions.
//!
//! This module provides a declarative, DRY approach to building computation node
//! decompositions in operator handlers. It automatically handles:
//! - Broadcasting nodes when input shapes don't match output shapes
//! - Rebase nodes for operators that change the fixed-point scale
//!
//! # Example
//!
//! ```ignore
//! // Simple operator (no rebase needed)
//! fn handle_add(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
//!     HandlerBuilder::new(hctx)
//!         .with_broadcast()
//!         .simple_op(Operator::Add(Default::default()))
//!         .build()
//! }
//!
//! // Operator with rebase (like Mul, Square)
//! fn handle_mul(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
//!     HandlerBuilder::new(hctx)
//!         .with_broadcast()
//!         .simple_op(Operator::Mul(Default::default()))
//!         .with_auto_rebase()
//!         .build()
//! }
//! ```

use crate::{
    node::{ComputationNode, handlers::HandlerContext},
    ops::{Constant, Operator},
    tensor::Tensor,
};

use super::parser::{DecompositionBuilder, GraphParser};

/// Fluent builder for creating handler decompositions.
///
/// This builder encapsulates the common patterns found in operator handlers:
/// 1. Optional broadcast node insertion
/// 2. Main operator node(s)
/// 3. Optional rebase (division by scale factor)
pub struct HandlerBuilder<'a, 'b> {
    hctx: &'a mut HandlerContext<'b>,
    broadcast_nodes: Vec<ComputationNode>,
    stages: Vec<Stage>,
    auto_rebase: bool,
    custom_rebase_factor: Option<i32>,
}

/// A stage in the computation pipeline
#[allow(dead_code)]
enum Stage {
    /// A simple operator that uses the current inputs
    SimpleOp { operator: Operator },
    /// A pipeline stage that transforms the previous output
    PipeOp { operator: Operator },
    /// A pipeline stage with custom output dimensions
    PipeOpWithDims {
        operator: Operator,
        output_dims: Vec<usize>,
    },
    /// A cionstant node
    Constant { tensor: Tensor<i32> },
    /// Division by the previous constant (for rebase)
    DivByPrevious,
    /// Division by a constant value
    DivByConstant { value: i32 },
}

impl<'a, 'b> HandlerBuilder<'a, 'b> {
    /// Creates a new handler builder.
    ///
    /// # Arguments
    /// * `hctx` - The handler context containing all necessary information
    pub fn new(hctx: &'a mut HandlerContext<'b>) -> Self {
        Self {
            hctx,
            broadcast_nodes: vec![],
            stages: vec![],
            auto_rebase: false,
            custom_rebase_factor: None,
        }
    }

    /// Automatically inserts broadcast nodes if input shapes don't match output shape.
    ///
    /// This should typically be called first, before adding any operators.
    pub fn with_broadcast(mut self) -> Self {
        self.broadcast_nodes = GraphParser::insert_broadcast_nodes(self.hctx);
        self
    }

    /// Adds a simple operator that uses the handler's input indices directly.
    ///
    /// This is the main operator for simple handlers like Add, Sub, etc.
    pub fn simple_op(mut self, operator: Operator) -> Self {
        self.stages.push(Stage::SimpleOp { operator });
        self
    }

    /// Adds a pipeline operator that takes the previous stage's output as input.
    ///
    /// Use this for building multi-stage decompositions.
    pub fn pipe(mut self, operator: Operator) -> Self {
        self.stages.push(Stage::PipeOp { operator });
        self
    }

    /// Adds a pipeline operator with custom output dimensions.
    ///
    /// Use this when the operator changes dimensions (e.g., reduction operations).
    pub fn pipe_with_dims(mut self, operator: Operator, output_dims: Vec<usize>) -> Self {
        self.stages.push(Stage::PipeOpWithDims {
            operator,
            output_dims,
        });
        self
    }

    /// Automatically adds rebase nodes based on the operator's `rebase_scale_factor()`.
    ///
    /// This queries the last added operator's trait method to determine if and how
    /// to rebase the result.
    pub fn with_auto_rebase(mut self) -> Self {
        self.auto_rebase = true;
        self
    }

    /// Explicitly sets a rebase factor (1 << (scale * factor)).
    ///
    /// Use this when you need custom rebase behavior.
    pub fn with_rebase_factor(mut self, scale_multiplier: usize) -> Self {
        let scale = self.hctx.run_args.scale;
        self.custom_rebase_factor = Some(1_i32 << (scale * scale_multiplier as i32));
        self
    }

    /// Adds a division by a constant value.
    ///
    /// Useful for reductions that need to divide by element count.
    pub fn div_by_constant(mut self, value: i32) -> Self {
        self.stages.push(Stage::DivByConstant { value });
        self
    }

    /// Builds and returns the computation nodes.
    ///
    /// This method:
    /// 1. Calculates the total number of nodes needed
    /// 2. Reserves indices from the parsing context
    /// 3. Constructs all nodes with proper index references
    pub fn build(mut self) -> Vec<ComputationNode> {
        let bc_count = self.broadcast_nodes.len();
        let stage_count = self.count_stage_nodes();
        let rebase_count = self.count_rebase_nodes();
        let total = bc_count + stage_count + rebase_count;

        let mut builder = DecompositionBuilder::new(self.hctx.ctx, total);

        // Add broadcast nodes
        for node in self.broadcast_nodes.drain(..) {
            builder.add_node(node);
        }

        // Track the current "output" index for pipeline connections
        let mut current_output_idx = if bc_count > 0 {
            // After broadcast, inputs have been updated in hctx.internal_input_indices
            None // Will use internal_input_indices
        } else {
            None
        };
        let mut node_offset = bc_count;

        // Add stage nodes
        for stage in &self.stages {
            match stage {
                Stage::SimpleOp { operator } => {
                    builder.add_node(ComputationNode {
                        idx: builder.idx(node_offset),
                        operator: operator.clone(),
                        inputs: self.hctx.internal_input_indices.clone(),
                        output_dims: self.hctx.output_dims.clone(),
                    });
                    current_output_idx = Some(builder.idx(node_offset));
                    node_offset += 1;
                }
                Stage::PipeOp { operator } => {
                    let inputs = match current_output_idx {
                        Some(idx) => vec![idx],
                        None => self.hctx.internal_input_indices.clone(),
                    };
                    builder.add_node(ComputationNode {
                        idx: builder.idx(node_offset),
                        operator: operator.clone(),
                        inputs,
                        output_dims: self.hctx.output_dims.clone(),
                    });
                    current_output_idx = Some(builder.idx(node_offset));
                    node_offset += 1;
                }
                Stage::PipeOpWithDims {
                    operator,
                    output_dims,
                } => {
                    let inputs = match current_output_idx {
                        Some(idx) => vec![idx],
                        None => self.hctx.internal_input_indices.clone(),
                    };
                    builder.add_node(ComputationNode {
                        idx: builder.idx(node_offset),
                        operator: operator.clone(),
                        inputs,
                        output_dims: output_dims.clone(),
                    });
                    current_output_idx = Some(builder.idx(node_offset));
                    node_offset += 1;
                }
                Stage::Constant { tensor } => {
                    builder.add_node(ComputationNode {
                        idx: builder.idx(node_offset),
                        operator: Operator::Constant(Constant(tensor.clone())),
                        inputs: vec![],
                        output_dims: self.hctx.output_dims.clone(),
                    });
                    current_output_idx = Some(builder.idx(node_offset));
                    node_offset += 1;
                }
                Stage::DivByPrevious => {
                    let prev_idx =
                        current_output_idx.expect("DivByPrevious requires a previous node");
                    let prev_prev_idx = builder.idx(node_offset - 2);
                    builder.add_node(ComputationNode {
                        idx: builder.idx(node_offset),
                        operator: Operator::Div(Default::default()),
                        inputs: vec![prev_prev_idx, prev_idx],
                        output_dims: self.hctx.output_dims.clone(),
                    });
                    current_output_idx = Some(builder.idx(node_offset));
                    node_offset += 1;
                }
                Stage::DivByConstant { value } => {
                    let prev_idx =
                        current_output_idx.expect("DivByConstant requires a previous node");
                    let output_dims = self.hctx.output_dims.clone();

                    // Add constant node
                    builder.add_node(ComputationNode {
                        idx: builder.idx(node_offset),
                        operator: Operator::Constant(Constant(Tensor::construct(
                            vec![*value; output_dims.iter().product()],
                            output_dims.clone(),
                        ))),
                        inputs: vec![],
                        output_dims: output_dims.clone(),
                    });
                    let const_idx = builder.idx(node_offset);
                    node_offset += 1;

                    // Add div node
                    builder.add_node(ComputationNode {
                        idx: builder.idx(node_offset),
                        operator: Operator::Div(Default::default()),
                        inputs: vec![prev_idx, const_idx],
                        output_dims,
                    });
                    current_output_idx = Some(builder.idx(node_offset));
                    node_offset += 1;
                }
            }
        }

        // Add rebase nodes if needed
        if let Some(factor) = self.determine_rebase_factor() {
            let prev_idx = current_output_idx.expect("Rebase requires a previous node");
            let output_dims = self.hctx.output_dims.clone();

            // Add constant node for scale divisor
            builder.add_node(ComputationNode {
                idx: builder.idx(node_offset),
                operator: Operator::Constant(Constant(Tensor::construct(
                    vec![factor; output_dims.iter().product()],
                    output_dims.clone(),
                ))),
                inputs: vec![],
                output_dims: output_dims.clone(),
            });
            let const_idx = builder.idx(node_offset);
            node_offset += 1;

            // Add div node
            builder.add_node(ComputationNode {
                idx: builder.idx(node_offset),
                operator: Operator::Div(Default::default()),
                inputs: vec![prev_idx, const_idx],
                output_dims,
            });
        }

        builder.finish()
    }

    /// Counts the number of nodes needed for stages.
    fn count_stage_nodes(&self) -> usize {
        self.stages
            .iter()
            .map(|s| match s {
                Stage::SimpleOp { .. }
                | Stage::PipeOp { .. }
                | Stage::PipeOpWithDims { .. }
                | Stage::Constant { .. } => 1,
                Stage::DivByPrevious => 1,
                Stage::DivByConstant { .. } => 2, // const + div
            })
            .sum()
    }

    /// Counts the number of nodes needed for rebase.
    fn count_rebase_nodes(&self) -> usize {
        if self.determine_rebase_factor().is_some() {
            2 // const + div
        } else {
            0
        }
    }

    /// Determines the rebase factor based on settings and operator traits.
    fn determine_rebase_factor(&self) -> Option<i32> {
        if let Some(factor) = self.custom_rebase_factor {
            return Some(factor);
        }

        if self.auto_rebase {
            // Find the last operator that might need rebase
            for stage in self.stages.iter().rev() {
                let operator = match stage {
                    Stage::SimpleOp { operator } => Some(operator),
                    Stage::PipeOp { operator } => Some(operator),
                    Stage::PipeOpWithDims { operator, .. } => Some(operator),
                    _ => None,
                };
                if let Some(op) = operator {
                    if let Some(scale_mult) = op.inner().rebase_scale_factor() {
                        let scale = self.hctx.run_args.scale;
                        return Some(1_i32 << (scale * scale_mult as i32));
                    }
                }
            }
        }

        None
    }
}

/// Macro for defining simple handlers with minimal boilerplate.
///
/// # Examples
///
/// ```ignore
/// // Simple operator without rebase
/// simple_handler!(handle_add, Operator::Add(Default::default()));
///
/// // Operator with automatic rebase
/// simple_handler!(handle_mul, Operator::Mul(Default::default()), rebase);
/// ```
#[macro_export]
macro_rules! simple_handler {
    ($name:ident, $op:expr) => {
        fn $name(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
            $crate::utils::handler_builder::HandlerBuilder::new(hctx)
                .with_broadcast()
                .simple_op($op)
                .build()
        }
    };
    ($name:ident, $op:expr, rebase) => {
        fn $name(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
            $crate::utils::handler_builder::HandlerBuilder::new(hctx)
                .with_broadcast()
                .simple_op($op)
                .with_auto_rebase()
                .build()
        }
    };
}

pub use simple_handler;
