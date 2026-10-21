//! The proof layer's per-node working view over an [`atlas_trace::Model`].
//!
//! Op provers and their params structs need an **owned** bundle of node
//! facts (index, operator, operand indices, dims, clamp width, scale) that
//! outlives model borrows across sumcheck lifetimes. [`ComputationNode`] is
//! that bundle, built on demand by [`ModelView::view`]; the model itself is
//! stored and serialized only as the raw `atlas_trace::Model`.

use atlas_trace::{Op, Tensor};
use serde::{Deserialize, Serialize};

/// A node viewed the way the proof layer consumes it: with its index,
/// operator, usize operand indices, dims, clamp width, and the model scale.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComputationNode {
    /// Dense node index (topological order).
    pub idx: usize,
    /// The operator (constants are `Arc`'d, so cloning is cheap).
    pub operator: Op,
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
    pub fn new(idx: usize, operator: Op, inputs: Vec<usize>, output_dims: Vec<usize>) -> Self {
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

/// Proof-layer accessors over a compiled [`atlas_trace::Model`].
pub trait ModelView {
    /// The model's log2 fixed-point scale as the proof layer carries it.
    fn scale_i32(&self) -> i32;
    /// Number of nodes in the graph.
    fn num_nodes(&self) -> usize;
    /// Build the owned node view at `idx`.
    fn view(&self, idx: usize) -> ComputationNode;
    /// Node views in topological order.
    fn views(&self) -> Vec<ComputationNode>;
    /// Owned views of a node's operands.
    fn operand_views(&self, n: &ComputationNode) -> Vec<ComputationNode>;
    /// Graph input node indices.
    fn input_indices(&self) -> Vec<usize>;
    /// Graph output node indices.
    fn output_indices(&self) -> Vec<usize>;
    /// Max committed-poly variable count over all nodes (SRS sizing).
    fn max_num_vars(&self) -> usize;
    /// Number of variables of the largest committed polynomial this node
    /// contributes (1 when it commits nothing).
    fn node_committed_poly_num_vars(&self, node: &ComputationNode) -> usize;
    /// One line per node.
    fn pretty_print(&self) -> String;
}

impl ModelView for atlas_trace::Model {
    fn scale_i32(&self) -> i32 {
        self.scale().log2() as i32
    }

    fn num_nodes(&self) -> usize {
        self.graph().nodes.len()
    }

    fn view(&self, idx: usize) -> ComputationNode {
        let id = atlas_trace::NodeId(idx as u32);
        let (node, meta) = self.node(id);
        ComputationNode {
            idx,
            operator: node.op.clone(),
            inputs: node.inputs.iter().map(|i| i.index()).collect(),
            output_dims: node.output_dims.clone(),
            sat_clamp_bits: meta.sat_clamp_bits.map(|w| w.bits()).unwrap_or(64),
            scale: self.scale_i32(),
        }
    }

    fn views(&self) -> Vec<ComputationNode> {
        (0..self.num_nodes()).map(|i| self.view(i)).collect()
    }

    fn operand_views(&self, n: &ComputationNode) -> Vec<ComputationNode> {
        n.inputs.iter().map(|&i| self.view(i)).collect()
    }

    fn input_indices(&self) -> Vec<usize> {
        self.inputs().iter().map(|i| i.index()).collect()
    }

    fn output_indices(&self) -> Vec<usize> {
        self.outputs().iter().map(|i| i.index()).collect()
    }

    fn max_num_vars(&self) -> usize {
        (0..self.num_nodes())
            .map(|i| self.node_committed_poly_num_vars(&self.view(i)))
            .max()
            .unwrap_or(1)
    }

    fn node_committed_poly_num_vars(&self, node: &ComputationNode) -> usize {
        use common::consts::LOG_K_CHUNK;
        let log_2 = |x: usize| {
            assert_ne!(x, 0);
            if x.is_power_of_two() {
                x.trailing_zeros() as usize
            } else {
                (usize::BITS - x.leading_zeros()) as usize
            }
        };
        match &node.operator {
            Op::Tanh(_)
            | Op::Cos(_)
            | Op::Div(_)
            | Op::Erf(_)
            | Op::Clamp(_)
            | Op::ReLU(_)
            | Op::Rsqrt(_)
            | Op::Sigmoid(_)
            | Op::Sin(_)
            | Op::Add(_)
            | Op::Sub(_)
            | Op::Sum(_)
            | Op::Einsum(_)
            | Op::MeanOfSquares(_)
            | Op::SoftmaxLastAxis(_) => LOG_K_CHUNK + log_2(node.pow2_padded_num_output_elements()),
            Op::Mul(op) if op.rebase => LOG_K_CHUNK + log_2(node.pow2_padded_num_output_elements()),
            Op::Square(op) if op.rebase => {
                LOG_K_CHUNK + log_2(node.pow2_padded_num_output_elements())
            }
            Op::Cube(op) if op.rebase => {
                LOG_K_CHUNK + log_2(node.pow2_padded_num_output_elements())
            }
            Op::ScalarConstDiv(_) => log_2(node.pow2_padded_num_output_elements()),
            Op::GatherSmall(_) => {
                let input_nodes = self.operand_views(node);
                let num_words = input_nodes[0].output_dims[0];
                let num_indices = input_nodes[1].pow2_padded_num_output_elements();
                log_2(num_words) + log_2(num_indices)
            }
            Op::GatherLarge(_) => {
                let input_nodes = self.operand_views(node);
                let num_words = input_nodes[0].output_dims[0].next_power_of_two();
                let num_indices = input_nodes[1].pow2_padded_num_output_elements();
                log_2(num_words) + log_2(num_indices)
            }
            _ => 1,
        }
    }

    fn pretty_print(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        for node in self.views() {
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
}

/// Sound magnitude bound on a node's pre-clamp value, or `None` for
/// unclamped nodes. Delegates to the op's own `clamp_bound`.
pub fn clamp_value_bound(model: &atlas_trace::Model, node: &ComputationNode) -> Option<u128> {
    let graph = model.graph();
    let operand = |i: usize| &graph.nodes[i];
    let input_dims: Vec<&[usize]> = node
        .inputs
        .iter()
        .map(|&i| operand(i).output_dims.as_slice())
        .collect();
    let input_consts: Vec<Option<&Tensor<i32>>> = node
        .inputs
        .iter()
        .map(|&i| match &operand(i).op {
            Op::Constant(c) => Some(c.0.as_ref()),
            _ => None,
        })
        .collect();
    let ctx = atlas_trace::op::BoundCtx {
        scale: model.scale(),
        input_dims,
        input_consts,
    };
    node.operator.spec().clamp_bound(&ctx)
}
