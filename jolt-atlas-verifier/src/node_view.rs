//! The proof layer's per-node working view over an [`atlas_trace::Model`].
//!
//! Op provers and their params structs need an **owned** bundle of node
//! facts (index, operator, operand indices, dims, clamp width, scale) that
//! outlives model borrows across sumcheck lifetimes. [`ComputationNode`] is
//! that bundle, built on demand by [`ModelView::view`]; the model itself is
//! stored and serialized only as the raw `atlas_trace::Model`.

use atlas_algebra::utils::errors::ProofVerifyError;
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
    pub scale: u8,
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
            scale: common::consts::MODEL_SCALE as u8,
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

    /// The node index of operand `position`. A position past the node's operands gives
    /// `usize::MAX`, the index of no node, so no accumulator holds an opening keyed by it.
    pub fn input(&self, position: usize) -> usize {
        self.inputs.get(position).copied().unwrap_or(usize::MAX)
    }
}

/// Every node's operands are earlier nodes: the topological order `atlas_trace::Model::compile`
/// establishes, checked again on the model the verifier is handed.
pub fn check_model_shape(model: &atlas_trace::Model) -> Result<(), ProofVerifyError> {
    for (idx, node) in model.graph().nodes.iter().enumerate() {
        if node.inputs.iter().any(|input| input.index() >= idx) {
            return Err(ProofVerifyError::Malformed(format!(
                "model: node {idx} ({}) has an operand that is not an earlier node",
                node.op
            )));
        }
    }
    Ok(())
}

/// Proof-layer accessors over a compiled [`atlas_trace::Model`].
pub trait ModelView {
    /// The model's log2 fixed-point scale as the proof layer carries it.
    fn scale_log2(&self) -> u8;
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
    /// Max committed-poly variable count over all nodes (SRS sizing). `Err` if a node's
    /// operands do not have the shape its op takes.
    fn max_num_vars(&self) -> Result<usize, ProofVerifyError>;
    /// Number of variables of the largest committed polynomial this node
    /// contributes (1 when it commits nothing). `Err` if the node's operands do not have the
    /// shape its op takes.
    fn node_committed_poly_num_vars(
        &self,
        node: &ComputationNode,
    ) -> Result<usize, ProofVerifyError>;
    /// One line per node.
    fn pretty_print(&self) -> String;
}

impl ModelView for atlas_trace::Model {
    fn scale_log2(&self) -> u8 {
        self.scale().log2() as u8
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
            scale: self.scale_log2(),
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

    fn max_num_vars(&self) -> Result<usize, ProofVerifyError> {
        let mut max = None;
        for i in 0..self.num_nodes() {
            max = max.max(Some(self.node_committed_poly_num_vars(&self.view(i))?));
        }
        Ok(max.unwrap_or(1))
    }

    fn node_committed_poly_num_vars(
        &self,
        node: &ComputationNode,
    ) -> Result<usize, ProofVerifyError> {
        use common::consts::LOG_K_CHUNK;
        // Of a nonzero count: a padded element count is at least one, and `gather_sizes`
        // rejects an empty dictionary.
        let log_2 = |x: usize| {
            if x.is_power_of_two() {
                x.trailing_zeros() as usize
            } else {
                (usize::BITS - x.leading_zeros()) as usize
            }
        };
        // A gather's dictionary size (its leading dim) and padded index count.
        let gather_sizes = || -> Result<(usize, usize), ProofVerifyError> {
            let input_nodes = self.operand_views(node);
            let (Some(dictionary), Some(indices)) = (input_nodes.first(), input_nodes.get(1))
            else {
                return Err(ProofVerifyError::Malformed(format!(
                    "model: gather node {} needs a dictionary and an index operand",
                    node.idx
                )));
            };
            let num_words = dictionary
                .output_dims
                .first()
                .copied()
                .filter(|&num_words| num_words != 0)
                .ok_or_else(|| {
                    ProofVerifyError::Malformed(format!(
                        "model: gather node {} has an empty dictionary",
                        node.idx
                    ))
                })?;
            Ok((num_words, indices.pow2_padded_num_output_elements()))
        };
        Ok(match &node.operator {
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
                let (num_words, num_indices) = gather_sizes()?;
                log_2(num_words) + log_2(num_indices)
            }
            Op::GatherLarge(_) => {
                let (num_words, num_indices) = gather_sizes()?;
                log_2(num_words.next_power_of_two()) + log_2(num_indices)
            }
            _ => 1,
        })
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
/// unclamped nodes. Delegates to the op's own `clamp_bound`. `Err` if an operand of the node is
/// not a node of the model.
pub fn clamp_value_bound(
    model: &atlas_trace::Model,
    node: &ComputationNode,
) -> Result<Option<u128>, ProofVerifyError> {
    let graph = model.graph();
    let mut input_dims: Vec<&[usize]> = Vec::with_capacity(node.inputs.len());
    let mut input_consts: Vec<Option<&Tensor<i32>>> = Vec::with_capacity(node.inputs.len());
    for &i in &node.inputs {
        let operand = graph.nodes.get(i).ok_or_else(|| {
            ProofVerifyError::Malformed(format!(
                "model: node {} has operand {i}, which is not a node",
                node.idx
            ))
        })?;
        input_dims.push(operand.output_dims.as_slice());
        input_consts.push(match &operand.op {
            Op::Constant(c) => Some(c.0.as_ref()),
            _ => None,
        });
    }
    let ctx = atlas_trace::op::BoundCtx {
        scale: model.scale(),
        input_dims,
        input_consts,
    };
    Ok(node.operator.spec().clamp_bound(&ctx))
}
