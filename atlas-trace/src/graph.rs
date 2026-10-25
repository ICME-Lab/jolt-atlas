//! The computation graph an importer (or test builder) hands to
//! [`crate::model::Model::compile`].

use serde::{Deserialize, Serialize};

use crate::{error::GraphError, id::NodeId, op::Op};

/// One node: an operator applied to earlier nodes' outputs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// The operator.
    pub op: Op,
    /// Operand node ids, all strictly smaller than this node's id.
    pub inputs: Vec<NodeId>,
    /// Output dims (logical before padding; padded after compile).
    pub output_dims: Vec<usize>,
}

/// A topologically ordered computation graph.
///
/// Nodes are stored densely: a [`NodeId`] is the index into `nodes`, and
/// every node may only reference strictly earlier nodes, so topological
/// order is structural rather than assumed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    /// The nodes, in topological order.
    pub nodes: Vec<Node>,
    /// The `Input` nodes, in the order the caller supplies tensors.
    pub inputs: Vec<NodeId>,
    /// The nodes whose outputs the model exposes.
    pub outputs: Vec<NodeId>,
}

impl Graph {
    /// Structural validation: id ranges, topological order, IO wiring, and
    /// per-op static parameters. Shape checking happens at compile.
    pub fn validate(&self) -> Result<(), GraphError> {
        for (i, node) in self.nodes.iter().enumerate() {
            self.validate_node(NodeId(i as u32), node)?;
        }
        self.validate_inputs()?;
        self.validate_outputs()
    }

    fn check_id(&self, id: NodeId) -> Result<(), GraphError> {
        let len = self.nodes.len();
        if id.index() >= len {
            return Err(GraphError::IdOutOfRange { id, len });
        }
        Ok(())
    }

    /// Operands are in range and strictly earlier, source ops take no
    /// operands, and the op's static parameters are valid.
    fn validate_node(&self, this: NodeId, node: &Node) -> Result<(), GraphError> {
        for &input in &node.inputs {
            self.check_id(input)?;
            if input.index() >= this.index() {
                return Err(GraphError::NotTopological { node: this, input });
            }
        }
        if matches!(node.op, Op::Input(_) | Op::Constant(_)) && !node.inputs.is_empty() {
            return Err(invalid_io(
                this,
                format!("{} nodes take no operands", node.op),
            ));
        }
        node.op
            .spec()
            .validate()
            .map_err(|reason| GraphError::InvalidOp {
                node: this,
                op: node.op.to_string(),
                reason,
            })
    }

    /// Every declared input is an Input node, and every Input node is
    /// declared exactly once.
    fn validate_inputs(&self) -> Result<(), GraphError> {
        for &id in &self.inputs {
            self.check_id(id)?;
            if !matches!(self.nodes[id.index()].op, Op::Input(_)) {
                return Err(invalid_io(
                    id,
                    "declared as graph input but is not an Input node",
                ));
            }
        }
        for (i, node) in self.nodes.iter().enumerate() {
            let this = NodeId(i as u32);
            if !matches!(node.op, Op::Input(_)) {
                continue;
            }
            match self.inputs.iter().filter(|&&id| id == this).count() {
                1 => {}
                0 => return Err(invalid_io(this, "Input node not declared in graph.inputs")),
                n => return Err(invalid_io(this, format!("Input node declared {n} times"))),
            }
        }
        Ok(())
    }

    /// At least one output is declared, and every output id is in range.
    fn validate_outputs(&self) -> Result<(), GraphError> {
        if self.outputs.is_empty() {
            return Err(invalid_io(NodeId(0), "graph declares no outputs"));
        }
        for &id in &self.outputs {
            self.check_id(id)?;
        }
        Ok(())
    }

    /// The node behind an id.
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.index()]
    }
}

fn invalid_io(node: NodeId, reason: impl Into<String>) -> GraphError {
    GraphError::InvalidIo {
        node,
        reason: reason.into(),
    }
}

impl Node {
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

impl std::ops::Index<NodeId> for Graph {
    type Output = Node;

    fn index(&self, id: NodeId) -> &Node {
        &self.nodes[id.index()]
    }
}
