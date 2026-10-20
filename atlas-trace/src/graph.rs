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
        let len = self.nodes.len();
        let check_id = |id: NodeId| {
            if id.index() >= len {
                Err(GraphError::IdOutOfRange { id, len })
            } else {
                Ok(())
            }
        };

        for (i, node) in self.nodes.iter().enumerate() {
            let this = NodeId(i as u32);
            for &input in &node.inputs {
                check_id(input)?;
                if input.index() >= i {
                    return Err(GraphError::NotTopological { node: this, input });
                }
            }
            if matches!(node.op, Op::Input(_) | Op::Constant(_)) && !node.inputs.is_empty() {
                return Err(GraphError::InvalidIo {
                    node: this,
                    reason: format!("{} nodes take no operands", node.op),
                });
            }
            node.op
                .spec()
                .validate()
                .map_err(|reason| GraphError::InvalidOp {
                    node: this,
                    op: node.op.to_string(),
                    reason,
                })?;
        }

        // Every declared input is an Input node, and every Input node is
        // declared exactly once.
        for &id in &self.inputs {
            check_id(id)?;
            if !matches!(self.nodes[id.index()].op, Op::Input(_)) {
                return Err(GraphError::InvalidIo {
                    node: id,
                    reason: "declared as graph input but is not an Input node".into(),
                });
            }
        }
        for (i, node) in self.nodes.iter().enumerate() {
            let this = NodeId(i as u32);
            if matches!(node.op, Op::Input(_)) {
                match self.inputs.iter().filter(|&&id| id == this).count() {
                    1 => {}
                    0 => {
                        return Err(GraphError::InvalidIo {
                            node: this,
                            reason: "Input node not declared in graph.inputs".into(),
                        });
                    }
                    n => {
                        return Err(GraphError::InvalidIo {
                            node: this,
                            reason: format!("Input node declared {n} times"),
                        });
                    }
                }
            }
        }

        if self.outputs.is_empty() {
            return Err(GraphError::InvalidIo {
                node: NodeId(0),
                reason: "graph declares no outputs".into(),
            });
        }
        for &id in &self.outputs {
            check_id(id)?;
        }
        Ok(())
    }

    /// The node behind an id.
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.index()]
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
