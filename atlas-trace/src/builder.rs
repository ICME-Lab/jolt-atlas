//! Programmatic graph construction for tests (the `ModelBuilder`
//! replacement the proof layer's test library builds on).
//!
//! Unlike the old builder, output dims are not hand-supplied: every pushed
//! node's shape comes from its op's own [`crate::op::OpSpec::infer_shape`],
//! so a builder graph can never disagree with shape checking.

use crate::{
    error::GraphError,
    graph::{Graph, Node},
    id::NodeId,
    model::{CompileOptions, Model},
    op::{self, Op},
    tensor::Tensor,
};

/// Builds a [`Graph`] node by node in topological order.
#[derive(Default)]
pub struct GraphBuilder {
    nodes: Vec<Node>,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
}

impl GraphBuilder {
    /// An empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a graph input with the given (unpadded) dims.
    pub fn input(&mut self, dims: &[usize]) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            op: Op::Input(op::Input),
            inputs: vec![],
            output_dims: dims.to_vec(),
        });
        self.inputs.push(id);
        id
    }

    /// Add a constant tensor node.
    pub fn constant(&mut self, tensor: Tensor<i32>) -> NodeId {
        let dims = tensor.dims().to_vec();
        self.push_with_dims(Op::Constant(op::Constant(tensor)), &[], dims)
    }

    /// Add a node; its output dims are inferred from its operands.
    ///
    /// # Panics
    /// On arity or shape errors — this is a test-construction API, and a
    /// mis-built graph should fail at the build site.
    pub fn push(&mut self, op: Op, inputs: &[NodeId]) -> NodeId {
        let input_dims: Vec<&[usize]> = inputs
            .iter()
            .map(|id| self.nodes[id.index()].output_dims.as_slice())
            .collect();
        let dims = op
            .spec()
            .infer_shape(&input_dims)
            .unwrap_or_else(|e| panic!("GraphBuilder: {op}: {e}"));
        self.push_with_dims(op, inputs, dims)
    }

    fn push_with_dims(&mut self, op: Op, inputs: &[NodeId], dims: Vec<usize>) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            op,
            inputs: inputs.to_vec(),
            output_dims: dims,
        });
        id
    }

    /// Declare a node as a graph output.
    pub fn output(&mut self, id: NodeId) {
        self.outputs.push(id);
    }

    /// Finish the graph (validated).
    pub fn build_graph(self) -> Result<Graph, GraphError> {
        let graph = Graph {
            nodes: self.nodes,
            inputs: self.inputs,
            outputs: self.outputs,
        };
        graph.validate()?;
        Ok(graph)
    }

    /// Finish and compile in one step.
    pub fn build(self, opts: CompileOptions) -> Result<Model, crate::error::CompileError> {
        Model::compile(self.build_graph()?, opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{Scale, SemVersion};

    #[test]
    fn builds_and_runs_a_small_graph() {
        let mut b = GraphBuilder::new();
        let x = b.input(&[2, 3]);
        let w = b.constant(Tensor::new(vec![1, 0, 0, 1, 1, 1], &[3, 2]).unwrap());
        let mm = b.push(
            Op::Einsum(op::Einsum {
                equation: "mk,kn->mn".parse().unwrap(),
            }),
            &[x, w],
        );
        let relu = b.push(Op::ReLU(op::ReLU), &[mm]);
        b.output(relu);
        let model = b
            .build(CompileOptions {
                scale: Scale::new(4).unwrap(),
                pad_pow2: true,
                version: SemVersion::V1,
            })
            .unwrap();
        // Inferred dims flow through: einsum output is [2, 2].
        let out = model
            .forward(&[Tensor::new(vec![16, 32, 48, -64, 80, 96], &[2, 3]).unwrap()])
            .unwrap();
        assert_eq!(out[0].dims(), &[2, 2]);
        // Row 1: relu over negative sums clamps at 0.
        assert!(out[0].data().iter().all(|&v| v >= 0));
    }

    #[test]
    #[should_panic(expected = "GraphBuilder")]
    fn bad_arity_fails_at_build_site() {
        let mut b = GraphBuilder::new();
        let x = b.input(&[2]);
        b.push(Op::Add(op::Add), &[x]); // Add needs two operands
    }
}
