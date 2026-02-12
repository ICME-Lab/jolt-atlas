//! Helpers for tracing model execution and inspecting per-node tensors.

use serde::{Deserialize, Serialize};

use crate::{model::Model, node::ComputationNode, tensor::Tensor};
use std::{collections::BTreeMap, ops::Index};

impl Model {
    /// Execute the graph and capture every node's output tensor.
    pub fn trace(&self, inputs: &[Tensor<i32>]) -> Trace {
        Trace::new(self.execute_graph(inputs))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
/// Captures intermediate node outputs from a model run.
pub struct Trace {
    pub node_outputs: BTreeMap<usize, Tensor<i32>>,
}

impl Trace {
    /// Create a trace from a map of node indices to their outputs.
    pub fn new(node_outputs: BTreeMap<usize, Tensor<i32>>) -> Self {
        Self { node_outputs }
    }

    /// Build a trace view of a specific node/layer -> its inputs and output.
    pub fn layer_data<'a>(&'a self, computation_node: &ComputationNode) -> LayerData<'a> {
        let output = &self[computation_node.idx];
        let operands = self.operand_tensors(computation_node);
        LayerData { output, operands }
    }

    /// Return all input tensors feeding the provided computation node.
    pub fn operand_tensors(&self, computation_node: &ComputationNode) -> Vec<&Tensor<i32>> {
        computation_node
            .inputs
            .iter()
            .map(|&input_node_idx| self.node_outputs.get(&input_node_idx).unwrap())
            .collect()
    }

    /// Construct an [ModelExecutionIO] instance
    pub fn io(&self, model: &Model) -> ModelExecutionIO {
        let inputs = model
            .inputs()
            .iter()
            .map(|&idx| self.node_outputs[&idx].clone())
            .collect();
        let outputs = model
            .outputs()
            .iter()
            .map(|&idx| self.node_outputs[&idx].clone())
            .collect();
        ModelExecutionIO { inputs, outputs }
    }
}

impl Index<usize> for Trace {
    type Output = Tensor<i32>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.node_outputs[&index]
    }
}

impl Index<usize> for Model {
    type Output = ComputationNode;

    fn index(&self, index: usize) -> &Self::Output {
        &self.graph.nodes[&index]
    }
}

#[derive(Debug, Clone)]
/// Metadata, operands, and output for a single computation node.
pub struct LayerData<'a> {
    pub output: &'a Tensor<i32>,
    pub operands: Vec<&'a Tensor<i32>>,
}

pub struct ModelExecutionIO {
    pub inputs: Vec<Tensor<i32>>,
    pub outputs: Vec<Tensor<i32>>,
}
