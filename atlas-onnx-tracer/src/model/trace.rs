//! Helpers for tracing model execution and inspecting per-node tensors.

use crate::{model::Model, node::ComputationNode, tensor::Tensor};
use std::{collections::BTreeMap, ops::Index};

impl Model {
    /// Execute the graph and capture every node's output tensor.
    pub fn trace(&self, inputs: &[Tensor<i32>]) -> Trace {
        Trace::new(self.execute_graph(inputs))
    }
}

/// Captures intermediate node outputs from a model run.
pub struct Trace {
    pub node_outputs: BTreeMap<usize, Tensor<i32>>,
}

impl Trace {
    /// Create a trace from a map of node indices to their outputs.
    pub fn new(node_outputs: BTreeMap<usize, Tensor<i32>>) -> Self {
        Self { node_outputs }
    }

    /// Build a view of a specific node, including its inputs and output.
    pub fn layer_data<'a>(&'a self, model: &'a Model, node_index: usize) -> LayerData<'a> {
        let computation_node = &model[node_index];
        let layer_output = &self[node_index];
        let input_tensors = self.input_tensors(computation_node);
        LayerData {
            computation_node,
            layer_output,
            input_tensors,
        }
    }

    /// Return all input tensors feeding the provided computation node.
    pub fn input_tensors(&self, computation_node: &ComputationNode) -> Vec<&Tensor<i32>> {
        computation_node
            .inputs
            .iter()
            .map(|&input_node_idx| self.node_outputs.get(&input_node_idx).unwrap())
            .collect()
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

/// Metadata, inputs, and output for a single computation node.
pub struct LayerData<'a> {
    pub computation_node: &'a ComputationNode,
    pub layer_output: &'a Tensor<i32>,
    pub input_tensors: Vec<&'a Tensor<i32>>,
}
