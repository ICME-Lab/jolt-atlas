//! Helpers for tracing model execution and inspecting per-node tensors.

use serde::{Deserialize, Serialize};

use crate::{model::Model, node::ComputationNode, tensor::Tensor};
use std::{collections::BTreeMap, ops::Index};

impl Model {
    #[tracing::instrument(name = "Model::trace", skip_all)]
    /// Execute the graph and capture every node's output tensor.
    pub fn trace(&self, inputs: &[Tensor<i32>]) -> Trace {
        Trace::new(self.execute_graph(inputs))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
/// Captures intermediate node outputs from a model run.
pub struct Trace {
    /// Map from node index to its output tensor.
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
        ModelExecutionIO {
            inputs,
            outputs,
            input_indices: model.inputs().to_vec(),
            output_indices: model.outputs().to_vec(),
        }
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
    /// The output tensor of this layer.
    pub output: &'a Tensor<i32>,
    /// Input tensors (operands) fed into this layer.
    pub operands: Vec<&'a Tensor<i32>>,
}

/// Inputs and outputs of a model execution.
pub struct ModelExecutionIO {
    /// Input tensors provided to the model.
    pub inputs: Vec<Tensor<i32>>,
    /// Output tensors produced by the model.
    pub outputs: Vec<Tensor<i32>>,
    /// Node indices corresponding to model inputs.
    pub input_indices: Vec<usize>,
    /// Node indices corresponding to model outputs.
    pub output_indices: Vec<usize>,
}
