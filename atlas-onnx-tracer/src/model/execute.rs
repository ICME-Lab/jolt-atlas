use crate::{
    model::Model,
    ops::{Op, Operator},
    tensor::Tensor,
};
use std::collections::BTreeMap;

impl Model {
    /// Executes the computational graph with the provided input tensors.
    ///
    /// This method processes all nodes in the graph sequentially, storing intermediate
    /// results and producing outputs for each node.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A slice of input tensors to feed into the graph
    ///
    /// # Returns
    ///
    /// A `BTreeMap` containing all node outputs, indexed by node ID
    pub fn execute_graph(&self, inputs: &[Tensor<i32>]) -> BTreeMap<usize, Tensor<i32>> {
        let mut node_outputs: BTreeMap<usize, Tensor<i32>> = BTreeMap::new();
        self.store_inputs(inputs, &mut node_outputs);
        for (node_idx, node) in &self.graph.nodes {
            // Skip input nodes as they're already processed
            if matches!(node.operator, Operator::Input(_)) {
                continue;
            }
            let input_tensors: Vec<&Tensor<i32>> = self.get_node_inputs(*node_idx, &node_outputs);
            let output_tensor = node.operator.f(input_tensors);
            node_outputs.insert(*node_idx, output_tensor);
        }
        node_outputs
    }

    /// Retrieves the input tensors for a specific node.
    ///
    /// # Arguments
    ///
    /// * `node_idx` - The index of the node whose inputs are being retrieved
    /// * `node_outputs` - A map of previously computed node outputs
    ///
    /// # Returns
    ///
    /// A vector of references to the input tensors for the specified node
    fn get_node_inputs<'model>(
        &'model self,
        node_idx: usize,
        node_outputs: &'model BTreeMap<usize, Tensor<i32>>,
    ) -> Vec<&'model Tensor<i32>> {
        let node = self.graph.nodes.get(&node_idx).unwrap();
        node.inputs
            .iter()
            .map(|&input_node_idx| node_outputs.get(&input_node_idx).unwrap())
            .collect()
    }

    /// Stores the initial input tensors into the node outputs map.
    ///
    /// This method maps each input tensor to its corresponding input node in the graph.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A slice of input tensors to store
    /// * `node_outputs` - A mutable map to store the input tensors, indexed by node ID
    fn store_inputs(
        &self,
        inputs: &[Tensor<i32>],
        node_outputs: &mut BTreeMap<usize, Tensor<i32>>,
    ) {
        for (i, input_tensor) in inputs.iter().enumerate() {
            let input_node_idx = self.graph.inputs[i];
            node_outputs.insert(input_node_idx, input_tensor.clone());
        }
    }

    /// Extracts the output tensors from the computed node outputs.
    ///
    /// # Arguments
    ///
    /// * `node_outputs` - A map containing all computed node outputs
    ///
    /// # Returns
    ///
    /// A vector of output tensors corresponding to the graph's output nodes
    pub(crate) fn extract_graph_outputs(
        &self,
        node_outputs: &BTreeMap<usize, Tensor<i32>>,
    ) -> Vec<Tensor<i32>> {
        self.graph
            .outputs
            .iter()
            .map(|&node_idx| node_outputs.get(&node_idx).unwrap().clone())
            .collect()
    }
}
