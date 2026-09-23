use crate::ops::FusedIntermediates;
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
    #[tracing::instrument(name = "Model::execute_graph", skip_all)]
    pub fn execute_graph(&self, inputs: &[Tensor<i32>]) -> BTreeMap<usize, Tensor<i32>> {
        self.execute_graph_with_intermediates(inputs).0
    }

    /// Like [`Self::execute_graph`], but also retains every fused rescale op's
    /// [`FusedIntermediates`] (pre-clamp quotient + remainder) so the prover can
    /// read them back instead of re-running the accumulation.
    pub fn execute_graph_with_intermediates(
        &self,
        inputs: &[Tensor<i32>],
    ) -> (
        BTreeMap<usize, Tensor<i32>>,
        BTreeMap<usize, FusedIntermediates>,
    ) {
        let mut node_outputs: BTreeMap<usize, Tensor<i32>> = BTreeMap::new();
        let mut fused_intermediates: BTreeMap<usize, FusedIntermediates> = BTreeMap::new();
        self.store_inputs(inputs, &mut node_outputs);
        for (node_idx, node) in &self.graph.nodes {
            // Skip input nodes as they're already processed
            if matches!(node.operator, Operator::Input(_)) {
                continue;
            }
            let input_tensors: Vec<&Tensor<i32>> = self.get_node_inputs(*node_idx, &node_outputs);
            let (output_tensor, intermediates) = node.operator.f_with_intermediates(input_tensors);
            node_outputs.insert(*node_idx, output_tensor);
            if let Some(intermediates) = intermediates {
                fused_intermediates.insert(*node_idx, intermediates);
            }
        }
        (node_outputs, fused_intermediates)
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
    /// If the model was loaded with padding, input tensors are automatically padded
    /// to match the expected padded dimensions.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A slice of input tensors to store
    /// * `node_outputs` - A mutable map to store the input tensors, indexed by node ID
    #[tracing::instrument(name = "Model::store_inputs", skip_all)]
    fn store_inputs(
        &self,
        inputs: &[Tensor<i32>],
        node_outputs: &mut BTreeMap<usize, Tensor<i32>>,
    ) {
        for (i, input_tensor) in inputs.iter().enumerate() {
            let input_node_idx = self.graph.inputs[i];

            // Verify input matches the dimensions the model declares for it
            let raw_dims = self.graph.raw_model_input_dims(i);
            assert_eq!(
                input_tensor.dims(),
                raw_dims.as_slice(),
                "Input tensor {} has dims {:?}, expected {:?}",
                i,
                input_tensor.dims(),
                raw_dims
            );

            // Pad up to the node's stored dimensions; a no-op when unpadded
            let node = self.graph.nodes.get(&input_node_idx).unwrap();
            let mut tensor_to_store = input_tensor.clone();
            tensor_to_store
                .pad_to_dims(&node.raw_or_padded_output_dims())
                .expect("Failed to pad input tensor");

            node_outputs.insert(input_node_idx, tensor_to_store);
        }
    }

    /// Extracts the output tensors from the computed node outputs.
    ///
    /// If the model was loaded with padding, output tensors are automatically
    /// cropped back to their original (unpadded) dimensions.
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
            .enumerate()
            .map(|(i, &node_idx)| {
                let tensor = node_outputs.get(&node_idx).unwrap();
                let raw_dims = self.graph.raw_model_output_dims(i);
                if raw_dims.as_slice() == tensor.dims() {
                    return tensor.clone();
                }
                let ranges: Vec<_> = raw_dims.iter().map(|&d| 0..d).collect();
                tensor
                    .get_slice(&ranges)
                    .expect("failed to crop padded output to original dims")
            })
            .collect()
    }
}
