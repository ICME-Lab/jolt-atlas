use crate::{node::ComputationNode, tensor::Tensor, utils::quantize};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

pub mod execute;
pub mod load;
pub mod test;
pub mod trace;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
/// A high-level representation of an ONNX machine learning model.
///
/// The `Model` struct encapsulates a computation graph and provides methods
/// for loading models from ONNX files and executing forward passes with quantized inputs.
pub struct Model {
    /// The computation graph of the model
    pub graph: ComputationGraph,
}

impl Model {
    /// Load a model from an ONNX file at the given path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the ONNX model file
    /// * `run_args` - Runtime arguments including variables, scale, and padding settings
    ///
    /// # Returns
    ///
    /// A `Model` instance with the loaded computation graph
    ///
    /// # Example
    ///
    /// ```no_run
    /// use atlas_onnx_tracer::model::{Model, RunArgs};
    ///
    /// let run_args = RunArgs::default();
    /// let model = Model::load("path/to/model.onnx", &run_args);
    /// ```
    #[tracing::instrument(name = "Model::load", skip_all)]
    pub fn load(path: &str, run_args: &RunArgs) -> Self {
        Self::load_onnx_model(path, run_args)
    }

    /// Execute a forward pass through the model with the given input tensors.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Slice of quantized input tensors (i32)
    ///
    /// # Returns
    ///
    /// A vector of quantized output tensors (i32) corresponding to the model's outputs
    ///
    /// # Example
    ///
    /// ```no_run
    /// use atlas_onnx_tracer::model::{Model, RunArgs};
    /// use atlas_onnx_tracer::tensor::Tensor;
    ///
    /// let run_args = RunArgs::default();
    /// let model = Model::load("path/to/model.onnx", &run_args);
    /// let inputs = vec![Tensor::new(vec![1, 2, 3, 4], vec![2, 2])];
    /// let outputs = model.forward(&inputs);
    /// ```
    #[tracing::instrument(name = "Model::forward", skip_all)]
    pub fn forward(&self, inputs: &[Tensor<i32>]) -> Vec<Tensor<i32>> {
        let node_outputs = self.execute_graph(inputs);
        self.extract_graph_outputs(&node_outputs)
    }
}

impl Model {
    /// Returns a reference to the underlying computation graph.
    pub fn graph(&self) -> &ComputationGraph {
        &self.graph
    }

    /// Returns a reference to the map of computation nodes in the graph.
    ///
    /// Each node is identified by its index in the computation graph.
    pub fn nodes(&self) -> &BTreeMap<usize, ComputationNode> {
        &self.graph.nodes
    }

    /// Returns the indices of input nodes in the computation graph.
    pub fn inputs(&self) -> &[usize] {
        &self.graph.inputs
    }

    /// Returns the indices of output nodes in the computation graph.
    pub fn outputs(&self) -> &[usize] {
        &self.graph.outputs
    }

    /// Returns the maximum T value across all nodes in the graph.
    ///
    /// T is defined as the next power of 2 greater than or equal to the number of
    /// output elements for each node. This is used to determine the size of the commitment key in the proof system.
    pub fn max_T(&self) -> usize {
        self.graph
            .nodes
            .values()
            .map(|node| node.num_output_elements().next_power_of_two())
            .max()
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
/// A computation graph representing the operations in a transformer model.
/// Each node corresponds to a specific operation, and edges represent data flow between operations.
pub struct ComputationGraph {
    /// Map of node indices to their corresponding node types
    pub nodes: BTreeMap<usize, ComputationNode>,
    /// Indices of input nodes
    pub inputs: Vec<usize>,
    /// Indices of output nodes
    pub outputs: Vec<usize>,
    /// Original (unpadded) dimensions for input nodes, indexed by node index
    /// Only populated when padding is enabled
    pub original_input_dims: HashMap<usize, Vec<usize>>,
    /// Original (unpadded) dimensions for output nodes, indexed by node index
    /// Only populated when padding is enabled
    pub original_output_dims: HashMap<usize, Vec<usize>>,
}

/// Runtime arguments for model execution and tracing.
///
/// `RunArgs` encapsulates configuration parameters needed to load and execute
/// a model, including variable bindings (e.g., batch_size, sequence_length),
/// quantization scale, and padding options.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunArgs {
    /// Map of variable names to their values
    pub variables: HashMap<String, usize>,
    /// The denominator in the fixed point representation used when quantizing the model
    pub scale: quantize::Scale,
    /// Whether to pad all dimensions to powers of 2.
    /// Defaults to true for optimal cryptographic performance.
    pub pad_to_power_of_2: bool,
}

impl Default for RunArgs {
    fn default() -> Self {
        let mut variables = HashMap::new();
        variables.insert("batch_size".to_string(), 1);
        RunArgs {
            variables,
            scale: DEFAULT_SCALE,
            pad_to_power_of_2: true, // Default to true for prover use-case
        }
    }
}

impl RunArgs {
    /// Create a new RunArgs with the given variables
    ///
    /// # Example
    /// ```
    /// use atlas_onnx_tracer::model::RunArgs;
    /// let run_args = RunArgs::new([
    ///     ("sequence_length", 512),
    ///     ("past_sequence_length", 0),
    /// ]);
    /// ```
    pub fn new<I, K>(variables: I) -> Self
    where
        I: IntoIterator<Item = (K, usize)>,
        K: Into<String>,
    {
        let variables = variables.into_iter().map(|(k, v)| (k.into(), v)).collect();
        RunArgs {
            variables,
            scale: DEFAULT_SCALE,
            pad_to_power_of_2: true,
        }
    }

    /// Create a new RunArgs with the given variables and scale
    ///
    /// # Example
    /// ```
    /// use atlas_onnx_tracer::model::RunArgs;
    /// let run_args = RunArgs::with_scale(
    ///     [("sequence_length", 512)],
    ///     128
    /// );
    /// ```
    pub fn with_scale<I, K>(variables: I, scale: i32) -> Self
    where
        I: IntoIterator<Item = (K, usize)>,
        K: Into<String>,
    {
        let variables = variables.into_iter().map(|(k, v)| (k.into(), v)).collect();
        RunArgs {
            variables,
            scale,
            pad_to_power_of_2: true,
        } // Default to true for optimal cryptographic performance
    }

    /// Add a variable to the RunArgs
    ///
    /// # Example
    /// ```
    /// use atlas_onnx_tracer::model::RunArgs;
    /// let run_args = RunArgs::default()
    ///     .with("sequence_length", 512)
    ///     .with("past_sequence_length", 0);
    /// ```
    pub fn with<K: Into<String>>(mut self, key: K, value: usize) -> Self {
        self.variables.insert(key.into(), value);
        self
    }

    /// Set the quantization scale for the RunArgs.
    ///
    /// The scale is the denominator in the fixed-point representation
    /// used when quantizing the model.
    pub fn set_scale(mut self, scale: i32) -> Self {
        self.scale = scale;
        self
    }

    /// Enable or disable power-of-2 dimension padding
    ///
    /// # Example
    /// ```
    /// use atlas_onnx_tracer::model::RunArgs;
    /// let run_args = RunArgs::default().with_padding(true);
    /// ```
    pub fn with_padding(mut self, enable: bool) -> Self {
        self.pad_to_power_of_2 = enable;
        self
    }
}

pub const DEFAULT_SCALE: i32 = 7;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // Run with `-- --nocapture`
    // Allows to assert the model builds as expected
    fn test_load_reshape_model() {
        let run_args = RunArgs::default().with_padding(false);
        let model = Model::load("models/reshape/network.onnx", &run_args);

        println!("{}", model.pretty_print());

        assert!(!model.graph.nodes.is_empty());
        assert!(!model.graph.inputs.is_empty());
        assert!(!model.graph.outputs.is_empty());
    }

    #[test]
    fn test_load_reshape_model_with_padding() {
        let run_args = RunArgs::default().with_padding(true);
        let model = Model::load("models/reshape/network.onnx", &run_args);

        println!("{}", model.pretty_print());

        assert!(!model.graph.nodes.is_empty());
        assert!(!model.graph.inputs.is_empty());
        assert!(!model.graph.outputs.is_empty());

        // Verify that padding metadata is populated
        assert!(
            !model.graph.original_input_dims.is_empty(),
            "Padded model should have original input dims stored"
        );
        assert!(
            !model.graph.original_output_dims.is_empty(),
            "Padded model should have original output dims stored"
        );

        // Verify all node output dims are powers of 2
        for (idx, node) in &model.graph.nodes {
            for &dim in &node.output_dims {
                assert_eq!(
                    dim,
                    dim.next_power_of_two(),
                    "Node {idx} has non-power-of-2 dimension: {dim}"
                );
            }
        }
    }
}
