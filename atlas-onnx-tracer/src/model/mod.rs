use crate::{node::ComputationNode, tensor::Tensor, utils::quantize};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

pub mod execute;
pub mod load;
pub mod test;
pub mod trace;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Model {
    /// The computation graph of the model
    pub graph: ComputationGraph,
}

impl Model {
    /// load a model from the given path
    pub fn load(path: &str, run_args: &RunArgs) -> Self {
        Self::load_onnx_model(path, run_args)
    }

    /// Forward pass through the model given the input tensors
    pub fn forward(&self, inputs: &[Tensor<i32>]) -> Vec<Tensor<i32>> {
        let node_outputs = self.execute_graph(inputs);
        self.extract_graph_outputs(&node_outputs)
    }
}

impl Model {
    pub fn graph(&self) -> &ComputationGraph {
        &self.graph
    }

    pub fn nodes(&self) -> &BTreeMap<usize, ComputationNode> {
        &self.graph.nodes
    }

    pub fn inputs(&self) -> &[usize] {
        &self.graph.inputs
    }

    pub fn outputs(&self) -> &[usize] {
        &self.graph.outputs
    }

    pub fn max_T(&self) -> usize {
        self.graph
            .nodes
            .values()
            .map(|node| node.num_output_elements())
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
    /// List of output connections (node_index)
    pub outputs: Vec<usize>,
}

/// Parameters specific to a proving run
/// Arguments for running the model
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunArgs {
    /// Map of variable names to their values
    pub variables: HashMap<String, usize>,
    /// The denominator in the fixed point representation used when quantizing the model
    pub scale: quantize::Scale,
}

impl Default for RunArgs {
    fn default() -> Self {
        let mut variables = HashMap::new();
        variables.insert("batch_size".to_string(), 1);
        RunArgs {
            variables,
            scale: DEFAULT_SCALE,
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
        RunArgs { variables, scale }
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

    /// Set the scale for the RunArgs
    pub fn set_scale(mut self, scale: i32) -> Self {
        self.scale = scale;
        self
    }
}

pub const DEFAULT_SCALE: i32 = 7;
