//! Core model types for loading and executing ONNX models.

use crate::{node::ComputationNode, ops::Operator, tensor::Tensor, utils::quantize};
use common::consts::LOG_K_CHUNK;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Functions for executing models and tracing intermediate outputs.
pub mod execute;
/// Functions for loading models from ONNX files.
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
    /// ```ignore
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

    /// Run an ONNX model directly through Tract (unquantized, f32 precision).
    ///
    /// This loads the ONNX file, concretizes dynamic dimensions from `run_args`,
    /// converts the typed graph into a runnable model, executes the forward pass
    /// with the supplied f32 input tensors, and returns f32 output tensors.
    ///
    /// Unlike [`Model::forward`], this does **not** quantize inputs/outputs—it
    /// operates entirely in native floating-point, which is useful for comparing
    /// against the quantized path or running stand-alone inference.
    ///
    /// Inputs are matched to model slots **by name** (e.g. `"input_ids"`,
    /// `"attention_mask"`). Any model input not present in the map is
    /// automatically filled with a zero tensor of the correct shape and dtype
    /// (useful for past-key-value caches on the first forward pass).
    ///
    /// # Arguments
    ///
    /// * `path`     - Path to the ONNX model file (e.g., `"models/gpt2/network.onnx"`)
    /// * `run_args` - Runtime arguments (variables such as `batch_size`, `sequence_length`, etc.)
    /// * `inputs`   - Named input tensors as `(name, Tensor<f32>)` pairs
    ///
    /// # Returns
    ///
    /// A `Vec<Tensor<f32>>` containing one tensor per model output.
    ///
    /// # Panics
    ///
    /// Panics if the ONNX file cannot be loaded, the model cannot be made runnable,
    /// or the forward pass fails.
    pub fn run_tract_forward(
        path: &str,
        run_args: &RunArgs,
        inputs: &[(&str, crate::tensor::Tensor<f32>)],
    ) -> Vec<crate::tensor::Tensor<f32>> {
        use tract_onnx::{prelude::*, tract_core::internal::IntoArcTensor};

        // 1. Load & prepare the typed graph (reusing the existing helper)
        let (typed_model, _symbol_values) = Self::load_onnx_using_tract(path, run_args);

        // 2. Discover each input's expected shape, DatumType, and name
        let input_outlets = typed_model.input_outlets().unwrap().to_vec();

        let input_facts: Vec<(Vec<usize>, DatumType, String)> = input_outlets
            .iter()
            .map(|o| {
                let node = typed_model.node(o.node);
                let fact = &node.outputs[o.slot].fact;
                let shape: Vec<usize> = fact
                    .shape
                    .as_concrete()
                    .expect("all input dims must be concrete after concretization")
                    .to_vec();
                (shape, fact.datum_type, node.name.clone())
            })
            .collect();

        // Build a lookup from caller-supplied name → tensor
        let input_map: HashMap<&str, &crate::tensor::Tensor<f32>> =
            inputs.iter().map(|(name, t)| (*name, t)).collect();

        // 3. Build a runnable model
        let runnable = typed_model
            .into_runnable()
            .expect("failed to build runnable tract model");

        // Helper: convert a crate::tensor::Tensor<f32> to a tract TValue
        let to_tvalue = |t: &crate::tensor::Tensor<f32>, dt: DatumType| -> TValue {
            let shape: Vec<usize> = t.dims().to_vec();
            let data = t.data();
            let tract_tensor: tract_onnx::prelude::Tensor = match dt {
                DatumType::I64 => {
                    let cast: Vec<i64> = data.iter().map(|&v| v as i64).collect();
                    tract_onnx::prelude::Tensor::from_shape(&shape, &cast).unwrap()
                }
                DatumType::I32 => {
                    let cast: Vec<i32> = data.iter().map(|&v| v as i32).collect();
                    tract_onnx::prelude::Tensor::from_shape(&shape, &cast).unwrap()
                }
                _ => tract_onnx::prelude::Tensor::from_shape(&shape, data).unwrap(),
            };
            tract_tensor.into_tvalue()
        };

        // 4. Build the full input TVec, matching by name.
        //    Missing inputs (e.g. past-key-value caches) get zero-filled.
        let tract_inputs: TVec<TValue> = input_facts
            .iter()
            .map(|(shape, dt, name)| {
                if let Some(tensor) = input_map.get(name.as_str()) {
                    to_tvalue(tensor, *dt)
                } else {
                    // Auto-fill with zeros
                    let numel: usize = shape.iter().product();
                    let zeros = vec![0.0f32; numel];
                    let zero_tensor = crate::tensor::Tensor::construct(zeros, shape.clone());
                    to_tvalue(&zero_tensor, *dt)
                }
            })
            .collect();

        // 5. Run the forward pass
        let results = runnable
            .run(tract_inputs)
            .expect("tract forward pass failed");

        // 6. Convert tract outputs → Tensor<f32>
        results
            .into_iter()
            .map(|tv| {
                crate::utils::parser::extract_tensor_value(tv.into_arc_tensor())
                    .expect("failed to extract tensor from tract output")
            })
            .collect()
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

    /// Get a nodes input nodes
    pub fn get_input_nodes(&self, node: &ComputationNode) -> Vec<&ComputationNode> {
        node.inputs
            .iter()
            .filter_map(|idx| self.graph.nodes.get(idx))
            .collect()
    }

    /// Calculate the maximum number of variables needed for any single node in the model.
    pub fn max_num_vars(&self) -> usize {
        let log_2 = |x: usize| {
            assert_ne!(x, 0);

            if x.is_power_of_two() {
                (1usize.leading_zeros() - x.leading_zeros()) as usize
            } else {
                (0usize.leading_zeros() - x.leading_zeros()) as usize
            }
        };

        self.graph
            .nodes
            .values()
            .map(|node| match &node.operator {
                Operator::Tanh(_)
                | Operator::Erf(_)
                | &Operator::ReLU(_)
                | Operator::Div(_)
                | Operator::Rsqrt(_) => LOG_K_CHUNK + log_2(node.num_output_elements()),
                Operator::ScalarConstDiv(_) => log_2(node.num_output_elements()),
                Operator::SoftmaxAxes(_) => {
                    LOG_K_CHUNK + log_2(*node.output_dims.last().unwrap_or(&1))
                }
                Operator::Gather(_) => {
                    let input_nodes = self.get_input_nodes(node);
                    let num_words = input_nodes[0].output_dims[0];
                    let num_indices = input_nodes[1].num_output_elements();
                    log_2(num_words) + log_2(num_indices) // TODO: Gather ra virtualization
                }
                _ => 1,
            })
            .max()
            .unwrap_or(1)
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
    /// When true, divide inputs by 1<<scale BEFORE Square/Cube (to prevent i32 overflow)
    /// instead of dividing the output AFTER (the default rebase).
    /// Enable this for large models (e.g., GPT-2) whose weight magnitudes would overflow.
    pub pre_rebase_nonlinear: bool,
}

impl Default for RunArgs {
    fn default() -> Self {
        let mut variables = HashMap::new();
        variables.insert("batch_size".to_string(), 1);
        RunArgs {
            variables,
            scale: DEFAULT_SCALE,
            pre_rebase_nonlinear: false,
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
            pre_rebase_nonlinear: false,
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
            pre_rebase_nonlinear: false,
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

    /// Enable pre-rebase for nonlinear ops (Square, Cube).
    ///
    /// When enabled, inputs to Square/Cube are divided by `1 << scale` BEFORE
    /// the operation instead of dividing the output AFTER. This prevents i32
    /// overflow for large models (e.g., GPT-2) at the cost of some precision.
    pub fn with_pre_rebase_nonlinear(mut self, enable: bool) -> Self {
        self.pre_rebase_nonlinear = enable;
        self
    }
}

/// Default quantization scale (denominator in fixed-point representation).
pub const DEFAULT_SCALE: i32 = 7;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // Run with `-- --nocapture`
    // Allows to assert the model builds as expected
    fn test_load_reshape_model() {
        let run_args = RunArgs::default();
        let model = Model::load("models/reshape/network.onnx", &run_args);

        println!("{}", model.pretty_print());

        assert!(!model.graph.nodes.is_empty());
        assert!(!model.graph.inputs.is_empty());
        assert!(!model.graph.outputs.is_empty());
    }

}
