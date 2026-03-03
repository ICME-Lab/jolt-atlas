use crate::{
    model::{ComputationGraph, Model, RunArgs},
    node::ComputationNode,
    ops::Operator,
    utils::parser::{GraphParser, NodeIndexMapper, map_outputs},
};
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};
use tract_onnx::prelude::*;

/// Type alias for the graph loading result
pub type GraphLoadResult = (Graph<TypedFact, Box<dyn TypedOp>>, SymbolValues);

impl Model {
    #[tracing::instrument(name = "Model::load_onnx_model", skip_all)]
    /// Load a model from an ONNX file.
    ///
    /// This is the main entry point for loading ONNX models.
    /// Uses a fluent builder pattern that makes each loading step explicit.
    ///
    /// If `run_args.pad_to_power_of_2` is true, all constant tensors and node output
    /// dimensions will be padded to the next power of 2 (e.g., [3, 7, 15, 8] → [4, 8, 16, 8]).
    /// The original input and output dimensions are preserved for proper handling during
    /// execution (inputs will be padded, outputs will be unpadded).
    pub fn load_onnx_model(path: &str, run_args: &RunArgs) -> Self {
        let mut loader = ModelLoader::new(path, run_args)
            .load_onnx_using_tract()
            .parse_nodes()
            .collect_input_nodes()
            .collect_outputs()
            .prune();

        if run_args.pad_to_power_of_2 {
            loader = loader.pad();
        }

        loader.build()
    }

    /// Loads and prepares an ONNX model using the Tract inference engine.
    ///
    /// This function performs the complete pipeline to load an ONNX file and prepare it for execution:
    /// 1. Read the ONNX file from disk
    /// 2. Parse it into a Tract inference model
    /// 3. Concretize any dynamic input dimensions (e.g., batch_size)
    /// 4. Relax output constraints for flexible inference
    /// 5. Build symbol values mapping from run arguments
    /// 6. Convert to a fully typed model with concrete dimensions
    ///
    /// # Arguments
    /// * `path` - Path to the ONNX model file
    /// * `run_args` - Runtime arguments including variable values and scaling parameters
    ///
    /// # Returns
    /// A tuple containing the typed computation graph and the symbol values used for concretization.
    ///
    /// # Panics
    /// Panics if the file cannot be opened, the ONNX model cannot be parsed, or any of the
    /// preparation steps fail (see helper method documentation for specific panic conditions).
    pub fn load_onnx_using_tract<P: AsRef<Path>>(path: P, run_args: &RunArgs) -> GraphLoadResult {
        TractModelLoader::new(path)
            .with_variables(&run_args.variables)
            .with_run_args(run_args)
            .build()
    }

    /// Replaces dynamic dimensions in model inputs with concrete values from variables.
    ///
    /// ONNX models can have dynamic input shapes (e.g., batch_size = -1). This function
    /// resolves those dynamic dimensions by substituting concrete values from the provided
    /// variable map. Any dimension marked as `Any` is replaced with the batch_size value.
    ///
    /// # Arguments
    /// * `model` - The inference graph with potentially dynamic input dimensions. This is a tract type.
    /// * `variables` - Map containing concrete values (must include "batch_size" if model has dynamic dims)
    ///
    /// # Panics
    /// Panics if the model has dynamic input shapes but "batch_size" is not provided in variables,
    /// or if unable to set the input fact for any input in the model.
    fn concretize_model_input_dims_from_variables(
        model: &mut InferenceModel,
        variables: &HashMap<String, usize>,
        symbol_values: &SymbolValues,
    ) {
        use tract_onnx::tract_hir::internal::GenericFactoid;

        // Iterate through each input node in the model
        for (i, id) in model.clone().inputs.iter().enumerate() {
            let input = model.node_mut(id.node);
            let mut fact: InferenceFact = input.outputs[0].fact.clone();

            // Check each dimension of the input shape
            for (dim_idx, x) in fact.clone().shape.dims().enumerate() {
                match x {
                    GenericFactoid::Any => {
                        // If dimension is dynamic (Any), replace it with batch_size
                        let batch_size = variables.get("batch_size").unwrap();
                        fact.shape.set_dim(dim_idx, TDim::Val(*batch_size as i64));
                    }
                    GenericFactoid::Only(tdim) => {
                        // Evaluate symbolic dimensions using symbol_values
                        let resolved = tdim.eval(symbol_values);
                        fact.shape.set_dim(dim_idx, resolved);
                    }
                }
            }

            // Update the model with the concretized input shape
            model.set_input_fact(i, fact).unwrap();
        }
    }

    /// Resets output constraints to allow flexible inference.
    ///
    /// By default, tract may infer specific output types/shapes. This function relaxes
    /// those constraints by setting outputs to default facts, allowing the inference
    /// engine more flexibility during graph analysis and optimization.
    ///
    /// # Panics
    /// Panics if unable to set the output fact for any output in the model.
    fn relax_output_facts(model: &mut InferenceModel) {
        // Reset each output to default inference fact (no type/shape constraints)
        for (i, _) in model.clone().outputs.iter().enumerate() {
            model.set_output_fact(i, InferenceFact::default()).unwrap();
        }
    }

    /// Builds a SymbolValues map that tells tract what concrete integer values to use for
    /// symbolic dimensions (e.g., batch_size, seq_len, etc.)
    /// when the model has shapes expressed with symbols
    #[tracing::instrument(name = "ModelLoader::build_symbol_values_from_run_args", skip_all)]
    fn build_symbol_values_from_run_args(
        model: &InferenceModel,
        run_args: &RunArgs,
    ) -> SymbolValues {
        let mut symbol_values = SymbolValues::default();
        for (symbol, value) in run_args.variables.iter() {
            let symbol = model.symbols.sym(symbol);
            symbol_values = symbol_values.with(&symbol, *value as i64);
        }
        symbol_values
    }

    /// Converts an inference model to a fully typed model with concrete dimensions.
    ///
    /// This performs three transformations:
    /// 1. Convert from inference model (with shape inference) to typed model (with known types)
    /// 2. Concretize all symbolic dimensions using the provided symbol values
    /// 3. Declutter the graph by removing identity operations and simplifying structure
    ///
    /// Note: The model is intentionally not optimized, as optimizations may be
    /// hardware-dependent and could affect portability.
    ///
    /// # Panics
    /// Panics if type conversion fails, dimension concretization fails, or decluttering fails.
    #[tracing::instrument(name = "ModelLoader::model_into_typed", skip_all)]
    fn model_into_typed(
        model: InferenceModel,
        symbol_values: &SymbolValues,
    ) -> Graph<TypedFact, Box<dyn TypedOp>> {
        // Note: do not optimize the model, as the layout will depend on
        // underlying hardware
        model
            .into_typed()
            .unwrap()
            .concretize_dims(symbol_values)
            .unwrap()
            .into_decluttered()
            .unwrap()
    }

    /// Parses the Tract graph into our ComputationNode representation.
    ///
    /// This involves traversing the Tract graph and converting each node
    /// into our domain-specific ComputationNode structure.
    ///
    /// # Arguments
    /// * `graph` - The typed Tract graph to parse
    /// * `run_args` - Runtime arguments for the model
    /// * `symbol_values` - Symbol values for concretizing dimensions
    ///
    /// # Returns
    /// A tuple containing:
    /// - A map of node indices to ComputationNode instances    
    /// - A NodeIndexMapper for mapping Tract node indices to internal indices
    ///
    /// # Panics
    /// Panics if any node cannot be parsed or converted.
    #[tracing::instrument(name = "ModelLoader::nodes_from_graph", skip_all)]
    pub fn nodes_from_graph(
        graph: &Graph<TypedFact, Box<dyn TypedOp>>,
        run_args: &RunArgs,
        symbol_values: &SymbolValues,
    ) -> (BTreeMap<usize, ComputationNode>, NodeIndexMapper) {
        GraphParser::new(graph, run_args, symbol_values).parse()
    }

    /// Collects output node indices from the Tract graph.
    ///
    /// Maps the Tract graph's output connections to internal node indices
    /// using the provided node index mapper.
    ///
    /// # Arguments
    /// * `model` - The typed Tract graph
    /// * `mapper` - Mapper for converting Tract node indices to internal indices
    ///
    /// # Returns
    /// A vector of internal node indices representing the graph outputs
    fn collect_outputs(
        model: &Graph<TypedFact, Box<dyn TypedOp>>,
        mapper: &NodeIndexMapper,
    ) -> Vec<usize> {
        map_outputs(&model.outputs, mapper)
    }

    /// Collects indices of all input nodes from the computation graph.
    ///
    /// Filters the nodes to find only those with an Input operator,
    /// which represent the entry points of the computation graph.
    ///
    /// # Arguments
    /// * `nodes` - Map of all computation nodes in the graph
    ///
    /// # Returns
    /// A vector of node indices that are input nodes
    fn collect_input_nodes(nodes: &BTreeMap<usize, ComputationNode>) -> Vec<usize> {
        nodes
            .iter()
            .filter_map(|(idx, node)| match node.operator {
                Operator::Input(_) => Some(*idx),
                _ => None,
            })
            .collect()
    }

    /// Pads all dimensions in a shape to their next power of 2.
    ///
    /// # Arguments
    /// * `dims` - The original dimensions
    ///
    /// # Returns
    /// A new vector with each dimension padded to the next power of 2
    ///
    /// # Example
    /// ```ignore
    /// let dims = vec![3, 7, 15, 8];
    /// let padded = Model::pad_dims_to_power_of_2(&dims);
    /// assert_eq!(padded, vec![4, 8, 16, 8]);
    /// ```
    fn pad_dims_to_power_of_2(dims: &[usize]) -> Vec<usize> {
        dims.iter().map(|&d| d.next_power_of_two()).collect()
    }

    /// Prunes unused nodes from the computation graph and remaps indices.
    ///
    /// A node is considered "used" if it is reachable from any output node
    /// by traversing the graph backwards through input connections.
    /// This is useful for removing dead code, such as constant nodes that
    /// were part of operations that got fused (e.g., max(0, x) → ReLU).
    ///
    /// # Algorithm
    /// 1. Start from output nodes and traverse backwards to mark all reachable nodes
    /// 2. Remove nodes that are not reachable
    /// 3. Create a mapping from old indices to new contiguous indices
    /// 4. Update all node indices and input references
    /// 5. Update input/output node lists
    ///
    /// # Arguments
    /// * `nodes` - The nodes to prune (modified in place)
    /// * `inputs` - Input node indices (modified in place)
    /// * `outputs` - Output node indices (modified in place)
    ///
    /// # Returns
    /// A mapping from old node indices to new node indices (only for retained nodes)
    #[tracing::instrument(name = "Model::prune_unused_nodes", skip_all)]
    pub fn prune_unused_nodes(
        nodes: &mut BTreeMap<usize, ComputationNode>,
        inputs: &mut Vec<usize>,
        outputs: &mut Vec<usize>,
    ) -> HashMap<usize, usize> {
        use std::collections::HashSet;

        // Step 1: Find all reachable nodes from outputs using BFS/DFS
        let mut reachable: HashSet<usize> = HashSet::new();
        let mut stack: Vec<usize> = outputs.clone();

        while let Some(node_idx) = stack.pop() {
            if reachable.contains(&node_idx) {
                continue;
            }
            reachable.insert(node_idx);

            // Add all inputs of this node to the stack
            if let Some(node) = nodes.get(&node_idx) {
                for &input_idx in &node.inputs {
                    if !reachable.contains(&input_idx) {
                        stack.push(input_idx);
                    }
                }
            }
        }

        // Step 2: Remove unreachable nodes
        let unreachable_nodes: Vec<usize> = nodes
            .keys()
            .filter(|idx| !reachable.contains(idx))
            .copied()
            .collect();

        for idx in unreachable_nodes {
            nodes.remove(&idx);
        }

        // Step 3: Create old->new index mapping for contiguous indices
        // Sort remaining indices and assign new contiguous indices
        let mut old_to_new: HashMap<usize, usize> = HashMap::new();
        let sorted_old_indices: Vec<usize> = nodes.keys().copied().collect();

        for (new_idx, &old_idx) in sorted_old_indices.iter().enumerate() {
            old_to_new.insert(old_idx, new_idx);
        }

        // Step 4: Rebuild nodes with new indices
        let mut new_nodes: BTreeMap<usize, ComputationNode> = BTreeMap::new();

        for old_idx in sorted_old_indices {
            let mut node = nodes.remove(&old_idx).unwrap();
            let new_idx = old_to_new[&old_idx];

            // Update the node's own index
            node.idx = new_idx;

            // Update input references
            node.inputs = node
                .inputs
                .iter()
                .map(|&input_idx| old_to_new[&input_idx])
                .collect();

            new_nodes.insert(new_idx, node);
        }

        *nodes = new_nodes;

        // Step 5: Update input and output node lists
        // Filter out input nodes that were pruned (unreachable from outputs)
        *inputs = inputs
            .iter()
            .filter_map(|&idx| old_to_new.get(&idx).copied())
            .collect();
        *outputs = outputs.iter().map(|&idx| old_to_new[&idx]).collect();

        old_to_new
    }
}

/// Builder for constructing a Model from an ONNX file.
///
/// This builder follows a step-by-step pipeline that mirrors the actual
/// model loading process. Each method performs one logical step and can
/// be called in sequence to build the final Model.
///
/// # Example
/// ```ignore
/// let model = ModelLoader::new(path, &run_args)
///     .load_onnx_using_tract()
///     .parse_nodes()
///     .collect_input_nodes()
///     .collect_outputs()
///     .build();
/// ```
pub struct ModelLoader<'a> {
    path: &'a str,
    run_args: &'a RunArgs,
    tract_graph: Option<Graph<TypedFact, Box<dyn TypedOp>>>,
    symbols: Option<SymbolValues>,
    nodes: Option<BTreeMap<usize, ComputationNode>>,
    mapper: Option<NodeIndexMapper>,
    inputs: Option<Vec<usize>>,
    outputs: Option<Vec<usize>>,
    original_input_dims: HashMap<usize, Vec<usize>>,
    original_output_dims: HashMap<usize, Vec<usize>>,
}

impl<'a> ModelLoader<'a> {
    /// Creates a new ModelBuilder for the given ONNX file and run arguments.
    pub fn new(path: &'a str, run_args: &'a RunArgs) -> Self {
        Self {
            path,
            run_args,
            tract_graph: None,
            symbols: None,
            nodes: None,
            mapper: None,
            inputs: None,
            outputs: None,
            original_input_dims: HashMap::new(),
            original_output_dims: HashMap::new(),
        }
    }

    #[tracing::instrument(name = "ModelLoader::load_onnx_using_tract", skip_all)]
    /// Loads and prepares the ONNX model using Tract.
    ///
    /// This step:
    /// - Reads the ONNX file
    /// - Concretizes dynamic dimensions
    /// - Converts to a typed graph
    pub fn load_onnx_using_tract(mut self) -> Self {
        let (tract_graph, symbols) = Model::load_onnx_using_tract(self.path, self.run_args);
        self.tract_graph = Some(tract_graph);
        self.symbols = Some(symbols);
        self
    }

    /// Parses the Tract graph into ComputationNode representation.
    ///
    /// Converts Tract's internal graph structure into our domain-specific
    /// node representation.
    #[tracing::instrument(name = "ModelLoader::parse_nodes", skip_all)]
    pub fn parse_nodes(mut self) -> Self {
        let tract_graph = self
            .tract_graph
            .as_ref()
            .expect("load_onnx_using_tract must be called first");
        let symbols = self
            .symbols
            .as_ref()
            .expect("load_onnx_using_tract must be called first");

        let (nodes, mapper) = Model::nodes_from_graph(tract_graph, self.run_args, symbols);
        self.nodes = Some(nodes);
        self.mapper = Some(mapper);
        self
    }

    #[tracing::instrument(name = "ModelLoader::collect_input_nodes", skip_all)]
    /// Collects indices of all input nodes from the parsed computation nodes.
    pub fn collect_input_nodes(mut self) -> Self {
        let nodes = self
            .nodes
            .as_ref()
            .expect("parse_nodes must be called first");
        let inputs = Model::collect_input_nodes(nodes);
        self.inputs = Some(inputs);
        self
    }

    #[tracing::instrument(name = "ModelLoader::collect_outputs", skip_all)]
    /// Collects output connections from the Tract graph.
    pub fn collect_outputs(mut self) -> Self {
        let tract_graph = self
            .tract_graph
            .as_ref()
            .expect("load_onnx_using_tract must be called first");
        let mapper = self
            .mapper
            .as_ref()
            .expect("parse_nodes must be called first");
        let outputs = Model::collect_outputs(tract_graph, mapper);
        self.outputs = Some(outputs);
        self
    }

    #[tracing::instrument(name = "ModelLoader::prune", skip_all)]
    /// Prunes unused nodes from the computation graph and remaps indices.
    ///
    /// This step removes nodes that are not reachable from any output node
    /// and reassigns contiguous indices to the remaining nodes.
    ///
    /// # Panics
    /// Panics if parse_nodes, collect_input_nodes, or collect_outputs have not been called.
    pub fn prune(mut self) -> Self {
        let nodes = self
            .nodes
            .as_mut()
            .expect("parse_nodes must be called first");
        let inputs = self
            .inputs
            .as_mut()
            .expect("collect_input_nodes must be called first");
        let outputs = self
            .outputs
            .as_mut()
            .expect("collect_outputs must be called first");

        Model::prune_unused_nodes(nodes, inputs, outputs);
        self
    }

    #[tracing::instrument(name = "ModelLoader::pad", skip_all)]
    /// Pads all constant tensors and node output dimensions to the next power of 2.
    ///
    /// This step:
    /// - Stores original input and output dimensions for later use during execution
    /// - Pads all Constant operator tensors using Tensor::pad_next_power_of_two()
    /// - Updates all node output_dims to their padded equivalents
    ///
    /// # Panics
    /// Panics if parse_nodes, collect_input_nodes, or collect_outputs have not been called.
    #[tracing::instrument(name = "ModelLoader::pad", skip_all)]
    pub fn pad(mut self) -> Self {
        let nodes = self
            .nodes
            .as_mut()
            .expect("parse_nodes must be called first");
        let inputs = self
            .inputs
            .as_ref()
            .expect("collect_input_nodes must be called first");
        let outputs = self
            .outputs
            .as_ref()
            .expect("collect_outputs must be called first");

        // Store original input dimensions
        for &input_idx in inputs {
            if let Some(node) = nodes.get(&input_idx) {
                self.original_input_dims
                    .insert(input_idx, node.output_dims.clone());
            }
        }

        // Store original output dimensions
        for &output_idx in outputs {
            if let Some(node) = nodes.get(&output_idx) {
                self.original_output_dims
                    .insert(output_idx, node.output_dims.clone());
            }
        }

        // Pad all nodes: constant tensors, operator-internal shapes, and output dimensions
        for node in nodes.values_mut() {
            // Pad constant tensors
            if let Operator::Constant(constant) = &mut node.operator {
                constant.0.pad_next_power_of_two();
            }

            // Pad operator-internal shapes that must stay consistent with padded dimensions
            match &mut node.operator {
                Operator::Broadcast(broadcast) => {
                    broadcast.shape = Model::pad_dims_to_power_of_2(&broadcast.shape);
                }
                Operator::Reshape(reshape) => {
                    reshape.shape = Model::pad_dims_to_power_of_2(&reshape.shape);
                }
                Operator::IsNan(is_nan) => {
                    is_nan.out_dims = Model::pad_dims_to_power_of_2(&is_nan.out_dims);
                }
                _ => {}
            }

            // Pad output dimensions for all nodes
            node.output_dims = Model::pad_dims_to_power_of_2(&node.output_dims);
        }

        self
    }

    /// Builds the final Model with the computation graph.
    ///
    /// # Panics
    /// Panics if any of the required builder steps were not called.
    #[tracing::instrument(name = "ModelLoader::build", skip_all)]
    pub fn build(self) -> Model {
        Model {
            graph: ComputationGraph {
                nodes: self.nodes.expect("parse_nodes must be called"),
                inputs: self.inputs.expect("collect_input_nodes must be called"),
                outputs: self.outputs.expect("collect_outputs must be called"),
                original_input_dims: self.original_input_dims,
                original_output_dims: self.original_output_dims,
            },
        }
    }
}

/// Builder for loading and configuring ONNX models with Tract.
///
/// This builder provides a fluent interface for loading ONNX models with various
/// configuration options at the Tract level. It encapsulates the multi-step process of:
/// - Loading the ONNX file
/// - Concretizing dynamic dimensions
/// - Relaxing output constraints
/// - Building symbol values
/// - Converting to typed model
///
/// # Example
/// ```ignore
/// let (graph, symbols) = TractModelLoader::new("model.onnx")
///     .with_variables(&variables)
///     .with_run_args(&run_args)
///     .build();
/// ```
pub struct TractModelLoader<P: AsRef<Path>> {
    path: P,
    variables: Option<HashMap<String, usize>>,
    run_args: Option<RunArgs>,
    relax_outputs: bool,
}

impl<P: AsRef<Path>> TractModelLoader<P> {
    /// Creates a new TractModelLoader for the given ONNX file path.
    ///
    /// # Arguments
    /// * `path` - Path to the ONNX model file
    pub fn new(path: P) -> Self {
        Self {
            path,
            variables: None,
            run_args: None,
            relax_outputs: true,
        }
    }

    /// Sets the variable map for concretizing dynamic dimensions.
    ///
    /// # Arguments
    /// * `variables` - Map of variable names (e.g., "batch_size") to their concrete values
    pub fn with_variables(mut self, variables: &HashMap<String, usize>) -> Self {
        self.variables = Some(variables.clone());
        self
    }

    /// Sets the run arguments for the model.
    ///
    /// # Arguments
    /// * `run_args` - Runtime arguments including variables and scaling parameters
    pub fn with_run_args(mut self, run_args: &RunArgs) -> Self {
        self.run_args = Some(run_args.clone());
        self
    }

    /// Controls whether output constraints should be relaxed (default: true).
    ///
    /// # Arguments
    /// * `relax` - If true, output facts will be reset to defaults for flexible inference
    pub fn relax_outputs(mut self, relax: bool) -> Self {
        self.relax_outputs = relax;
        self
    }

    /// Builds the typed model by executing all configured steps.
    ///
    /// # Returns
    /// A tuple containing the typed computation graph and symbol values.
    ///
    /// # Panics
    /// Panics if:
    /// - The file cannot be opened
    /// - The ONNX model cannot be parsed
    /// - Variables are not provided but the model has dynamic dimensions
    /// - Any preparation step fails
    #[tracing::instrument(name = "TractModelLoader::build", skip_all)]
    pub fn build(self) -> GraphLoadResult {
        // Step 1: Load ONNX file
        let mut model = self.load_onnx_file();

        // Step 2: Build symbol values from run args
        // Build symbol_values mapping first, before processing inputs
        let symbol_values = self.build_symbol_values(&model);

        // Step 3: Concretize dynamic input dimensions if variables provided
        if let Some(ref variables) = self.variables {
            Self::concretize_input_dimensions(&mut model, variables, &symbol_values);
        }

        // Step 4: Relax output constraints if configured
        if self.relax_outputs {
            Self::relax_output_facts(&mut model);
        }

        // Step 5: Convert to typed model
        let typed_model = Self::convert_to_typed(model, &symbol_values);

        (typed_model, symbol_values)
    }

    /// Loads the ONNX file into a Tract inference model.
    #[tracing::instrument(name = "TractModelLoader::load_onnx_file", skip_all)]
    fn load_onnx_file(&self) -> InferenceModel {
        tract_onnx::onnx().model_for_path(&self.path).unwrap()
    }

    /// Concretizes dynamic input dimensions using provided variables.
    #[tracing::instrument(name = "TractModelLoader::concretize_input_dimensions", skip_all)]
    fn concretize_input_dimensions(
        model: &mut InferenceModel,
        variables: &HashMap<String, usize>,
        symbol_values: &SymbolValues,
    ) {
        Model::concretize_model_input_dims_from_variables(model, variables, symbol_values);
    }

    /// Relaxes output constraints for flexible inference.
    fn relax_output_facts(model: &mut InferenceModel) {
        Model::relax_output_facts(model);
    }

    /// Builds symbol values from run arguments.
    fn build_symbol_values(&self, model: &InferenceModel) -> SymbolValues {
        if let Some(ref run_args) = self.run_args {
            Model::build_symbol_values_from_run_args(model, run_args)
        } else {
            SymbolValues::default()
        }
    }

    /// Converts inference model to typed model with concrete dimensions.
    #[tracing::instrument(name = "TractModelLoader::convert_to_typed", skip_all)]
    fn convert_to_typed(
        model: InferenceModel,
        symbol_values: &SymbolValues,
    ) -> Graph<TypedFact, Box<dyn TypedOp>> {
        Model::model_into_typed(model, symbol_values)
    }
}
