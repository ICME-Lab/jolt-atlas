use super::node::*;
use crate::{
    decode_node,
    graph::{
        input::GraphData, tracer::Tracer, utilities::node_output_shapes, vars::VarScales,
        GraphError,
    },
    ops::{poly::PolyOp, Input, Op, Unknown},
    tensor::Tensor,
    RunArgs,
};
use log::{debug, info, trace};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
};
use tabled::Table;
use tract_onnx::{
    prelude::{
        tract_itertools::Itertools, Framework, Graph, InferenceFact, InferenceModelExt,
        Node as OnnxNode, SymbolValues, TDim, TypedFact, TypedOp,
    },
    tract_core::internal::DatumType,
    tract_hir::ops::scan::Scan,
};

/// A struct for loading from an Onnx file and converting a computational graph to a
/// circuit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Model {
    pub graph: ParsedNodes,
    pub tracer: Tracer,
}

impl Model {
    /// Creates a `Model` from a specified path to an Onnx file.
    /// # Arguments
    /// * `reader` - A reader for an Onnx file.
    /// * `run_args` - [RunArgs]
    pub fn new(reader: &mut dyn std::io::Read, run_args: &RunArgs) -> Self {
        let graph = Self::load_onnx_model(reader, run_args);
        let om = Model {
            graph,
            tracer: Tracer::default(),
        };
        info!("\n {}", om.table_nodes());
        om
    }

    /// Executes a forward pass through the parsed ONNX model using provided input tensors.
    ///
    /// # Purpose
    /// This function simulates running the ONNX model on input data, producing the model's outputs
    /// as if it were being executed in a standard inference engine. It is essential for testing,
    /// debugging, and validating the model conversion pipeline, as well as for extracting
    /// intermediate and final outputs for further processing or verification.
    ///
    /// # Arguments
    /// * `model_inputs` - A slice of [`Tensor<Fp>`] representing the input data for the model.
    ///   Each tensor in this slice should correspond to one of the model's input nodes, and must
    ///   have the correct shape and data type expected by the model. The order of tensors must
    ///   match the order of the model's input nodes.
    ///
    /// # Returns
    /// Returns a [`Result<ForwardResult, Box<dyn Error>>`] where:
    /// - `ForwardResult` contains:
    ///     - `outputs`: The output tensors produced by the model (in the order of the model's outputs).
    ///     - `max_lookup_inputs`/`min_lookup_inputs`: The maximum and minimum values encountered as inputs to any lookup operation during execution (useful for quantization or table sizing).
    /// - If an error occurs (e.g., shape mismatch, missing node, or execution failure), returns an error describing the issue.
    ///
    /// # How It Works (Step by Step)
    /// 1. **Prepare Results Map:** Initializes a map to store the output tensors of each node as the graph is executed.
    /// 2. **Reshape and Insert Inputs:** Reshapes each provided input tensor to match the expected input shape, and inserts it into the results map keyed by the input node index.
    /// 3. **Node Execution Loop:** Iterates through each node in the graph in topological order:
    ///     - Gathers the required input tensors for the node from the results map.
    ///     - Executes the node's operation (or recursively executes subgraphs for control flow nodes).
    ///     - Tracks min/max values for lookup operations.
    ///     - Stores the node's output(s) in the results map.
    /// 4. **Collect Outputs:** After all nodes have been executed, collects the output tensors corresponding to the model's output nodes.
    /// 5. **Return Results:** Packages the outputs and lookup statistics into a `ForwardResult` and returns it.
    ///
    /// # When to Use
    /// Use this function whenever you need to:
    /// - Simulate model inference on sample data (e.g., for testing or debugging).
    /// - Validate that the model conversion from ONNX to the internal graph representation is correct.
    /// - Extract intermediate or final outputs for further analysis.
    ///
    /// # Example Usage
    /// ```ignore
    /// let model = Model::new(&mut onnx_file, &run_args);
    /// let input_tensors = vec![...]; // Prepare input tensors matching model's input shapes
    /// let result = model.forward(&input_tensors)?;
    /// println!("Model outputs: {:?}", result.outputs);
    /// ```
    ///
    /// # Notes
    /// - Input tensors must be in the correct order and shape.
    /// - This function does not perform any hardware-accelerated inference; it executes the model using the internal Rust implementation.
    /// - Handles both standard nodes and subgraphs (e.g., for ONNX Scan/Loop constructs).
    pub fn forward(&self, model_inputs: &[Tensor<i32>]) -> Result<ForwardResult, Box<dyn Error>> {
        // A map that stores the output tensors of each node in the computation graph.
        //
        // # Purpose
        // `results` is used to keep track of the intermediate and final outputs produced by each node
        // (identified by their unique index) during the execution of the model. The key is a reference to
        // the node's index (`&usize`), and the value is a vector of `Tensor<Fp>`, representing the output
        // tensors generated by that node.
        //
        // # Why we need `results`
        // In a computational graph, nodes may depend on the outputs of previous nodes. By storing the
        // results in a `BTreeMap`, we can efficiently retrieve the outputs of any node as needed for
        // subsequent computations. This structure also ensures deterministic iteration order, which can be
        // important for reproducibility and debugging.
        //
        // # Usage
        // - When a node is executed, its output tensors are inserted into `results` under its index.
        // - When another node requires the output of a previous node, it can look up the corresponding
        //   entry in `results`.
        // - After the entire graph has been executed, `results` contains the outputs of all nodes, which
        //   can be used for further processing or for extracting the final model outputs.
        //
        // DESIGN NOTE: Why use `BTreeMap<&usize, Vec<Tensor<Fp>>>` for `results`?
        //
        // 1. Why a BTreeMap?
        //    - Deterministic ordering: BTreeMap iterates in sorted order, which helps with reproducibility and debugging.
        //    - Efficient lookup: We need to quickly retrieve the output(s) of any node by its index.
        //    - Nodes are indexed by their unique usize index in the graph.
        //
        // 2. Why does the value type store a Vec<Tensor<Fp>> instead of just Tensor<Fp>?
        //    - Some nodes (especially subgraphs or nodes with multiple outputs) can produce multiple output tensors.
        //    - The Vec allows us to store all outputs for a node, indexed by their outlet (output slot).
        //    - For most nodes, this Vec will have a single element, but for nodes with multiple outputs, each output is stored at its respective index.
        //
        // 3. How do you use this map to get the output of a node?
        //    - To get the output tensor(s) of node 10, you would do:
        //        `let outputs = results.get(&10);`
        //      - If you want the first output (most common case): `let output = &outputs[0];`
        //      - If the node has multiple outputs, you can access them by their outlet index: `let output = &outputs[outlet_index];`
        //
        // 4. Why is the key a reference (&usize) instead of just usize?
        //    - This is because we often have references to node indices from elsewhere in the graph, and using references avoids unnecessary copies.
        //    - BTreeMap allows lookups with either &usize or usize, but here the code is consistent with using references.
        //
        // 5. Summary:
        //    - This structure allows us to efficiently store and retrieve all intermediate and final outputs of the computation graph,
        //      supporting both single-output and multi-output nodes, and ensuring deterministic iteration order.
        let mut results: BTreeMap<&usize, Vec<Tensor<i32>>> = BTreeMap::new();
        let mut max_lookup_inputs = 0;
        let mut min_lookup_inputs = 0;
        // Retrieves the shapes of all input tensors for the current computational graph.
        //
        // # Intent
        // This line obtains the dimensions (shapes) of each input tensor that the model expects,
        // which is essential for validating input data, constructing subsequent layers, and ensuring
        // compatibility throughout the model's execution.
        //
        // # Why it's needed
        // Knowing the input shapes is crucial for tasks such as input validation, dynamic graph construction,
        // and for informing downstream operations about the expected data structure. It helps prevent
        // runtime errors due to shape mismatches and is often required when exporting, tracing, or
        // transforming the model.
        //
        // # How it's used
        // The returned `input_shapes` value is typically used to:
        // - Validate that provided input data matches the model's requirements.
        // - Dynamically allocate memory or buffers for inputs.
        // - Inform other components or tools (e.g., ONNX exporters, tracers) about the model's input signature.
        //
        // # When it's used
        // This is usually called during model initialization, tracing, or before running inference,
        // whenever the input specification of the model needs to be known or verified.
        let input_shapes = self.graph.input_shapes()?;
        // Insert model inputs into the results map after reshaping them to match expected input shapes.
        //
        // # Why this code is needed
        // The model's forward execution relies on a map (`results`) that holds the output tensors of each node.
        // For input nodes, we must provide the actual input tensors supplied by the user, but these may need to be reshaped
        // to match the expected dimensions as defined by the model (e.g., to handle batch size or symbolic shapes).
        //
        // # Intent
        // This code ensures that each input tensor provided to the model is reshaped to the correct shape and then
        // inserted into the results map under the corresponding input node index. This allows subsequent nodes in the
        // computation graph to retrieve the correct input data during execution.
        //
        // # How it works
        // - Iterates over all input node indices and their corresponding position in the input tensor list.
        // - For each input:
        //     - Clones the provided input tensor.
        //     - Reshapes it to match the expected shape for that input node.
        //     - Inserts the reshaped tensor into the `results` map under the input node's index.
        for (i, input_idx) in self.graph.inputs.iter().enumerate() {
            let mut input = model_inputs[i].clone();
            input.reshape(&input_shapes[i])?;
            results.insert(input_idx, vec![input]);
        }

        // --- Fetch Decode Execute ---
        for (idx, n) in self.graph.nodes.iter() {
            // Fetch and Decode
            let mut inputs = Self::node_inputs(idx, n, &results)?;
            {
                let instr = decode_node((idx, n));
                let mut tracer_inputs = inputs.clone();
                if n.is_pow2() {
                    tracer_inputs.push(inputs[0].clone());
                }
                self.tracer.capture_pre_state(instr.clone(), tracer_inputs);
            }
            if n.is_lookup() {
                Self::lookup_check(&inputs, &mut max_lookup_inputs, &mut min_lookup_inputs)?;
            }
            match n {
                NodeType::Node(n) => {
                    // Execute
                    let mut res = Op::<i32>::f(&n.opkind, &inputs)?;
                    res.output.reshape(&n.out_dims)?;
                    // see if any of the intermediate lookup calcs are the max
                    if !res.intermediate_lookups.is_empty() {
                        Self::lookup_check(
                            &res.intermediate_lookups,
                            &mut max_lookup_inputs,
                            &mut min_lookup_inputs,
                        )?;
                    }
                    debug!(
                        "------------ output node int {}: {} \n  ------------ scale: {}",
                        idx,
                        res.output.show(),
                        n.out_scale
                    );
                    results.insert(idx, vec![res.output.clone()]);
                    self.tracer.capture_post_state(res.output);
                }
                // --- SubGraph Node Execution ---
                //
                // This block handles the execution of a subgraph node (NodeType::SubGraph), which is fundamentally
                // different from executing a standard node (NodeType::Node). Subgraphs are used for control flow
                // constructs like ONNX Scan, Loop, or custom nested models, where a portion of the graph is executed
                // multiple times with varying inputs (e.g., sequence processing, RNNs).
                //
                // Why is this needed?
                // - Standard nodes perform a single computation given their inputs.
                // - Subgraph nodes encapsulate an entire model that must be executed repeatedly, often with sliced or
                //   stateful inputs, and may have complex input/output mappings.
                // - This code ensures correct iteration, input slicing, state management, and output collection for
                //   subgraph execution.
                //
                // Intent & Purpose:
                // - To simulate the iterative execution of a subgraph as required by ONNX control flow semantics.
                // - To correctly map parent graph inputs to subgraph inputs, handle state variables, and collect outputs.
                // - To recursively execute the subgraph and aggregate results, including lookup statistics.
                //
                // How it works (Step-by-Step):
                // 1. Clone the original inputs and input mappings for reference.
                // 2. Determine the number of iterations required by inspecting the input mappings and dimensions.
                //    (For example, if an input is chunked along an axis, the number of iterations is dim_size / chunk_size.)
                // 3. For each iteration:
                //    a. Slice or update the inputs as specified by the input mappings (e.g., Stacked inputs get a chunk).
                //    b. Recursively call `model.forward(&inputs)` to execute the subgraph for this iteration.
                //    c. Track min/max lookup values from subgraph execution for quantization/table sizing.
                //    d. Map subgraph outputs back to parent graph outputs using output mappings, handling stacking
                //       (concatenation) and state variables.
                //    e. Update stateful inputs for the next iteration using output states.
                // 4. After all iterations, insert the aggregated outputs into the parent graph's results map.
                //
                // Key differences from normal node execution:
                // - Iterative: Subgraphs may execute multiple times, while normal nodes execute once.
                // - Input/Output Mapping: Inputs/outputs may be sliced, stacked, or carried as state, requiring complex mapping.
                // - Recursion: Subgraph execution is recursive, calling `forward` on the sub-model.
                // - State Management: Handles state variables that persist across iterations.
                // - Output Aggregation: May need to concatenate outputs across iterations (stacked outputs).
                //
                // Gotchas:
                // - Input slicing must match the expected chunk size and axis; mismatches can cause runtime errors.
                // - State variables must be correctly updated between iterations.
                // - Output mappings can be complex; ensure correct outlet and axis handling.
                // - Recursion can lead to stack overflows if subgraphs are deeply nested.
                //
                // Summary:
                // This code is essential for supporting ONNX models with control flow, enabling correct execution of
                // iterative constructs and nested graphs. It ensures that subgraph semantics are faithfully reproduced,
                // including input slicing, state management, and output aggregation.
                NodeType::SubGraph {
                    model,
                    output_mappings,
                    input_mappings,
                    inputs: input_tuple,
                    ..
                } => {
                    let orig_inputs = inputs.clone();
                    let input_mappings = input_mappings.clone();

                    let input_dims = inputs.iter().map(|inp| inp.dims());
                    let num_iter = number_of_iterations(&input_mappings, input_dims.collect());
                    debug!(
                        "{} iteration(s) in a subgraph with inputs {:?} and sources {:?}",
                        num_iter, input_tuple, model.graph.inputs
                    );
                    debug!("input_mappings: {input_mappings:?}",);
                    let mut full_results: Vec<Tensor<i32>> = vec![];
                    for i in 0..num_iter {
                        // replace the Stacked input with the current chunk iter
                        for ((mapping, inp), og_input) in
                            input_mappings.iter().zip(&mut inputs).zip(&orig_inputs)
                        {
                            if let InputMapping::Stacked { axis, chunk } = mapping {
                                let start = i * chunk;
                                let end = (i + 1) * chunk;
                                let t = crate::tensor::ops::slice(og_input, axis, &start, &end)?;
                                *inp = t;
                            }
                        }
                        let res = model.forward(&inputs)?;
                        // recursively get the max lookup inputs for subgraphs
                        max_lookup_inputs = max_lookup_inputs.max(res.max_lookup_inputs);
                        min_lookup_inputs = min_lookup_inputs.min(res.min_lookup_inputs);
                        let mut outlets = BTreeMap::new();
                        for (mappings, outlet_res) in output_mappings.iter().zip(res.outputs) {
                            for mapping in mappings {
                                match mapping {
                                    OutputMapping::Single { outlet, .. } => {
                                        outlets.insert(outlet, outlet_res.clone());
                                    }
                                    OutputMapping::Stacked { outlet, axis, .. } => {
                                        if !full_results.is_empty() {
                                            let stacked_res = crate::tensor::ops::concat(
                                                &[&full_results[*outlet], &outlet_res],
                                                *axis,
                                            )?;

                                            outlets.insert(outlet, stacked_res);
                                        } else {
                                            outlets.insert(outlet, outlet_res.clone());
                                        }
                                    }
                                }
                            }
                        }
                        full_results = outlets.into_values().collect_vec();
                        let output_states = output_state_idx(output_mappings);
                        let input_states = input_state_idx(&input_mappings);
                        assert_eq!(input_states.len(), output_states.len());
                        for (input_idx, output_idx) in input_states.iter().zip(output_states) {
                            inputs[*input_idx] = full_results[output_idx].clone();
                        }
                    }
                    trace!(
                        "------------ output subgraph node {}: {:?}",
                        idx,
                        full_results.iter().map(|x| x.show()).collect_vec()
                    );
                    results.insert(idx, full_results);
                }
            }
        }
        // Collects the output tensors of the model from the results map.
        //
        // # Why we need this code
        // After executing all nodes in the computational graph, we need to extract the final outputs of the model.
        // The model's outputs are defined as specific nodes (and their output slots) in the graph.
        // This code gathers those outputs from the `results` map, which contains the outputs of every node.
        //
        // # What it does
        // - Iterates over the list of output node indices and outlet slots (`self.graph.outputs`).
        // - For each output, retrieves the corresponding tensor from the `results` map.
        // - Collects all output tensors into a vector, preserving the order defined by the model's outputs.
        // - Wraps the outputs and lookup statistics into a `ForwardResult` struct.
        //
        // # How it works
        // - Uses `.map()` to iterate over each output node and outlet.
        // - Looks up the node's outputs in the `results` map using the node index.
        // - Selects the correct output tensor by indexing into the vector with the outlet index.
        // - Handles missing results by returning a `GraphError`.
        // - Collects all outputs into a `Vec<Tensor<Fp>>`.
        //
        // # Intent
        // The intent is to provide the user with the final outputs of the model in the correct order,
        // as well as any statistics (such as min/max lookup inputs) gathered during execution.
        let output_nodes = self.graph.outputs.iter();
        debug!(
            "model outputs are nodes: {:?}",
            output_nodes.clone().collect_vec()
        );
        let outputs = output_nodes
            .map(|(idx, outlet)| {
                Ok(results.get(&idx).ok_or(GraphError::MissingResults)?[*outlet].clone())
            })
            .collect::<Result<Vec<_>, GraphError>>()?;
        Ok(ForwardResult {
            outputs,
            max_lookup_inputs,
            min_lookup_inputs,
        })
    }

    /// Gathers the input tensors required for the current node's execution.
    ///
    /// # Intent
    /// This code block prepares the list of input tensors (`inputs`) that will be fed into the current node's operation.
    /// For each node in the graph, we must collect its input tensors from the results of previously executed nodes.
    ///
    /// # Why this code is needed
    /// In a computational graph, each node may depend on the outputs of other nodes (its inputs).
    /// Before executing a node, we must gather all its required input tensors in the correct order.
    /// This ensures that the node receives the correct data for computation, and is essential for correct model execution.
    ///
    /// # How it works
    /// - Initializes an empty `inputs` vector.
    /// - If the current node is an input node, retrieves its tensor from the `results` map.
    /// - Otherwise, iterates over the node's input connections (each a tuple of node index and outlet).
    ///   For each input:
    ///     - Looks up the output tensor of the source node from the `results` map.
    ///     - Pushes the required output (by outlet index) into the `inputs` vector.
    ///   - If any required input is missing, returns an error.
    ///
    /// # What it does
    /// After this block, `inputs` contains the tensors that should be passed to the current node's operation,
    /// in the order expected by the node. This enables the subsequent execution of the node's computation.
    fn node_inputs(
        idx: &usize,
        n: &NodeType,
        results: &BTreeMap<&usize, Vec<Tensor<i32>>>,
    ) -> Result<Vec<Tensor<i32>>, Box<dyn Error>> {
        let mut inputs = vec![];
        if n.is_input() {
            let t = results.get(idx).ok_or(GraphError::MissingResults)?[0].clone();
            inputs.push(t);
        } else {
            for (idx, outlet) in n.inputs().iter() {
                match results.get(&idx) {
                    Some(value) => inputs.push(value[*outlet].clone()),
                    None => return Err(Box::new(GraphError::MissingNode(*idx))),
                }
            }
        };
        debug!("executing {}: {}", idx, n.as_str());
        debug!("dims: {:?}", n.out_dims());
        debug!(
            "input_dims: {:?}",
            inputs.iter().map(|x| x.dims()).collect::<Vec<_>>()
        );
        Ok(inputs)
    }

    fn lookup_check(
        inputs: &[Tensor<i32>],
        max_lookup_inputs: &mut i32,
        min_lookup_inputs: &mut i32,
    ) -> Result<(), Box<dyn Error>> {
        let (mut min, mut max) = (0, 0);
        for i in inputs {
            max = max.max(i.iter().copied().max().ok_or("missing max")?);
            min = min.min(i.iter().copied().min().ok_or("missing min")?);
        }
        *max_lookup_inputs = (*max_lookup_inputs).max(max);
        *min_lookup_inputs = (*min_lookup_inputs).min(min);
        debug!("max lookup inputs: {max}");
        debug!("min lookup inputs: {min}");
        Ok(())
    }

    /// Loads an Onnx model from a specified path.
    /// # Arguments
    /// * `reader` - A reader for an Onnx file.
    /// * `scale` - The scale to use for quantization.
    /// * `public_params` - Whether to make the params public.
    fn load_onnx_model(reader: &mut dyn std::io::Read, run_args: &RunArgs) -> ParsedNodes {
        let start_time = instant::Instant::now();
        let (model, symbol_values) = Self::load_onnx_using_tract(reader, run_args);
        let scales = VarScales::from_args(run_args);
        let nodes = Self::nodes_from_graph(&model, run_args, &scales, &symbol_values, None, None);
        debug!("\n {model}",);
        let output_node = (*nodes.iter().last().unwrap().0, 0);
        let parsed_nodes = ParsedNodes {
            nodes,
            inputs: vec![0],
            outputs: vec![output_node],
        };
        let duration = start_time.elapsed();
        trace!("model loading took: {duration:?}",);
        parsed_nodes
    }

    /// Loads an Onnx model from a specified path.
    /// # Arguments
    /// * `reader` - A reader for an Onnx file.
    /// * `scale` - The scale to use for quantization.
    /// * `public_params` - Whether to make the params public.
    fn load_onnx_using_tract(
        reader: &mut dyn std::io::Read,
        run_args: &RunArgs,
    ) -> (Graph<TypedFact, Box<dyn TypedOp>>, SymbolValues) {
        let mut model = tract_onnx::onnx().model_for_read(reader).unwrap();
        let variables: std::collections::HashMap<String, usize> =
            std::collections::HashMap::from_iter(run_args.variables.clone());
        for (i, id) in model.clone().inputs.iter().enumerate() {
            let input = model.node_mut(id.node);
            let mut fact: InferenceFact = input.outputs[0].fact.clone();
            for (i, x) in fact.clone().shape.dims().enumerate() {
                use tract_onnx::tract_hir::infer::GenericFactoid;
                if matches!(x, GenericFactoid::Any) {
                    let batch_size = variables.get("batch_size").unwrap();
                    fact.shape
                        .set_dim(i, tract_onnx::prelude::TDim::Val(*batch_size as i64));
                }
            }
            model.set_input_fact(i, fact).unwrap();
        }
        for (i, _) in model.clone().outputs.iter().enumerate() {
            model.set_output_fact(i, InferenceFact::default()).unwrap();
        }
        let mut symbol_values = SymbolValues::default();
        for (symbol, value) in run_args.variables.iter() {
            use log::info;
            let symbol = model.symbols.sym(symbol);
            symbol_values = symbol_values.with(&symbol, *value as i64);
            info!("set {symbol} to {value}");
            println!("set {symbol} to {value}");
        }

        // Note: do not optimize the model, as the layout will depend on
        // underlying hardware
        let mut typed_model = model
            .into_typed()
            .unwrap()
            .concretize_dims(&symbol_values)
            .unwrap()
            .into_decluttered()
            .unwrap();
        // concretize constants
        for node in typed_model.eval_order().unwrap() {
            let node = typed_model.node_mut(node);
            if node.op_is::<tract_onnx::tract_hir::ops::konst::Const>() {
                // map option to err
                let op = node
                    .op_as_mut::<tract_onnx::tract_hir::ops::konst::Const>()
                    .unwrap();
                // get inner value to Arc<Tensor>
                let constant = op.0.as_ref();
                if constant.datum_type() == DatumType::TDim {
                    // Generally a shape or hyperparam
                    use tract_onnx::prelude::TDim;
                    let vec = constant
                        .as_slice::<tract_onnx::prelude::TDim>()
                        .unwrap()
                        .to_vec();
                    let data: Vec<TDim> = vec.into_iter().map(|x| x.eval(&symbol_values)).collect();
                    unsafe {
                        let bytes = std::slice::from_raw_parts(
                            data.as_ptr() as *const u8,
                            data.len() * DatumType::TDim.size_of(),
                        );
                        op.0 = std::sync::Arc::new(
                            tract_onnx::prelude::Tensor::from_raw_dt(
                                DatumType::TDim,
                                constant.shape(),
                                bytes,
                            )
                            .unwrap(),
                        );
                    }
                }
            }
        }
        (typed_model, symbol_values)
    }

    /// Creates ezkl nodes from a tract graph
    /// # Arguments
    /// * `graph` - A tract graph.
    /// * `run_args` - [RunArgs]
    /// * `visibility` - Which inputs to the model are public and private (params,
    // inputs, outputs) using [VarVisibility].
    /// * `input_scales` - The scales of
    // the model's inputs.
    pub fn nodes_from_graph(
        graph: &Graph<TypedFact, Box<dyn TypedOp>>,
        _run_args: &RunArgs,
        scales: &VarScales,
        symbol_values: &SymbolValues,
        override_input_scales: Option<Vec<crate::Scale>>,
        override_output_scales: Option<HashMap<usize, crate::Scale>>,
    ) -> BTreeMap<usize, NodeType> {
        let mut nodes = BTreeMap::<usize, NodeType>::new();
        // Insert dummy node at idx 0, which will be replaced with the actual Input node later.
        nodes.insert(0, NodeType::Node(Node::default()));

        let mut input_idx = 0;
        let mut remappings = BTreeMap::<usize, usize>::new();

        for (i, n) in graph.nodes.iter().enumerate() {
            match n.op().downcast_ref::<Scan>() {
                Some(b) => {
                    let subgraph_node = Self::process_subgraph_node(
                        i,
                        n,
                        b,
                        &nodes,
                        _run_args,
                        scales,
                        symbol_values,
                    );
                    nodes.insert(i, subgraph_node);
                }
                None => {
                    let mut node =
                        Node::new(n.clone(), &mut nodes, scales, symbol_values, &remappings)
                            .expect("Failed to create node");

                    if node.opkind.requires_shape_equality() {
                        node.homogenize_input_shapes(&mut nodes);
                    }

                    Self::apply_input_scale_override(
                        &mut node,
                        &override_input_scales,
                        &mut input_idx,
                    );
                    Self::apply_output_scale_override(&mut node, i, &override_output_scales);

                    Self::handle_node_insertion(&mut nodes, &mut remappings, i, node);
                }
            }
        }

        Self::remove_unused_nodes(&mut nodes);
        Self::ensure_consecutive_indices(&mut nodes);

        nodes
    }

    /// Processes a subgraph node (Scan operation) and returns the corresponding NodeType
    fn process_subgraph_node(
        idx: usize,
        node: &OnnxNode<TypedFact, Box<dyn TypedOp>>,
        scan_op: &Scan,
        nodes: &BTreeMap<usize, NodeType>,
        run_args: &RunArgs,
        scales: &VarScales,
        symbol_values: &SymbolValues,
    ) -> NodeType {
        let model = scan_op.body.clone();
        let input_scales = node
            .inputs
            .iter()
            .map(|i| nodes.get(&i.node).unwrap().out_scales()[0])
            .collect::<Vec<_>>();

        let input_mappings = Self::build_input_mappings(&scan_op.input_mapping);
        let output_mappings = Self::build_output_mappings(&scan_op.output_mapping);

        let input_state_idx = input_state_idx(&input_mappings);
        let output_state_idx = output_state_idx(&output_mappings);

        let output_scale_override = Self::build_output_scale_override(
            &input_state_idx,
            output_state_idx,
            &input_scales,
            &output_mappings,
            &scan_op.body,
        );

        let subgraph_nodes = Self::nodes_from_graph(
            &model,
            run_args,
            scales,
            symbol_values,
            Some(input_scales.clone()),
            Some(output_scale_override),
        );

        let subgraph = ParsedNodes {
            nodes: subgraph_nodes,
            inputs: model.inputs.iter().map(|o| o.node).collect(),
            outputs: model.outputs.iter().map(|o| (o.node, o.slot)).collect(),
        };

        let subgraph_model = Model {
            graph: subgraph,
            tracer: Tracer::default(),
        };

        let out_dims = node_output_shapes(node, symbol_values).unwrap();
        let out_scales = Self::extract_output_scales(&scan_op.output_mapping, &subgraph_model);

        NodeType::SubGraph {
            model: subgraph_model,
            inputs: node.inputs.iter().map(|i| (i.node, i.slot)).collect_vec(),
            idx,
            output_mappings,
            input_mappings,
            out_dims,
            out_scales,
        }
    }

    /// Builds input mappings from ONNX scan input mapping
    fn build_input_mappings(
        input_mapping: &[tract_onnx::tract_hir::ops::scan::InputMapping],
    ) -> Vec<InputMapping> {
        let mut input_mappings = vec![];
        for mapping in input_mapping {
            match mapping {
                tract_onnx::tract_hir::ops::scan::InputMapping::Scan(info) => {
                    input_mappings.push(InputMapping::Stacked {
                        axis: info.axis,
                        chunk: info.chunk as usize,
                    });
                }
                tract_onnx::tract_hir::ops::scan::InputMapping::State => {
                    input_mappings.push(InputMapping::State);
                }
                tract_onnx::tract_hir::ops::scan::InputMapping::Full => {
                    input_mappings.push(InputMapping::Full);
                }
            }
        }
        input_mappings
    }

    /// Builds output mappings from ONNX scan output mapping
    fn build_output_mappings(
        output_mapping: &[tract_onnx::tract_hir::ops::scan::OutputMapping<TDim>],
    ) -> Vec<Vec<OutputMapping>> {
        let mut output_mappings = vec![];
        for mapping in output_mapping.iter() {
            let mut mappings = vec![];
            if let Some(outlet) = mapping.last_value_slot {
                mappings.push(OutputMapping::Single {
                    outlet,
                    is_state: mapping.state,
                });
            }
            if let Some(last) = mapping.scan {
                mappings.push(OutputMapping::Stacked {
                    outlet: last.0,
                    axis: last.1.axis,
                    is_state: false,
                });
            }
            output_mappings.push(mappings);
        }
        output_mappings
    }

    /// Builds output scale override map for subgraph nodes
    fn build_output_scale_override(
        input_state_idx: &[usize],
        output_state_idx: Vec<usize>,
        input_scales: &[crate::Scale],
        output_mappings: &[Vec<OutputMapping>],
        body: &Graph<TypedFact, Box<dyn TypedOp>>,
    ) -> HashMap<usize, crate::Scale> {
        let mut output_scale_override = HashMap::new();

        for (input_idx, output_idx) in input_state_idx.iter().zip(output_state_idx) {
            let input_scale = input_scales[*input_idx];
            let mut traversed_len = 0;
            for (outer_idx, mappings) in output_mappings.iter().enumerate() {
                let mapping_len = mappings.len();
                if traversed_len + mapping_len > output_idx {
                    let output_node_idx = body.outputs[outer_idx].node;
                    output_scale_override.insert(output_node_idx, input_scale);
                }
                traversed_len += mapping_len;
            }
        }

        output_scale_override
    }

    /// Extracts output scales from the subgraph model
    fn extract_output_scales(
        output_mapping: &[tract_onnx::tract_hir::ops::scan::OutputMapping<TDim>],
        subgraph_model: &Model,
    ) -> Vec<crate::Scale> {
        let mut output_scales = BTreeMap::new();
        for (i, _mapping) in output_mapping.iter().enumerate() {
            for mapping in output_mapping.iter() {
                if let Some(outlet) = mapping.last_value_slot {
                    output_scales.insert(outlet, subgraph_model.graph.get_output_scales()[i]);
                }
                if let Some(last) = mapping.scan {
                    output_scales.insert(last.0, subgraph_model.graph.get_output_scales()[i]);
                }
            }
        }
        output_scales.into_values().collect_vec()
    }

    /// Applies input scale override to a node if applicable
    pub fn apply_input_scale_override(
        node: &mut Node,
        override_input_scales: &Option<Vec<crate::Scale>>,
        input_idx: &mut usize,
    ) {
        if let Some(ref scales) = override_input_scales {
            if let Some(inp) = node.opkind.get_input() {
                let scale = scales[*input_idx];
                node.opkind = SupportedOp::Input(Input {
                    scale,
                    datum_type: inp.datum_type,
                });
                *input_idx += 1;
                node.out_scale = scale;
            }
        }
    }

    /// Applies output scale override to a node if applicable
    pub fn apply_output_scale_override(
        node: &mut Node,
        node_index: usize,
        override_output_scales: &Option<HashMap<usize, crate::Scale>>,
    ) {
        if let Some(ref scales) = override_output_scales {
            if scales.contains_key(&node_index) {
                let scale_diff = node.out_scale - scales[&node_index];
                node.opkind = if scale_diff > 0 {
                    RebaseScale::rebase(node.opkind.clone(), scales[&node_index], node.out_scale, 1)
                } else {
                    RebaseScale::rebase_up(node.opkind.clone(), scales[&node_index], node.out_scale)
                };
                node.out_scale = scales[&node_index];
            }
        }
    }

    /// Handles the insertion of a node, expanding RebaseScale nodes if necessary
    pub fn handle_node_insertion(
        nodes: &mut BTreeMap<usize, NodeType>,
        remappings: &mut BTreeMap<usize, usize>,
        original_index: usize,
        node: Node,
    ) {
        // Check if this node is a RebaseScale and expand it into two separate nodes
        if let SupportedOp::RebaseScale(rebase_scale) = &node.opkind {
            // Special case: if the inner operation is MeanOfSquares, expand it first
            if let SupportedOp::Linear(PolyOp::MeanOfSquares { axes }) = rebase_scale.inner.as_ref()
            {
                // First, expand the MeanOfSquares (which is inside the RebaseScale)
                let mos_expanded_nodes =
                    Self::expand_mean_of_squares_node(&node, axes, nodes.len(), nodes);

                // Insert all the MeanOfSquares expanded nodes
                for expanded_node in &mos_expanded_nodes {
                    nodes.insert(expanded_node.idx, NodeType::Node(expanded_node.clone()));
                }

                // The last node from MeanOfSquares expansion is the final DIV
                let mos_final_idx = mos_expanded_nodes.last().unwrap().idx;

                // Now create the RebaseScale's division node that takes the MeanOfSquares result as input
                let rebase_div_idx = nodes.len();
                let rebase_div_node = Node {
                    idx: rebase_div_idx,
                    opkind: SupportedOp::Nonlinear(crate::ops::lookup::LookupOp::Div {
                        denom: crate::ops::utils::F32(rebase_scale.multiplier as f32),
                    }),
                    inputs: vec![(mos_final_idx, 0)], // Takes output from MeanOfSquares expansion
                    out_dims: node.out_dims.clone(),
                    out_scale: rebase_scale.target_scale,
                    num_uses: node.num_uses,
                };
                nodes.insert(rebase_div_idx, NodeType::Node(rebase_div_node));
                remappings.insert(original_index, rebase_div_idx);
            } else {
                // Normal RebaseScale expansion (inner is not MeanOfSquares)
                let (inner_node, div_node) =
                    Self::expand_rebase_scale_node(&node, rebase_scale, nodes.len());

                // Insert the inner node first
                let inner_node_idx = inner_node.idx;
                nodes.insert(inner_node_idx, NodeType::Node(inner_node));

                // Insert the div node and set up remapping to point to the final div node
                let div_node_idx = div_node.idx;
                remappings.insert(original_index, div_node_idx);
                nodes.insert(div_node_idx, NodeType::Node(div_node));
            }
        } else if let SupportedOp::Linear(PolyOp::MeanOfSquares { axes }) = &node.opkind {
            // Expand MeanOfSquares into [POW, SUM, DIV, DIV] nodes
            let expanded_nodes = Self::expand_mean_of_squares_node(&node, axes, nodes.len(), nodes);

            // Insert all expanded nodes
            for expanded_node in &expanded_nodes {
                nodes.insert(expanded_node.idx, NodeType::Node(expanded_node.clone()));
            }

            // Set up remapping to point to the final div node (last in the expansion)
            let final_node_idx = expanded_nodes.last().unwrap().idx;
            remappings.insert(original_index, final_node_idx);
        } else {
            // For all other nodes, just insert them directly
            remappings.insert(original_index, node.idx);
            nodes.insert(node.idx, NodeType::Node(node));
        }
    }

    /// Expands a RebaseScale node into an inner operation node and a division node
    pub fn expand_rebase_scale_node(
        original_node: &Node,
        rebase_scale: &RebaseScale,
        next_available_index: usize,
    ) -> (Node, Node) {
        // Create first node: the inner operation
        let inner_node_idx = next_available_index;
        let inner_node = Node {
            idx: inner_node_idx,
            opkind: (*rebase_scale.inner).clone(),
            inputs: original_node.inputs.clone(),
            out_dims: original_node.out_dims.clone(),
            out_scale: rebase_scale.original_scale,
            num_uses: 1, // Will be used by the div node
        };

        // Create second node: the division operation
        let div_node_idx = next_available_index + 1;
        let div_node = Node {
            idx: div_node_idx,
            opkind: SupportedOp::Nonlinear(crate::ops::lookup::LookupOp::Div {
                denom: crate::ops::utils::F32(rebase_scale.multiplier as f32),
            }),
            inputs: vec![(inner_node_idx, 0)], // Takes output from inner node
            out_dims: original_node.out_dims.clone(),
            out_scale: rebase_scale.target_scale,
            num_uses: original_node.num_uses, // Inherits the usage count from original node
        };

        (inner_node, div_node)
    }

    /// Expands a MeanOfSquares node into [Square, Sum, Div, Div] nodes
    pub fn expand_mean_of_squares_node(
        original_node: &Node,
        axes: &[usize],
        next_available_index: usize,
        nodes: &BTreeMap<usize, NodeType>,
    ) -> Vec<Node> {
        let mut result_nodes = Vec::new();

        // Calculate the input scale - we know that MeanOfSquares output scale is 2 * input_scale
        // So input_scale = output_scale / 2
        let input_scale = original_node.out_scale / 2;

        // Get the actual input dimensions by looking up the input node
        let input_dims = if let Some((input_idx, _)) = original_node.inputs.first() {
            if let Some(NodeType::Node(input_node)) = nodes.get(input_idx) {
                input_node.out_dims.clone()
            } else {
                // If we can't find the input node, this is an error condition
                // but we'll fall back to using the current approach for now
                eprintln!(
                    "Warning: Could not find input node {input_idx} for MeanOfSquares expansion"
                );
                original_node.out_dims.clone()
            }
        } else {
            // No inputs specified - this shouldn't happen for MeanOfSquares
            eprintln!("Warning: MeanOfSquares node has no inputs");
            original_node.out_dims.clone()
        };

        // Calculate sum output dimensions (axes are reduced to size 1)
        let mut sum_output_dims = input_dims.clone();
        for &axis in axes {
            if axis < sum_output_dims.len() {
                sum_output_dims[axis] = 1;
            }
        }

        // Node 1: Square operation (Power of 2)
        // Square operation preserves input dimensions
        let square_node_idx = next_available_index;
        let square_node = Node {
            idx: square_node_idx,
            opkind: SupportedOp::Linear(PolyOp::Pow(2)), // Square is power of 2
            inputs: original_node.inputs.clone(),
            out_dims: input_dims.clone(), // Square preserves input dimensions
            out_scale: 2 * input_scale,   // Squaring doubles the scale
            num_uses: 1,                  // Will be used by sum node
        };
        result_nodes.push(square_node);

        // Node 2: Sum operation
        // Sum reduces dimensions along specified axes to size 1
        let sum_node_idx = next_available_index + 1;
        let sum_node = Node {
            idx: sum_node_idx,
            opkind: SupportedOp::Linear(PolyOp::Sum {
                axes: axes.to_vec(),
            }),
            inputs: vec![(square_node_idx, 0)], // Takes output from square node
            out_dims: sum_output_dims.clone(),  // Sum reduces dimensions along axes
            out_scale: 2 * input_scale,         // Same scale as square
            num_uses: 1,                        // Will be used by first div node
        };
        result_nodes.push(sum_node);

        // Node 3: First Division operation (divide by count to get mean)
        // Calculate the denominator: total elements in reduced axes
        let denominator = axes
            .iter()
            .map(|&axis| input_dims.get(axis).unwrap_or(&1))
            .product::<usize>() as f32;

        let div1_node_idx = next_available_index + 2;
        let div1_node = Node {
            idx: div1_node_idx,
            opkind: SupportedOp::Nonlinear(crate::ops::lookup::LookupOp::Div {
                denom: crate::ops::utils::F32(denominator),
            }),
            inputs: vec![(sum_node_idx, 0)], // Takes output from sum node
            out_dims: sum_output_dims.clone(), // Division preserves sum output dimensions
            out_scale: 2 * input_scale,      // Still same scale after dividing by count
            num_uses: 1,                     // Will be used by second div node
        };
        result_nodes.push(div1_node);

        // Node 4: Second Division operation (for scaling purposes)
        let div2_node_idx = next_available_index + 3;
        let div2_node = Node {
            idx: div2_node_idx,
            opkind: SupportedOp::Nonlinear(crate::ops::lookup::LookupOp::Div {
                denom: crate::ops::utils::F32(1.0), // Identity division for now
            }),
            inputs: vec![(div1_node_idx, 0)], // Takes output from first div node
            out_dims: original_node.out_dims.clone(), // Final output dimensions should match original MeanOfSquares output
            out_scale: original_node.out_scale,       // Final output scale matches original
            num_uses: original_node.num_uses,         // Inherits the usage count from original node
        };
        result_nodes.push(div2_node);

        result_nodes
    }

    /// Run tract onnx model on sample data !
    pub fn run_onnx_predictions(
        run_args: &RunArgs,
        model_path: &std::path::Path,
        data_chunks: &[GraphData],
        input_shapes: Vec<Vec<usize>>,
    ) -> Result<Vec<Vec<Tensor<f32>>>, Box<dyn Error>> {
        use tract_onnx::tract_core::internal::IntoArcTensor;
        let (model, _) = Model::load_onnx_using_tract(
            &mut std::fs::File::open(model_path)
                .map_err(|_| format!("failed to load model at {}", model_path.display()))?,
            run_args,
        );
        let datum_types: Vec<DatumType> = model
            .input_outlets()?
            .iter()
            .map(|o| model.node(o.node).outputs[o.slot].fact.datum_type)
            .collect();
        let runnable_model = model.into_runnable()?;
        let mut outputs = vec![];
        for chunk in data_chunks {
            #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
            let result = runnable_model.run(chunk.to_tract_data(&input_shapes, &datum_types)?)?;
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            {
                // WASM fallback: to_tract_data method not available
                return Err(Box::new(GraphError::UnsupportedOp));
            }
            #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
            outputs.push(
                result
                    .into_iter()
                    .map(|t| {
                        crate::graph::utilities::extract_tensor_value(t.into_arc_tensor()).unwrap()
                    })
                    .collect(),
            );
        }
        Ok(outputs)
    }

    /// Removes all nodes that are consts with 0 uses
    fn remove_unused_nodes(nodes: &mut BTreeMap<usize, NodeType>) {
        // remove all nodes that are consts with 0 uses now
        nodes.retain(|_, n| match n {
            NodeType::Node(n) => match &mut n.opkind {
                SupportedOp::Constant(c) => {
                    c.empty_raw_value();
                    n.num_uses > 0
                }
                _ => n.num_uses > 0,
            },
            NodeType::SubGraph { model, .. } => {
                Self::remove_unused_nodes(&mut model.graph.nodes);
                true
            }
        });
    }

    /// Ensures that node indices are consecutive starting from 0.
    /// This is needed after RebaseScale expansion which can create gaps in indexing.
    fn ensure_consecutive_indices(nodes: &mut BTreeMap<usize, NodeType>) {
        // Create a mapping from old indices to new indices
        let old_indices: Vec<usize> = nodes.keys().cloned().collect();
        let mut index_mapping: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();

        for (new_idx, &old_idx) in old_indices.iter().enumerate() {
            index_mapping.insert(old_idx, new_idx);
        }

        // Collect all nodes and update their indices
        let mut updated_nodes = BTreeMap::new();

        for (new_idx, (_old_idx, node)) in nodes.iter().enumerate() {
            let mut updated_node = node.clone();

            match &mut updated_node {
                NodeType::Node(n) => {
                    // Update the node's internal index
                    n.idx = new_idx;

                    // Update input references to use new indices
                    for (input_idx, _outlet) in &mut n.inputs {
                        if let Some(&new_input_idx) = index_mapping.get(input_idx) {
                            *input_idx = new_input_idx;
                        }
                    }
                }
                NodeType::SubGraph { .. } => {
                    // Handle subgraphs if needed - for now we don't modify them
                }
            }

            updated_nodes.insert(new_idx, updated_node);
        }

        // Replace the original nodes map
        *nodes = updated_nodes;
    }

    /// Expands any MeanOfSquares nodes into [Square, Sum, Div, Div] nodes
    pub fn table_nodes(&self) -> String {
        let mut node_accumulator = vec![];
        let mut string = String::new();
        for (idx, node) in &self.graph.nodes {
            match node {
                NodeType::Node(n) => {
                    node_accumulator.push(n);
                }
                NodeType::SubGraph { model, inputs, .. } => {
                    let mut table = Table::new(node_accumulator.iter());
                    table.with(tabled::settings::Style::modern());
                    table.with(tabled::settings::Shadow::new(1));
                    table.with(
                        tabled::settings::style::BorderColor::default()
                            .top(tabled::settings::Color::BG_YELLOW),
                    );
                    string = format!("{string} \n\n  MAIN GRAPH \n\n{table}",);
                    node_accumulator = vec![];
                    string = format!(
                        "{}\n\n SUBGRAPH AT IDX {} WITH INPUTS {:?}\n{}",
                        string,
                        idx,
                        inputs,
                        model.table_nodes(),
                    );
                }
            }
        }
        let mut table = Table::new(node_accumulator.iter());
        table.with(tabled::settings::Style::modern());
        format!("{string} \n{table}",)
    }

    pub fn add_node(
        &mut self,
        op: SupportedOp,
        inputs: Vec<Outlet>,
        out_dims: Vec<usize>,
    ) -> Result<usize, Box<dyn Error>> {
        let node_id = (0..self.graph.nodes.len() + 1)
            .find(|i| !self.graph.nodes.contains_key(i))
            .ok_or(GraphError::MissingNode(0))?;
        self.graph.nodes.insert(
            node_id,
            NodeType::Node(op.gen_node(inputs, out_dims, node_id)),
        );
        Ok(node_id)
    }

    pub fn insert_node(&mut self, node: Node) {
        let node_id = node.idx;
        if self.graph.nodes.contains_key(&node_id) {
            panic!("Node with index {node_id} already exists.");
        }
        self.graph.nodes.insert(node_id, NodeType::Node(node));
    }

    pub fn set_inputs(&mut self, inputs: Vec<usize>) {
        self.graph.inputs = inputs;
    }

    pub fn set_outputs(&mut self, outputs: Vec<Outlet>) {
        self.graph.outputs = outputs;
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
/// Represents a parsed computational graph consisting of ONNX nodes, inputs, and outputs.
///
/// `ParsedNodes` is the core data structure used to describe the internal representation of a computational graph
/// after it has been loaded and converted from an ONNX model. It contains all the nodes (operations and subgraphs),
/// as well as metadata about which nodes are considered inputs and outputs. This struct is central to the execution,
/// transformation, and analysis of models within this module.
///
/// - After loading an ONNX model, the graph is parsed into a `ParsedNodes` instance.
/// - The [`Model`] struct holds a `ParsedNodes` as its main graph representation.
/// - During model execution (inference), the `inputs` field is used to map user-provided tensors to the correct nodes,
///   and the `outputs` field is used to extract the final results after computation.
/// - When exporting, tracing, or analyzing the model, `ParsedNodes` provides a complete and queryable view of the graph structure.
///
/// - Encapsulates the entire computational graph, including all nodes and their relationships.
/// - Clearly separates the graph's structure (nodes) from its entry points (inputs) and exit points (outputs).
/// - Enables flexible manipulation, traversal, and execution of the graph for various purposes (inference, optimization, etc.).
/// - Supports both simple models and complex graphs with subgraphs (e.g., for control flow).
///
/// # Example
/// ```ignore
/// // Construct a simple graph: input -> add (with a constant) -> output
/// let mut nodes = BTreeMap::new();
/// nodes.insert(0, NodeType::Node(input_node));
/// nodes.insert(1, NodeType::Node(const_node));
/// nodes.insert(2, NodeType::Node(add_node));
///
/// let parsed = ParsedNodes {
///     nodes,
///     inputs: vec![0],           // Node 0 is the input node
///     outputs: vec![(2, 0)],     // Node 2's output (outlet 0) is the model output
/// };
///
/// // Use in a Model:
/// let model = Model { graph: parsed, tracer: Tracer::default() };
/// let result = model.forward(&[input_tensor]).unwrap();
/// println!("Model output: {:?}", result.outputs);
/// ```
pub struct ParsedNodes {
    /// The nodes in the graph.
    pub nodes: BTreeMap<usize, NodeType>,
    /// Indices of the input nodes for this computational graph.
    ///
    /// # Why we need this field
    /// This field specifies which nodes in the graph are considered as inputs—i.e., the entry points where external data is fed into the model.
    /// Each entry in this vector is the unique index (usize) of a node in the `nodes` map that acts as an input node.
    /// This is essential for:
    /// - Mapping user-provided input tensors to the correct nodes during model execution.
    /// - Determining the expected input signature (shapes, types) of the model.
    /// - Supporting models with multiple inputs (e.g., multi-branch networks).
    ///
    /// # What it does
    /// When running inference, the code uses this vector to know which nodes should receive the input tensors provided by the user.
    /// The order of indices in this vector determines the order in which input tensors should be supplied.
    ///
    /// # Example
    /// For a simple model: `input -> const -> add`, where "input" is node 0, "const" is node 1, and "add" is node 2,
    /// the `inputs` field would be: `vec![0]` (only node 0 is an input node).
    pub inputs: Vec<usize>,

    /// Output nodes and their outlet indices for this computational graph.
    ///
    /// # Why we need this field
    /// This field defines which nodes (and which output slots, if a node has multiple outputs) are considered as the outputs of the model.
    /// Each entry is a tuple `(node_index, outlet_index)`, where:
    /// - `node_index` is the index of the node in the `nodes` map.
    /// - `outlet_index` specifies which output of the node is used (for nodes with multiple outputs).
    ///
    /// This is essential for:
    /// - Collecting the final results after model execution.
    /// - Supporting models with multiple outputs (e.g., multi-task networks).
    /// - Mapping the internal graph outputs to the user-facing model outputs.
    ///
    /// # What it does
    /// After executing the graph, the code uses this vector to extract the correct output tensors from the results map.
    /// The order of entries determines the order of outputs returned to the user.
    ///
    /// # Example
    /// For the model: `input -> const -> add`, where "add" is node 2 and produces a single output at outlet 0,
    /// the `outputs` field would be: `vec![(2, 0)]` (the output of node 2, outlet 0, is the model's output).
    ///
    /// # Usage Example
    /// ```ignore
    /// // Suppose we have a graph:
    /// // Node 0: Input
    /// // Node 1: Constant
    /// // Node 2: Add (inputs: Node 0, Node 1)
    /// let parsed_nodes = ParsedNodes {
    ///     nodes: /* ... */,
    ///     inputs: vec![0],           // Node 0 is the input node
    ///     outputs: vec![(2, 0)],     // Node 2's output (outlet 0) is the model output
    /// };
    /// // When running inference:
    /// // - The input tensor is mapped to node 0.
    /// // - After execution, the output is taken from node 2, outlet 0.
    /// ```
    pub outputs: Vec<Outlet>,
}

impl ParsedNodes {
    /// Returns the fixed point scale of the computational graph's inputs
    pub fn get_input_scales(&self) -> Vec<crate::Scale> {
        let input_nodes = self.inputs.iter();
        input_nodes
            .flat_map(|idx| {
                self.nodes
                    .get(idx)
                    .ok_or(GraphError::MissingNode(*idx))
                    .map(|n| n.out_scales())
                    .unwrap_or_default()
            })
            .collect()
    }

    ///  Returns shapes of the computational graph's inputs
    pub fn input_shapes(&self) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
        let mut inputs = vec![];
        for input in self.inputs.iter() {
            let node = self
                .nodes
                .get(input)
                .ok_or(GraphError::MissingNode(*input))?;
            let input_dims = node.out_dims();
            let input_dim = input_dims.first().unwrap();
            inputs.push(input_dim.clone());
        }
        Ok(inputs)
    }

    /// Returns the fixed point scale of the computational graph's outputs
    pub fn get_output_scales(&self) -> Vec<crate::Scale> {
        let output_nodes = self.outputs.iter();
        output_nodes
            .map(|(idx, outlet)| self.nodes.get(idx).unwrap().out_scales()[*outlet])
            .collect::<Vec<_>>()
    }
}

// /// Enables model as subnode of other models
#[derive(Clone, Debug, PartialEq)]
pub enum NodeType {
    /// A node in the model
    Node(Node),
    /// A submodel
    SubGraph {
        /// The subgraph
        model: Model,
        /// The subgraph's inputs
        inputs: Vec<Outlet>,
        /// the subgraph's idx within the parent graph
        idx: usize,
        /// output mappings
        output_mappings: Vec<Vec<OutputMapping>>,
        /// input mappings
        input_mappings: Vec<InputMapping>,
        ///
        out_dims: Vec<Vec<usize>>,
        ///
        out_scales: Vec<crate::Scale>,
    },
}

impl NodeType {
    pub fn is_lookup(&self) -> bool {
        match self {
            NodeType::Node(n) => n.opkind.is_lookup(),
            NodeType::SubGraph { .. } => false,
        }
    }

    pub fn num_uses(&self) -> usize {
        match self {
            NodeType::Node(n) => n.num_uses,
            NodeType::SubGraph { .. } => 0,
        }
    }

    pub fn is_pow2(&self) -> bool {
        match self {
            NodeType::Node(n) => matches!(n.opkind, SupportedOp::Linear(PolyOp::Pow(2))),
            NodeType::SubGraph { .. } => false,
        }
    }

    /// Returns the indices of the node's inputs.
    pub fn inputs(&self) -> Vec<Outlet> {
        match self {
            NodeType::Node(n) => n.inputs.clone(),
            NodeType::SubGraph { inputs, .. } => inputs.clone(),
        }
    }

    /// Returns the dimensions of the node's output.
    pub fn out_dims(&self) -> Vec<Vec<usize>> {
        match self {
            NodeType::Node(n) => vec![n.out_dims.clone()],
            NodeType::SubGraph { out_dims, .. } => out_dims.clone(),
        }
    }

    /// Returns the scales of the node's output.
    pub fn out_scales(&self) -> Vec<crate::Scale> {
        match self {
            NodeType::Node(n) => vec![n.out_scale],
            NodeType::SubGraph { out_scales, .. } => out_scales.clone(),
        }
    }

    /// Returns a string representation of the operation.
    pub fn as_str(&self) -> String {
        match self {
            NodeType::Node(n) => n.opkind.as_string(),
            NodeType::SubGraph { .. } => "SUBGRAPH".into(),
        }
    }

    /// Returns true if the operation is a rebase
    pub fn is_rebase(&self) -> bool {
        match self {
            NodeType::Node(n) => matches!(n.opkind, SupportedOp::RebaseScale { .. }),
            NodeType::SubGraph { .. } => false,
        }
    }

    /// Returns true if the operation is an input.
    pub fn is_input(&self) -> bool {
        match self {
            NodeType::Node(n) => n.opkind.is_input(),
            NodeType::SubGraph { .. } => false,
        }
    }

    /// Returns true if the operation is a const.
    pub fn is_constant(&self) -> bool {
        match self {
            NodeType::Node(n) => n.opkind.is_constant(),
            NodeType::SubGraph { .. } => false,
        }
    }

    /// Returns the node's unique identifier.
    pub fn idx(&self) -> usize {
        match self {
            NodeType::Node(n) => n.idx,
            NodeType::SubGraph { idx, .. } => *idx,
        }
    }

    /// decrement const num times used
    pub fn decrement_use(&mut self) {
        match self {
            NodeType::Node(n) => n.num_uses -= 1,
            NodeType::SubGraph { .. } => log::warn!("Cannot decrement const of subgraph"),
        }
    }

    /// bunp scale of node
    pub fn bump_scale(&mut self, scale: crate::Scale) {
        match self {
            NodeType::Node(n) => n.out_scale = scale,
            NodeType::SubGraph { .. } => log::warn!("Cannot bump scale of subgraph"),
        }
    }

    /// Replace the operation kind of the node.
    pub fn replace_opkind(&mut self, opkind: SupportedOp) {
        match self {
            NodeType::Node(n) => n.opkind = opkind,
            NodeType::SubGraph { .. } => log::warn!("Cannot replace opkind of subgraph"),
        }
    }

    /// Returns the operation kind of the node (if any).
    pub fn opkind(&self) -> SupportedOp {
        match self {
            NodeType::Node(n) => n.opkind.clone(),
            NodeType::SubGraph { .. } => SupportedOp::Unknown(Unknown),
        }
    }
}

/// The result of a forward pass.
#[derive(Clone, Debug)]

pub struct ForwardResult {
    /// The outputs of the forward pass.
    pub outputs: Vec<Tensor<i32>>,
    /// The maximum value of any input to a lookup operation.
    pub max_lookup_inputs: i32,
    /// The minimum value of any input to a lookup operation.
    pub min_lookup_inputs: i32,
}

/// Representation of execution graph
pub type NodeGraph = BTreeMap<usize, NodeType>;

///
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum OutputMapping {
    ///
    Single {
        ///
        outlet: usize,
        ///
        is_state: bool,
    },
    ///
    Stacked {
        ///
        outlet: usize,
        ///
        axis: usize,
        ///
        is_state: bool,
    },
}

impl OutputMapping {
    ///
    pub fn is_state(&self) -> bool {
        match self {
            OutputMapping::Single { is_state, .. } => *is_state,
            OutputMapping::Stacked { is_state, .. } => *is_state,
        }
    }

    ///
    pub fn outlet(&self) -> usize {
        match self {
            OutputMapping::Single { outlet, .. } => *outlet,
            OutputMapping::Stacked { outlet, .. } => *outlet,
        }
    }
}

/// Describes how each input to a subgraph (such as a Scan/Loop in ONNX) is mapped from the parent graph.
///
/// # Overview
/// `InputMapping` is used to specify how the inputs to a subgraph are fed from the outer graph.
/// This is essential for handling ONNX control flow operators like Scan, Loop, or custom subgraphs,
/// where each input may be treated differently (e.g., as a state variable, as a full tensor, or as a chunked/stacked input).
///
/// # Variants
/// - `Full`: The entire input tensor is passed as-is to the subgraph input.
/// - `State`: The input acts as a state variable, typically carried across iterations (e.g., hidden state in RNNs).
/// - `Stacked { axis, chunk }`: The input is split along the specified `axis` into chunks of size `chunk`,
///   and each chunk is fed to the subgraph in each iteration (used for sequence processing).
///
/// # Role in the Codebase
/// `InputMapping` is crucial for correctly wiring up subgraphs during model parsing and execution.
/// It allows the code to handle dynamic and complex ONNX models that use control flow or iterative constructs,
/// ensuring that data is fed into subgraphs in the correct shape and order. This enables support for models
/// with recurrent or iterative computation patterns.
///
/// # Example
/// For an ONNX Scan node processing a sequence, the input sequence tensor might be mapped as `Stacked`,
/// while an initial hidden state would be mapped as `State`.
///
/// # Usage
/// The mapping is determined during graph parsing (`nodes_from_graph`) and is later used during execution
/// (`forward`) to slice, reshape, or carry inputs as needed for each subgraph invocation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum InputMapping {
    ///
    Full,
    ///
    State,
    ///
    Stacked {
        ///
        axis: usize,
        ///
        chunk: usize,
    },
}

fn number_of_iterations(mappings: &[InputMapping], dims: Vec<&[usize]>) -> usize {
    let mut number_of_iterations =
        dims.iter()
            .zip(mappings)
            .filter_map(|(dims, mapping)| match mapping {
                InputMapping::Stacked { axis, chunk } => Some(
                    // number of iterations given the dim size along the axis
                    // and the chunk size
                    dims[*axis].div_ceil(*chunk),
                ),
                _ => None,
            });
    // assert all collected number of iterations are equal
    assert!(number_of_iterations.clone().all_equal());

    number_of_iterations.next().unwrap_or(1)
}

fn input_state_idx(input_mappings: &[InputMapping]) -> Vec<usize> {
    input_mappings
        .iter()
        .enumerate()
        .filter(|(_, r)| matches!(r, InputMapping::State))
        .map(|(index, _)| index)
        .collect::<Vec<_>>()
}

fn output_state_idx(output_mappings: &[Vec<OutputMapping>]) -> Vec<usize> {
    output_mappings
        .iter()
        .flatten()
        .filter_map(|x| if x.is_state() { Some(x.outlet()) } else { None })
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::{
        graph::{
            node::{map_outlet_indices, Node, SupportedOp},
            utilities::{create_const_node, create_input_node, create_polyop_node},
        },
        ops::poly::PolyOp,
    };
    use tract_onnx::prelude::OutletId;

    /// Test loading a model that requires broadcasting functionality
    #[test]
    fn test_model_with_broadcast_requirements() {
        // Test the map_outlet_indices function directly
        let outlets = vec![
            OutletId::new(0, 0),
            OutletId::new(1, 0),
            OutletId::new(2, 0),
        ];

        // Graph:       [Node0, Node1, Node2]
        // Node list:   [Broadcast, Node0, Broadcast, Node1, Node2]

        let mut remappings = BTreeMap::new();
        remappings.insert(0, 1); // Node at idx 0 of Graph maps to index 1 in nodes list
        remappings.insert(1, 3); // Node at idx 1 of Graph maps to index 3 in nodes list
        remappings.insert(2, 4); // Node at idx 2 of Graph maps to index 4 in nodes list
        let mapped = map_outlet_indices(&outlets, &remappings);

        // Each outlet should be mapped to the node's index in the nodes list
        assert_eq!(mapped[0], (1, 0));
        assert_eq!(mapped[1], (3, 0));
        assert_eq!(mapped[2], (4, 0));
    }

    /// Test model building with broadcasting using model manipulation directly
    #[test]
    fn test_model_with_broadcasting_manual() {
        // Test using manual model construction instead of ModelBuilder
        let mut model = Model::default();

        // Create a scalar input [1]
        let scalar_input = create_input_node(7, vec![1], 0, 1);
        model.insert_node(scalar_input);

        // Create a vector constant [1, 3]
        let vector_data = vec![5, 10, 15];
        let vector_tensor = Tensor::new(Some(&vector_data), &[1, 3]).unwrap();
        let raw_tensor = Tensor::new(Some(&[5.0, 10.0, 15.0]), &[1, 3]).unwrap();
        let vector_const = create_const_node(vector_tensor, raw_tensor, 7, vec![1, 3], 1, 1);
        model.insert_node(vector_const);

        // Add operation that requires matching shapes
        let add_node = create_polyop_node(
            PolyOp::<i32>::Add,
            7,
            vec![(0, 0), (1, 0)],
            vec![1, 3],
            2,
            1,
        );
        model.insert_node(add_node);

        model.set_inputs(vec![0]);
        model.set_outputs(vec![(2, 0)]);

        // Test the model with a scalar input that should be broadcasted
        let scalar_val = Tensor::new(Some(&[2]), &[1]).unwrap();
        let result = model.forward(&[scalar_val]).unwrap();

        assert_eq!(result.outputs.len(), 1);
        // The scalar 2 should be broadcasted and added to [5, 10, 15] -> [7, 12, 17]
        assert_eq!(
            result.outputs[0],
            Tensor::new(Some(&[7, 12, 17]), &[1, 3]).unwrap()
        );
    }

    /// Test the nodes_from_graph method with remapping functionality
    #[test]
    fn test_nodes_from_graph_with_remappings() {
        // Create a simple test to verify remapping logic works

        // Create input with [1] dimensions
        let input_node = create_input_node(7, vec![1], 0, 1);

        // Create an operation that requires [1, 3] dimensions
        // This should trigger broadcasting when processed by nodes_from_graph
        let mut add_node = Node {
            idx: 1,
            opkind: SupportedOp::Linear(PolyOp::Add),
            inputs: vec![(0, 0)],
            out_dims: vec![1, 3],
            num_uses: 1,
            out_scale: 7,
        };

        // Simulate what happens when a broadcast node is inserted
        let mut nodes = std::collections::BTreeMap::new();
        nodes.insert(0, crate::graph::model::NodeType::Node(input_node));

        // Test homogenization which should add broadcast nodes
        add_node.homogenize_input_shapes(&mut nodes);
        assert!(nodes.len() > 1); // Additional broadcast node added
    }

    /// Test edge cases for broadcasting
    #[test]
    fn test_broadcast_edge_cases() {
        // Test with empty outlets
        let outlets = vec![];

        // Graph:       [Node0, Node1]
        // Node list:   [Broadcast, Node0, Node1]

        let mut remappings = BTreeMap::new();
        remappings.insert(0, 1); // Node at idx 0 of Graph maps to index 1 in nodes list
        remappings.insert(1, 2); // Node at idx 1 of Graph maps to index 2 in nodes list

        let result = map_outlet_indices(&outlets, &remappings);
        assert_eq!(result, vec![]);

        // Test with empty remappings
        let outlets = vec![OutletId::new(0, 0), OutletId::new(1, 1)];

        // Graph:       [Node0, Node1]
        // Node list:   [Node0, Node1]

        let mut remappings = BTreeMap::new();
        remappings.insert(0, 0); // No broadcasts, direct mapping
        remappings.insert(1, 1); // No broadcasts, direct mapping

        let result = map_outlet_indices(&outlets, &remappings);
        assert_eq!(result, vec![(0, 0), (1, 1)]);

        // Test with single outlet and multiple remappings
        let outlets = vec![OutletId::new(2, 0)];

        // Graph:       [Node0, Node1, Node2]
        // Node list:   [Broadcast, Broadcast, Node0, Broadcast, Broadcast, Node1, Broadcast, Node2]

        let mut remappings = BTreeMap::new();
        remappings.insert(0, 2); // Node at idx 0 of Graph maps to index 2 in nodes list
        remappings.insert(1, 5); // Node at idx 1 of Graph maps to index 5 in nodes list
        remappings.insert(2, 7); // Node at idx 2 of Graph maps to index 7 in nodes list

        let result = map_outlet_indices(&outlets, &remappings);
        assert_eq!(result, vec![(7, 0)]); // 2 + 5 = 7
    }

    /// Test that RebaseScale expansion maintains consecutive addressing
    #[test]
    fn test_rebase_scale_consecutive_addressing() {
        // Test with simple_mlp_small model which has RebaseScale nodes
        let model_path = Path::new("models/simple_mlp_small/network.onnx");

        // Skip test if model file doesn't exist (e.g., in CI environments)
        if !model_path.exists() {
            return;
        }

        let model = crate::model(&model_path.into());
        let bytecode = crate::decode_model(model);

        // Check that addresses are consecutive starting from 1
        let mut expected_address = 1;
        for instr in &bytecode {
            assert_eq!(
                instr.address,
                expected_address,
                "Address gap detected: expected {}, got {}. Previous instruction: {:?}",
                expected_address,
                instr.address,
                if expected_address > 1 {
                    Some(&bytecode[expected_address - 2])
                } else {
                    None
                }
            );
            expected_address += 1;
        }

        // Also check that td (target destination) values are consecutive starting from 0
        for (i, instr) in bytecode.iter().enumerate() {
            if let Some(td) = instr.td {
                assert_eq!(
                    td, i,
                    "Target destination gap detected: expected {i}, got {td} at instruction {i}"
                );
            }
        }
    }

    /// Test with addsubmul1 model for RebaseScale expansion
    #[test]
    fn test_rebase_scale_expansion_addsubmul1() {
        let model_path = Path::new("models/addsubmul1/network.onnx");

        // Skip test if model file doesn't exist
        if !model_path.exists() {
            return;
        }

        let model = crate::model(&model_path.into());
        let bytecode = crate::decode_model(model);

        // Should have Input, Constant, Add, Sub, Mul (inner), Div operations
        assert_eq!(
            bytecode.len(),
            6,
            "Expected 6 instructions after RebaseScale expansion"
        );

        // Check specific sequence: last operation should be Mul followed by Div
        let mul_found = bytecode
            .iter()
            .any(|instr| matches!(instr.opcode, crate::trace_types::ONNXOpcode::Mul));
        let div_found = bytecode
            .iter()
            .any(|instr| matches!(instr.opcode, crate::trace_types::ONNXOpcode::Div));

        assert!(mul_found, "Mul operation should be present");
        assert!(
            div_found,
            "Div operation should be present after RebaseScale expansion"
        );

        // Check addresses are consecutive
        for (i, instr) in bytecode.iter().enumerate() {
            assert_eq!(instr.address, i + 1, "Address should be consecutive");
        }
    }

    #[test]
    fn test_mean_of_squares_expansion() {
        // Test that MeanOfSquares gets expanded into 4 nodes: [Square, Sum, Div, Div]
        use crate::ops::poly::PolyOp;
        use std::collections::BTreeMap;

        // Create a mock input node that the MeanOfSquares node references
        let input_node = Node {
            idx: 4,
            opkind: SupportedOp::Input(crate::ops::Input {
                scale: 7,
                datum_type: crate::ops::InputType::F32,
            }),
            inputs: vec![],
            out_dims: vec![1, 16], // Original input dimensions before reduction
            out_scale: 7,
            num_uses: 1,
        };

        // Create a nodes map with the input node
        let mut nodes = BTreeMap::new();
        nodes.insert(4, NodeType::Node(input_node));

        // Create a mock MeanOfSquares node
        let mean_of_squares_node = Node {
            idx: 5,
            opkind: SupportedOp::Linear(PolyOp::MeanOfSquares { axes: vec![1] }),
            inputs: vec![(4, 0)],
            out_dims: vec![1, 1], // After reduction: axis 1 reduced from 16 to 1
            out_scale: 14,        // 2 * input_scale (7)
            num_uses: 1,
        };

        let expanded = Model::expand_mean_of_squares_node(&mean_of_squares_node, &[1], 10, &nodes);

        // Should have 4 nodes
        assert_eq!(expanded.len(), 4);

        // Check node types and indices
        assert_eq!(expanded[0].idx, 10); // Square
        assert_eq!(expanded[1].idx, 11); // Sum
        assert_eq!(expanded[2].idx, 12); // Div
        assert_eq!(expanded[3].idx, 13); // Div

        // Check operations
        if let SupportedOp::Linear(PolyOp::Pow(2)) = expanded[0].opkind {
            // Square is Pow(2) - correct
        } else {
            panic!("First node should be Square (Pow(2))");
        }

        if let SupportedOp::Linear(PolyOp::Sum { axes }) = &expanded[1].opkind {
            assert_eq!(axes, &vec![1]);
        } else {
            panic!("Second node should be Sum");
        }

        if let SupportedOp::Nonlinear(crate::ops::lookup::LookupOp::Div { .. }) = expanded[2].opkind
        {
            // Third node is Div - correct
        } else {
            panic!("Third node should be Div");
        }

        if let SupportedOp::Nonlinear(crate::ops::lookup::LookupOp::Div { .. }) = expanded[3].opkind
        {
            // Fourth node is Div - correct
        } else {
            panic!("Fourth node should be Div");
        }

        // Check that nodes are chained correctly
        assert_eq!(expanded[1].inputs, vec![(10, 0)]); // Sum takes from Square
        assert_eq!(expanded[2].inputs, vec![(11, 0)]); // First Div takes from Sum
        assert_eq!(expanded[3].inputs, vec![(12, 0)]); // Second Div takes from First Div
    }

    #[test]
    fn test_mean_of_squares_expansion_in_layernorm_model() {
        // Test MeanOfSquares expansion in the layernorm_head model which should contain MeanOfSquares ops
        use std::fs::File;

        let model_result = File::open("./models/layernorm_head/network.onnx")
            .map(|mut file| Model::new(&mut file, &crate::RunArgs::default()));

        if let Ok(model) = model_result {
            let parsed_nodes = &model.graph;

            // Check if any MeanOfSquares nodes were expanded
            let mut square_count = 0;
            let mut sum_count = 0;
            let mut div_count = 0;
            let mut mean_of_squares_count = 0;

            for node_type in parsed_nodes.nodes.values() {
                if let NodeType::Node(node) = node_type {
                    match &node.opkind {
                        SupportedOp::Linear(PolyOp::Pow(2)) => square_count += 1,
                        SupportedOp::Linear(PolyOp::Sum { .. }) => sum_count += 1,
                        SupportedOp::Linear(PolyOp::MeanOfSquares { .. }) => {
                            mean_of_squares_count += 1
                        }
                        SupportedOp::Nonlinear(crate::ops::lookup::LookupOp::Div { .. }) => {
                            div_count += 1
                        }
                        _ => {}
                    }
                }
            }

            // If the model was properly loaded and had MeanOfSquares ops, we should see the expanded operations
            println!("MeanOfSquares nodes: {mean_of_squares_count}");
            println!("Square (Pow(2)) nodes: {square_count}");
            println!("Sum nodes: {sum_count}");
            println!("Div nodes: {div_count}");

            // Verify addresses are consecutive
            let mut addresses = Vec::new();
            for node_type in parsed_nodes.nodes.values() {
                if let NodeType::Node(node) = node_type {
                    addresses.push(node.idx);
                }
            }
            addresses.sort();

            for i in 1..addresses.len() {
                assert_eq!(
                    addresses[i],
                    addresses[i - 1] + 1,
                    "Non-consecutive addresses found: {} followed by {}",
                    addresses[i - 1],
                    addresses[i]
                );
            }
        } else {
            // If the layernorm model doesn't exist or can't be loaded, skip this test
            println!("Layernorm model not available, skipping test");
        }
    }

    #[test]
    fn debug_layernorm_expansion() {
        // Safe logger initialization that doesn't panic if already initialized
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .try_init();
        use std::path::PathBuf;

        // Load the layernorm_head model
        let model = crate::model(&PathBuf::from("models/layernorm_head/network.onnx"));

        // Check for MeanOfSquares nodes
        let mut mean_of_squares_count = 0;
        let mut square_count = 0;
        let mut sum_count = 0;
        let mut div_count = 0;

        for node_type in model.graph.nodes.values() {
            if let NodeType::Node(node) = node_type {
                match &node.opkind {
                    SupportedOp::Linear(PolyOp::MeanOfSquares { .. }) => mean_of_squares_count += 1,
                    SupportedOp::Linear(PolyOp::Pow(2)) => square_count += 1,
                    SupportedOp::Linear(PolyOp::Sum { .. }) => sum_count += 1,
                    SupportedOp::Nonlinear(crate::ops::lookup::LookupOp::Div { .. }) => {
                        div_count += 1
                    }
                    _ => {}
                }
            }
        }

        // Should have 0 MeanOfSquares nodes (all expanded)
        assert_eq!(
            mean_of_squares_count, 0,
            "MeanOfSquares nodes should be expanded"
        );

        // Should have at least some square, sum, and div nodes from expansion
        assert!(square_count > 0, "Should have square nodes from expansion");
        assert!(sum_count > 0, "Should have sum nodes from expansion");
        assert!(div_count > 0, "Should have div nodes from expansion");
    }

    #[test]
    fn test_mean_of_squares_execution() {
        // Safe logger initialization that doesn't panic if already initialized
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .try_init();

        // Create a simple test model with MeanOfSquares that we can execute
        let mut nodes = std::collections::BTreeMap::new();

        // Input node
        let input_node = Node {
            idx: 0,
            opkind: SupportedOp::Input(crate::ops::Input {
                scale: 7,
                datum_type: crate::ops::InputType::Int,
            }),
            inputs: vec![],
            out_dims: vec![2, 2],
            out_scale: 7,
            num_uses: 1,
        };
        nodes.insert(0, NodeType::Node(input_node));

        // MeanOfSquares node
        let mean_of_squares_node = Node {
            idx: 1,
            opkind: SupportedOp::Linear(PolyOp::MeanOfSquares { axes: vec![1] }),
            inputs: vec![(0, 0)],
            out_dims: vec![2, 1],
            out_scale: 14, // 2 * input_scale
            num_uses: 1,
        };
        nodes.insert(1, NodeType::Node(mean_of_squares_node));

        // Before expansion - check indices
        println!("Before expansion:");
        for idx in nodes.keys() {
            println!("Node {idx}");
        }

        // Note: MeanOfSquares expansion now happens automatically in handle_node_insertion
        // For this manual test, we need to manually expand since we're not using handle_node_insertion
        // This is just for testing the expansion logic itself
        let nodes_to_expand: Vec<_> = nodes
            .iter()
            .filter_map(|(idx, node_type)| {
                if let NodeType::Node(node) = node_type {
                    if let SupportedOp::Linear(PolyOp::MeanOfSquares { axes }) = &node.opkind {
                        return Some((*idx, node.clone(), axes.clone()));
                    }
                }
                None
            })
            .collect();

        for (original_idx, original_node, axes) in nodes_to_expand {
            let next_index = nodes.keys().max().unwrap_or(&0) + 1;
            let expanded_nodes =
                Model::expand_mean_of_squares_node(&original_node, &axes, next_index, &nodes);
            let final_node_idx = next_index + expanded_nodes.len() - 1;

            nodes.remove(&original_idx);
            for expanded_node in expanded_nodes {
                nodes.insert(expanded_node.idx, NodeType::Node(expanded_node));
            }

            // Update references
            for node_type in nodes.values_mut() {
                if let NodeType::Node(node) = node_type {
                    for (input_idx, _) in &mut node.inputs {
                        if *input_idx == original_idx {
                            *input_idx = final_node_idx;
                        }
                    }
                }
            }
        }

        // After expansion - check indices and references
        println!("After expansion:");
        let indices: Vec<usize> = nodes.keys().cloned().collect();
        for &idx in &indices {
            if let Some(NodeType::Node(node)) = nodes.get(&idx) {
                println!("Node {}: {:?}, inputs: {:?}", idx, node.opkind, node.inputs);
            }
        }

        // Ensure consecutive indices
        Model::ensure_consecutive_indices(&mut nodes);

        // After consecutive indexing
        println!("After ensuring consecutive indices:");
        let indices: Vec<usize> = nodes.keys().cloned().collect();
        for &idx in &indices {
            if let Some(NodeType::Node(node)) = nodes.get(&idx) {
                println!("Node {}: {:?}, inputs: {:?}", idx, node.opkind, node.inputs);
            }
        }

        // Verify that all references are valid
        for (idx, node_type) in &nodes {
            if let NodeType::Node(node) = node_type {
                for (input_idx, _) in &node.inputs {
                    assert!(
                        nodes.contains_key(input_idx),
                        "Node {idx} references missing node {input_idx}"
                    );
                    assert!(
                        *input_idx < *idx,
                        "Node {idx} references future node {input_idx} (forward reference)"
                    );
                }
            }
        }

        println!("All references valid and no forward references");
    }

    /// Test MeanOfSquares expansion creates correct node sequence
    #[test]
    fn test_mean_of_squares_node_sequence() {
        let mut nodes = BTreeMap::new();

        // Input node with shape [2, 4]
        let input_node = Node {
            idx: 0,
            opkind: SupportedOp::Input(crate::ops::Input {
                scale: 7,
                datum_type: crate::ops::InputType::F32,
            }),
            inputs: vec![],
            out_dims: vec![2, 4],
            out_scale: 7,
            num_uses: 1,
        };
        nodes.insert(0, NodeType::Node(input_node));

        // MeanOfSquares node reducing axis 1
        let mos_node = Node {
            idx: 1,
            opkind: SupportedOp::Linear(PolyOp::MeanOfSquares { axes: vec![1] }),
            inputs: vec![(0, 0)],
            out_dims: vec![2, 1],
            out_scale: 14,
            num_uses: 1,
        };

        let expanded = Model::expand_mean_of_squares_node(&mos_node, &[1], 10, &nodes);

        // Verify sequence: [Pow, Sum, Div, Div]
        assert_eq!(expanded.len(), 4, "Should expand to 4 nodes");

        // Node 0: Pow(2) - squares the input
        assert!(matches!(
            expanded[0].opkind,
            SupportedOp::Linear(PolyOp::Pow(2))
        ));
        assert_eq!(expanded[0].out_dims, vec![2, 4]); // Preserves input dims
        assert_eq!(expanded[0].out_scale, 14); // 2 * input_scale

        // Node 1: Sum - reduces along axis
        if let SupportedOp::Linear(PolyOp::Sum { axes }) = &expanded[1].opkind {
            assert_eq!(axes, &vec![1]);
        } else {
            panic!("Second node should be Sum");
        }
        assert_eq!(expanded[1].out_dims, vec![2, 1]); // Reduced dimension

        // Node 2: Div - divides by count (4 elements)
        if let SupportedOp::Nonlinear(crate::ops::lookup::LookupOp::Div { denom }) =
            &expanded[2].opkind
        {
            assert_eq!(denom.0, 4.0);
        } else {
            panic!("Third node should be Div");
        }

        // Node 3: Div - final scaling division
        assert!(matches!(
            expanded[3].opkind,
            SupportedOp::Nonlinear(crate::ops::lookup::LookupOp::Div { .. })
        ));
        assert_eq!(expanded[3].out_dims, vec![2, 1]); // Final output dims
        assert_eq!(expanded[3].out_scale, 14); // Matches original MOS output scale
    }

    /// Test MeanOfSquares expansion with multiple axes
    #[test]
    fn test_mean_of_squares_multi_axis() {
        let mut nodes = BTreeMap::new();

        // Input node with shape [2, 3, 4]
        let input_node = Node {
            idx: 5,
            opkind: SupportedOp::Input(crate::ops::Input {
                scale: 10,
                datum_type: crate::ops::InputType::F32,
            }),
            inputs: vec![],
            out_dims: vec![2, 3, 4],
            out_scale: 10,
            num_uses: 1,
        };
        nodes.insert(5, NodeType::Node(input_node));

        // MeanOfSquares reducing axes [1, 2]
        let mos_node = Node {
            idx: 6,
            opkind: SupportedOp::Linear(PolyOp::MeanOfSquares { axes: vec![1, 2] }),
            inputs: vec![(5, 0)],
            out_dims: vec![2, 1, 1],
            out_scale: 20,
            num_uses: 1,
        };

        let expanded = Model::expand_mean_of_squares_node(&mos_node, &[1, 2], 20, &nodes);

        // Verify denominator is product of reduced axes: 3 * 4 = 12
        if let SupportedOp::Nonlinear(crate::ops::lookup::LookupOp::Div { denom }) =
            &expanded[2].opkind
        {
            assert_eq!(
                denom.0, 12.0,
                "Denominator should be product of reduced dimensions"
            );
        } else {
            panic!("Third node should be Div with correct denominator");
        }

        // Verify output dimensions
        assert_eq!(
            expanded[1].out_dims,
            vec![2, 1, 1],
            "Sum should reduce to [2, 1, 1]"
        );
    }

    /// Test that MeanOfSquares nodes are removed after expansion
    #[test]
    fn test_mean_of_squares_removal() {
        let mut nodes = BTreeMap::new();

        let input_node = Node {
            idx: 0,
            opkind: SupportedOp::Input(crate::ops::Input {
                scale: 7,
                datum_type: crate::ops::InputType::F32,
            }),
            inputs: vec![],
            out_dims: vec![1, 8],
            out_scale: 7,
            num_uses: 1,
        };
        nodes.insert(0, NodeType::Node(input_node));

        let mos_node = Node {
            idx: 1,
            opkind: SupportedOp::Linear(PolyOp::MeanOfSquares { axes: vec![1] }),
            inputs: vec![(0, 0)],
            out_dims: vec![1, 1],
            out_scale: 14,
            num_uses: 1,
        };
        nodes.insert(1, NodeType::Node(mos_node));

        // Before expansion: should have MeanOfSquares
        let has_mos_before = nodes.values().any(|n| {
            if let NodeType::Node(node) = n {
                matches!(
                    node.opkind,
                    SupportedOp::Linear(PolyOp::MeanOfSquares { .. })
                )
            } else {
                false
            }
        });
        assert!(has_mos_before, "Should have MeanOfSquares before expansion");

        // Expand (inline since handle_node_insertion does this automatically now)
        let nodes_to_expand: Vec<_> = nodes
            .iter()
            .filter_map(|(idx, node_type)| {
                if let NodeType::Node(node) = node_type {
                    if let SupportedOp::Linear(PolyOp::MeanOfSquares { axes }) = &node.opkind {
                        return Some((*idx, node.clone(), axes.clone()));
                    }
                }
                None
            })
            .collect();

        for (original_idx, original_node, axes) in nodes_to_expand {
            let next_index = nodes.keys().max().unwrap_or(&0) + 1;
            let expanded_nodes =
                Model::expand_mean_of_squares_node(&original_node, &axes, next_index, &nodes);
            let final_node_idx = next_index + expanded_nodes.len() - 1;

            nodes.remove(&original_idx);
            for expanded_node in expanded_nodes {
                nodes.insert(expanded_node.idx, NodeType::Node(expanded_node));
            }

            for node_type in nodes.values_mut() {
                if let NodeType::Node(node) = node_type {
                    for (input_idx, _) in &mut node.inputs {
                        if *input_idx == original_idx {
                            *input_idx = final_node_idx;
                        }
                    }
                }
            }
        }

        // After expansion: should NOT have MeanOfSquares
        let has_mos_after = nodes.values().any(|n| {
            if let NodeType::Node(node) = n {
                matches!(
                    node.opkind,
                    SupportedOp::Linear(PolyOp::MeanOfSquares { .. })
                )
            } else {
                false
            }
        });
        assert!(
            !has_mos_after,
            "Should NOT have MeanOfSquares after expansion"
        );

        // Should have the expanded nodes instead
        let has_pow = nodes.values().any(|n| {
            if let NodeType::Node(node) = n {
                matches!(node.opkind, SupportedOp::Linear(PolyOp::Pow(2)))
            } else {
                false
            }
        });
        assert!(has_pow, "Should have Pow(2) node after expansion");
    }

    /// Test that input node stays at index 0 after MeanOfSquares expansion
    #[test]
    fn test_mean_of_squares_preserves_input_at_zero() {
        let mut nodes = BTreeMap::new();

        let input_node = Node {
            idx: 0,
            opkind: SupportedOp::Input(crate::ops::Input {
                scale: 7,
                datum_type: crate::ops::InputType::F32,
            }),
            inputs: vec![],
            out_dims: vec![4, 4],
            out_scale: 7,
            num_uses: 1,
        };
        nodes.insert(0, NodeType::Node(input_node));

        let mos_node = Node {
            idx: 1,
            opkind: SupportedOp::Linear(PolyOp::MeanOfSquares { axes: vec![1] }),
            inputs: vec![(0, 0)],
            out_dims: vec![4, 1],
            out_scale: 14,
            num_uses: 1,
        };
        nodes.insert(1, NodeType::Node(mos_node));

        // Expand and ensure consecutive (inline expansion for manual test)
        let nodes_to_expand: Vec<_> = nodes
            .iter()
            .filter_map(|(idx, node_type)| {
                if let NodeType::Node(node) = node_type {
                    if let SupportedOp::Linear(PolyOp::MeanOfSquares { axes }) = &node.opkind {
                        return Some((*idx, node.clone(), axes.clone()));
                    }
                }
                None
            })
            .collect();

        for (original_idx, original_node, axes) in nodes_to_expand {
            let next_index = nodes.keys().max().unwrap_or(&0) + 1;
            let expanded_nodes =
                Model::expand_mean_of_squares_node(&original_node, &axes, next_index, &nodes);
            let final_node_idx = next_index + expanded_nodes.len() - 1;

            nodes.remove(&original_idx);
            for expanded_node in expanded_nodes {
                nodes.insert(expanded_node.idx, NodeType::Node(expanded_node));
            }

            for node_type in nodes.values_mut() {
                if let NodeType::Node(node) = node_type {
                    for (input_idx, _) in &mut node.inputs {
                        if *input_idx == original_idx {
                            *input_idx = final_node_idx;
                        }
                    }
                }
            }
        }
        Model::ensure_consecutive_indices(&mut nodes);

        // Input must be at index 0
        if let Some(NodeType::Node(node)) = nodes.get(&0) {
            assert!(
                node.opkind.is_input(),
                "Node 0 must be input after expansion"
            );
        } else {
            panic!("No node at index 0 after expansion");
        }

        // Verify all indices are consecutive
        let indices: Vec<usize> = nodes.keys().cloned().collect();
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(idx, i, "Indices must be consecutive");
        }
    }

    /// Test MeanOfSquares with single-element axis reduction
    #[test]
    fn test_mean_of_squares_single_element() {
        let mut nodes = BTreeMap::new();

        let input_node = Node {
            idx: 0,
            opkind: SupportedOp::Input(crate::ops::Input {
                scale: 5,
                datum_type: crate::ops::InputType::F32,
            }),
            inputs: vec![],
            out_dims: vec![1, 1],
            out_scale: 5,
            num_uses: 1,
        };
        nodes.insert(0, NodeType::Node(input_node));

        let mos_node = Node {
            idx: 1,
            opkind: SupportedOp::Linear(PolyOp::MeanOfSquares { axes: vec![1] }),
            inputs: vec![(0, 0)],
            out_dims: vec![1, 1],
            out_scale: 10,
            num_uses: 1,
        };

        let expanded = Model::expand_mean_of_squares_node(&mos_node, &[1], 5, &nodes);

        // With single element, denominator should be 1.0
        if let SupportedOp::Nonlinear(crate::ops::lookup::LookupOp::Div { denom }) =
            &expanded[2].opkind
        {
            assert_eq!(denom.0, 1.0, "Denominator should be 1 for single element");
        } else {
            panic!("Third node should be Div");
        }
    }

    /// Test that node chaining is correct after expansion
    #[test]
    fn test_mean_of_squares_node_chaining() {
        let mut nodes = BTreeMap::new();

        let input_node = Node {
            idx: 0,
            opkind: SupportedOp::Input(crate::ops::Input {
                scale: 7,
                datum_type: crate::ops::InputType::F32,
            }),
            inputs: vec![],
            out_dims: vec![2, 8],
            out_scale: 7,
            num_uses: 1,
        };
        nodes.insert(0, NodeType::Node(input_node));

        let mos_node = Node {
            idx: 1,
            opkind: SupportedOp::Linear(PolyOp::MeanOfSquares { axes: vec![1] }),
            inputs: vec![(0, 0)],
            out_dims: vec![2, 1],
            out_scale: 14,
            num_uses: 1,
        };
        nodes.insert(1, NodeType::Node(mos_node));

        // Inline expansion for manual test
        let nodes_to_expand: Vec<_> = nodes
            .iter()
            .filter_map(|(idx, node_type)| {
                if let NodeType::Node(node) = node_type {
                    if let SupportedOp::Linear(PolyOp::MeanOfSquares { axes }) = &node.opkind {
                        return Some((*idx, node.clone(), axes.clone()));
                    }
                }
                None
            })
            .collect();

        for (original_idx, original_node, axes) in nodes_to_expand {
            let next_index = nodes.keys().max().unwrap_or(&0) + 1;
            let expanded_nodes =
                Model::expand_mean_of_squares_node(&original_node, &axes, next_index, &nodes);
            let final_node_idx = next_index + expanded_nodes.len() - 1;

            nodes.remove(&original_idx);
            for expanded_node in expanded_nodes {
                nodes.insert(expanded_node.idx, NodeType::Node(expanded_node));
            }

            for node_type in nodes.values_mut() {
                if let NodeType::Node(node) = node_type {
                    for (input_idx, _) in &mut node.inputs {
                        if *input_idx == original_idx {
                            *input_idx = final_node_idx;
                        }
                    }
                }
            }
        }

        // Find the expanded nodes
        let pow_node = nodes
            .values()
            .find_map(|n| {
                if let NodeType::Node(node) = n {
                    if matches!(node.opkind, SupportedOp::Linear(PolyOp::Pow(2))) {
                        Some(node)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .expect("Should have Pow node");

        let sum_node = nodes
            .values()
            .find_map(|n| {
                if let NodeType::Node(node) = n {
                    if matches!(node.opkind, SupportedOp::Linear(PolyOp::Sum { .. })) {
                        Some(node)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .expect("Should have Sum node");

        // Verify chaining: Sum takes input from Pow
        assert_eq!(sum_node.inputs.len(), 1, "Sum should have 1 input");
        assert_eq!(
            sum_node.inputs[0].0, pow_node.idx,
            "Sum should take input from Pow"
        );

        // Verify Pow takes input from original input node
        assert_eq!(pow_node.inputs.len(), 1, "Pow should have 1 input");
        assert_eq!(
            pow_node.inputs[0].0, 0,
            "Pow should take input from input node at index 0"
        );
    }

    /// Test loading actual MeanOfSquares ONNX model
    #[test]
    fn test_mean_of_squares_onnx_model_loading() {
        use std::fs::File;

        let model_path = Path::new("./models/mean_of_squares_simple/network.onnx");
        if !model_path.exists() {
            println!("Skipping test: model not found at {model_path:?}");
            return;
        }

        let model_result = File::open(model_path)
            .map(|mut file| Model::new(&mut file, &crate::RunArgs::default()));

        if let Ok(model) = model_result {
            let parsed_nodes = &model.graph;

            // Should NOT have any MeanOfSquares nodes after loading
            let has_mos = parsed_nodes.nodes.values().any(|n| {
                if let NodeType::Node(node) = n {
                    matches!(
                        node.opkind,
                        SupportedOp::Linear(PolyOp::MeanOfSquares { .. })
                    )
                } else {
                    false
                }
            });
            assert!(
                !has_mos,
                "Model should not have MeanOfSquares nodes after loading"
            );

            // Should have the expanded operations
            let has_pow = parsed_nodes.nodes.values().any(|n| {
                if let NodeType::Node(node) = n {
                    matches!(node.opkind, SupportedOp::Linear(PolyOp::Pow(2)))
                } else {
                    false
                }
            });
            assert!(has_pow, "Model should have Pow(2) node after expansion");

            let has_sum = parsed_nodes.nodes.values().any(|n| {
                if let NodeType::Node(node) = n {
                    matches!(node.opkind, SupportedOp::Linear(PolyOp::Sum { .. }))
                } else {
                    false
                }
            });
            assert!(has_sum, "Model should have Sum node after expansion");

            // Verify indices are consecutive
            let indices: Vec<usize> = parsed_nodes.nodes.keys().cloned().collect();
            for (i, &idx) in indices.iter().enumerate() {
                assert_eq!(idx, i, "Indices should be consecutive from 0");
            }

            // Verify input is at index 0
            if let Some(NodeType::Node(node)) = parsed_nodes.nodes.get(&0) {
                assert!(node.opkind.is_input(), "Node 0 should be input");
            }
        }
    }

    /// Test loading multi-axis MeanOfSquares ONNX model
    #[test]
    fn test_mean_of_squares_multi_axis_onnx_model() {
        use std::fs::File;

        let model_path = Path::new("./models/mean_of_squares_multi_axis/network.onnx");
        if !model_path.exists() {
            println!("Skipping test: model not found at {model_path:?}");
            return;
        }

        let model_result = File::open(model_path)
            .map(|mut file| Model::new(&mut file, &crate::RunArgs::default()));

        if let Ok(model) = model_result {
            let parsed_nodes = &model.graph;

            // Find the Div node and verify denominator is 12 (3*4)
            let div_nodes: Vec<_> = parsed_nodes
                .nodes
                .values()
                .filter_map(|n| {
                    if let NodeType::Node(node) = n {
                        if let SupportedOp::Nonlinear(crate::ops::lookup::LookupOp::Div { denom }) =
                            &node.opkind
                        {
                            Some(denom.0)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();

            // Should have at least one Div with denominator 12
            let has_correct_denom = div_nodes.iter().any(|&d| (d - 12.0).abs() < 0.01);
            assert!(
                has_correct_denom,
                "Should have Div node with denominator ~12 for axes [1,2] reduction"
            );

            // Verify no MeanOfSquares nodes remain
            let has_mos = parsed_nodes.nodes.values().any(|n| {
                if let NodeType::Node(node) = n {
                    matches!(
                        node.opkind,
                        SupportedOp::Linear(PolyOp::MeanOfSquares { .. })
                    )
                } else {
                    false
                }
            });
            assert!(!has_mos, "Should not have MeanOfSquares after expansion");
        }
    }
}
