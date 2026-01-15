//! Graph parser for converting tract-onnx representations to internal computation nodes.
//!
//! This module provides the core parsing infrastructure for transforming ONNX models
//! (via tract-onnx) into the internal computation graph representation used by the tracer.
//!
//! # Overview
//!
//! The parsing process involves:
//! - Converting tract nodes to internal `ComputationNode`s
//! - Handling node decomposition (e.g., complex ops split into multiple nodes)
//! - Managing index mappings between ONNX and internal representations
//! - Resolving symbolic dimensions and dynamic shapes
//!
//! # Key Components
//!
//! - [`GraphParser`]: Main parser orchestrating the transformation process
//! - [`NodeIndexMapper`]: Tracks mappings between ONNX and internal node indices
//! - [`DecompositionBuilder`]: Helper for building multi-node decompositions
//! - [`ParsingContext`]: Mutable state accumulating nodes during parsing
//!
//! # Example Flow
//!
//! ```text
//! ONNX Graph (tract-onnx)
//!     ↓
//! GraphParser::parse()
//!     ↓
//! For each node:
//!   - Resolve inputs via NodeIndexMapper
//!   - Invoke operator handler
//!   - Handler may use DecompositionBuilder for multi-node ops
//!   - Register mapping in ParsingContext
//!     ↓
//! Internal Computation Graph
//! ```

use crate::{
    model::RunArgs,
    node::{
        ComputationNode,
        handlers::{HANDLERS, HandlerContext},
    },
    ops::Operator,
    tensor::Tensor,
};

use std::{collections::BTreeMap, sync::Arc};
use tract_onnx::{
    prelude::{Node as TractNode, *},
    tract_hir::ops::scan::Scan,
};

/// Parser for converting tract ONNX graphs into internal computation nodes.
///
/// This parser orchestrates the transformation of a tract-onnx graph representation
/// into our internal graph structure, handling node decomposition, index mapping,
/// and operator-specific transformations.
pub struct GraphParser<'a> {
    /// The tract-onnx graph being parsed
    graph: &'a Graph<TypedFact, Box<dyn TypedOp>>,
    /// Runtime arguments for the model execution
    run_args: &'a RunArgs,
    /// Resolved symbol values for dynamic shapes
    symbol_values: &'a SymbolValues,
}

impl<'a> GraphParser<'a> {
    /// Creates a new graph parser.
    ///
    /// # Arguments
    ///
    /// * `graph` - The tract-onnx graph to parse
    /// * `run_args` - Runtime arguments for model execution
    /// * `symbol_values` - Resolved values for symbolic dimensions
    pub fn new(
        graph: &'a Graph<TypedFact, Box<dyn TypedOp>>,
        run_args: &'a RunArgs,
        symbol_values: &'a SymbolValues,
    ) -> Self {
        Self {
            graph,
            run_args,
            symbol_values,
        }
    }

    /// Main entry point - orchestrates the entire parsing process.
    ///
    /// Iterates through all nodes in the tract graph, transforms them into
    /// computation nodes, and builds the index mapping.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * A map of internal node indices to computation nodes
    /// * A mapper tracking the relationship between original and internal indices
    pub fn parse(self) -> (BTreeMap<usize, ComputationNode>, NodeIndexMapper) {
        // nodes
        let mut context = ParsingContext::new();

        // Pass 1: Iterate through nodes in order and transform them
        for node in self.graph.nodes.iter() {
            self.visit_node(node.clone(), &mut context);
        }

        (context.nodes, context.mapper)
    }

    /// Visits a single node and processes it.
    ///
    /// # Arguments
    ///
    /// * `node` - The tract node to visit
    /// * `context` - Mutable parsing context for accumulating results
    fn visit_node(
        &self,
        node: TractNode<TypedFact, Box<dyn TypedOp>>,
        context: &mut ParsingContext,
    ) {
        self.update_graph(context, node);
    }

    /// Updates the internal graph with a tract node.
    ///
    /// Determines whether the node contains a subgraph (Scan operation) or is
    /// a regular node, and routes to the appropriate handler.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Mutable parsing context
    /// * `node` - The tract node to process
    pub fn update_graph(
        &self,
        ctx: &mut ParsingContext,
        node: TractNode<TypedFact, Box<dyn TypedOp>>,
    ) {
        // Extract the slope layer hyperparams
        match node.op().downcast_ref::<Scan>() {
            None => {
                self.update_graph_with_node(node, ctx);
            }
            Some(scan_op) => {
                self.update_graph_with_subgraph(scan_op, ctx);
            }
        }
    }

    /// Updates the graph with a regular (non-subgraph) node.
    ///
    /// Converts the tract node into one or more computation nodes and registers
    /// the mapping in the context.
    ///
    /// # Arguments
    ///
    /// * `node` - The tract node to process
    /// * `ctx` - Mutable parsing context
    pub fn update_graph_with_node(
        &self,
        node: TractNode<TypedFact, Box<dyn TypedOp>>,
        ctx: &mut ParsingContext,
    ) {
        let onnx_node_idx = node.id;
        let computation_nodes = self.tract_node_to_computation_nodes(node, ctx);
        // Use add_reserved_nodes which handles pre-assigned indices from builder
        let last_idx = ctx.add_reserved_nodes(computation_nodes);
        ctx.mapper.register_direct(onnx_node_idx, last_idx);
    }

    /// Converts a tract node into a vector of computation nodes.
    ///
    /// This method resolves input dependencies, fetches input nodes, determines
    /// output shapes, and invokes the appropriate operator handler to generate
    /// the computation node(s).
    ///
    /// # Arguments
    ///
    /// * `node` - The tract node to convert
    /// * `ctx` - Mutable parsing context
    ///
    /// # Returns
    ///
    /// A vector of computation nodes (may be multiple if the operator decomposes)
    fn tract_node_to_computation_nodes(
        &self,
        node: TractNode<TypedFact, Box<dyn TypedOp>>,
        ctx: &mut ParsingContext,
    ) -> Vec<ComputationNode> {
        let internal_input_indices = self.resolve_input_indices(&node, ctx);
        let internal_input_nodes = self.fetch_input_nodes(&internal_input_indices, ctx);
        let output_dims = GraphParser::node_output_shape(&node, self.symbol_values);
        let handler = self.get_operator_handler(&node);

        let mut handler_ctx = HandlerContext {
            ctx,
            node: &node,
            graph: self.graph,
            run_args: self.run_args,
            symbol_values: self.symbol_values,
            internal_input_indices,
            internal_input_nodes,
            output_dims,
        };

        handler(&mut handler_ctx)
    }

    /// Resolves the internal indices for all input nodes of the given tract node.
    ///
    /// # Arguments
    ///
    /// * `node` - The tract node whose inputs to resolve
    /// * `ctx` - Parsing context containing the index mapper
    ///
    /// # Returns
    ///
    /// A vector of internal node indices corresponding to the node's inputs
    fn resolve_input_indices(
        &self,
        node: &TractNode<TypedFact, Box<dyn TypedOp>>,
        ctx: &ParsingContext,
    ) -> Vec<usize> {
        node.inputs
            .iter()
            .map(|outlet| {
                let input_node = self.graph.node(outlet.node);
                ctx.mapper
                    .get(input_node.id)
                    .expect("Input node must have been processed before the current node")
            })
            .collect()
    }

    /// Fetches the computation nodes corresponding to the given internal indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - The internal node indices to fetch
    /// * `ctx` - Parsing context containing the node map
    ///
    /// # Returns
    ///
    /// A vector of cloned computation nodes
    fn fetch_input_nodes(&self, indices: &[usize], ctx: &ParsingContext) -> Vec<ComputationNode> {
        indices
            .iter()
            .map(|&idx| ctx.nodes.get(&idx).unwrap().clone())
            .collect()
    }

    /// Retrieves the handler function for the node's operator.
    ///
    /// Looks up the operator name in the global HANDLERS registry and returns
    /// the corresponding handler function.
    ///
    /// # Arguments
    ///
    /// * `node` - The tract node whose operator handler to retrieve
    ///
    /// # Returns
    ///
    /// A reference to the handler function
    ///
    /// # Panics
    ///
    /// Panics if the operator is not implemented
    fn get_operator_handler(
        &self,
        node: &TractNode<TypedFact, Box<dyn TypedOp>>,
    ) -> &'static dyn Fn(&mut HandlerContext) -> Vec<ComputationNode> {
        let op_name = node.op().name();
        HANDLERS
            .get(op_name.as_ref())
            .unwrap_or_else(|| unimplemented!("Unimplemented ONNX operator: {op_name}"))
    }

    /// Updates the graph with a subgraph (Scan operation).
    ///
    /// # Arguments
    ///
    /// * `_scan_op` - The Scan operation containing the subgraph
    /// * `_context` - Mutable parsing context
    ///
    /// # Panics
    ///
    /// Currently unimplemented and will panic if called
    fn update_graph_with_subgraph(&self, _scan_op: &Scan, _context: &mut ParsingContext) {
        unimplemented!("Sub-graphs (Scan) are not yet supported");
    }

    /// Extracts the output shape of a tract node.
    ///
    /// # Arguments
    ///
    /// * `node` - The tract node to extract the shape from
    /// * `symbol_values` - Resolved symbol values for evaluating dynamic shapes
    ///
    /// # Returns
    ///
    /// The output shape as a vector of dimensions
    ///
    /// # Panics
    ///
    /// Panics if the node has more than one output (currently unsupported)
    pub fn node_output_shape(
        node: &TractNode<TypedFact, Box<dyn TypedOp>>,
        symbol_values: &SymbolValues,
    ) -> Vec<usize> {
        let output_shapes = Self::node_output_shapes(node, symbol_values);
        assert!(output_shapes.len() == 1);
        output_shapes[0].clone()
    }

    /// Extracts all output shapes of a tract node.
    ///
    /// # Arguments
    ///
    /// * `node` - The tract node to extract shapes from
    /// * `symbol_values` - Resolved symbol values for evaluating dynamic shapes
    ///
    /// # Returns
    ///
    /// A vector of output shapes, where each shape is a vector of dimensions
    pub fn node_output_shapes(
        node: &TractNode<TypedFact, Box<dyn TypedOp>>,
        symbol_values: &SymbolValues,
    ) -> Vec<Vec<usize>> {
        let mut shapes = Vec::new();
        let outputs = node.outputs.to_vec();
        for output in outputs {
            let shape = output.fact.shape;
            let shape = shape.eval_to_usize(symbol_values).unwrap();
            let mv = shape.to_vec();
            shapes.push(mv)
        }
        shapes
    }

    /// Inserts broadcast nodes to ensure input shapes match output dimensions.
    pub fn insert_broadcast_nodes(hctx: &mut HandlerContext) -> Vec<ComputationNode> {
        Self::process_broadcast_inputs(
            hctx.ctx,
            &mut hctx.internal_input_indices,
            &hctx.output_dims,
            hctx.ctx.next_idx,
        )
    }

    /// Inner method to process broadcast insertions.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The parsing context containing input nodes
    /// * `input_indices` - Mutable indices of input nodes, will be updated to point to broadcast nodes if created
    /// * `output_dims` - Target output dimensions
    /// * `node_idx` - The ID of the current node (used to generate broadcast node indices)
    ///
    /// # Returns
    ///
    /// A vector of newly created broadcast nodes
    pub fn process_broadcast_inputs(
        ctx: &ParsingContext,
        input_indices: &mut [usize],
        output_dims: &[usize],
        node_idx: usize,
    ) -> Vec<ComputationNode> {
        let mut new_nodes = Vec::new();
        let mut added_nodes = 0;

        for input_idx in input_indices.iter_mut() {
            let input_node = ctx.nodes.get(input_idx).expect("Input node must exist");

            if input_node.output_dims != output_dims {
                // Insert a broadcast node
                let broadcast_idx = node_idx + added_nodes;
                let broadcast_node = ComputationNode {
                    idx: broadcast_idx,
                    operator: Operator::Broadcast(crate::ops::Broadcast {
                        shape: output_dims.to_vec(),
                    }),
                    inputs: vec![*input_idx],
                    output_dims: output_dims.to_vec(),
                };
                new_nodes.push(broadcast_node);
                *input_idx = broadcast_idx;
                added_nodes += 1;
            }
        }

        new_nodes
    }
}

/// Maps tract output indices to internal node indices using the node mapper.
///
/// # Arguments
///
/// * `tract_outputs` - The tract output outlets to map
/// * `mapper` - The node index mapper containing the mappings
///
/// # Returns
///
/// A vector of internal node indices
///
/// # Panics
///
/// Panics if any output node is not found in the mapper
pub fn map_outputs(tract_outputs: &[OutletId], mapper: &NodeIndexMapper) -> Vec<usize> {
    tract_outputs
        .iter()
        .map(|outlet| {
            mapper
                .get(outlet.node)
                .unwrap_or_else(|| panic!("Output node {} not found in mapper", outlet.node))
        })
        .collect()
}

/// Tracks the mapping between original ONNX node indices and their expanded representations
/// in the internal graph. This is needed because some ONNX nodes are decomposed into multiple
/// internal nodes (e.g., RebaseScale -> [inner_op, const, sra], MeanOfSquares -> [square, sum, div, div]).
#[derive(Debug, Clone, Default)]
pub struct NodeIndexMapper {
    /// Maps original ONNX node index -> final internal node index (the last node in the expansion)
    mappings: BTreeMap<usize, usize>,
}

impl NodeIndexMapper {
    /// Creates a new empty node index mapper.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a direct 1:1 mapping (no expansion occurred).
    ///
    /// # Arguments
    ///
    /// * `onnx_idx` - The original ONNX node index
    /// * `internal_idx` - The corresponding internal node index
    pub fn register_direct(&mut self, onnx_idx: usize, internal_idx: usize) {
        self.mappings.insert(onnx_idx, internal_idx);
    }

    /// Registers an expansion where one ONNX node maps to multiple internal nodes.
    ///
    /// The final node (end_idx) becomes the "output" of this expansion and is
    /// used as the anchor for subsequent node connections.
    ///
    /// # Arguments
    ///
    /// * `onnx_idx` - The original ONNX node index
    /// * `expansion_range` - The range of internal node indices this node expands to
    pub fn register_expansion(
        &mut self,
        onnx_idx: usize,
        expansion_range: std::ops::RangeInclusive<usize>,
    ) {
        let end_idx = *expansion_range.end();
        self.mappings.insert(onnx_idx, end_idx);
    }

    /// Gets the final internal index for an ONNX node.
    ///
    /// # Arguments
    ///
    /// * `onnx_idx` - The original ONNX node index
    ///
    /// # Returns
    ///
    /// The corresponding internal node index, or None if not found
    pub fn get(&self, onnx_idx: usize) -> Option<usize> {
        self.mappings.get(&onnx_idx).copied()
    }

    /// Gets all mapped internal indices (the "anchors").
    ///
    /// Returns an iterator over all internal node indices that serve as
    /// outputs/anchors for ONNX nodes.
    ///
    /// # Returns
    ///
    /// An iterator over internal node indices
    pub fn internal_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.mappings.values().copied()
    }

    /// Checks if an internal index is an anchor (final node of an expansion).
    ///
    /// # Arguments
    ///
    /// * `internal_idx` - The internal node index to check
    ///
    /// # Returns
    ///
    /// `true` if the index is an anchor, `false` otherwise
    pub fn is_anchor(&self, internal_idx: usize) -> bool {
        self.mappings.values().any(|&idx| idx == internal_idx)
    }
}

/// Builder for multi-node decompositions with explicit index reservation.
///
/// This builder is used by operator handlers to construct multi-node decompositions
/// (e.g., RebaseScale -> [inner_op, const, sra]). It pre-reserves a block of indices
/// to ensure stable references during construction.
pub struct DecompositionBuilder {
    base_idx: usize,
    nodes: Vec<ComputationNode>,
}

impl DecompositionBuilder {
    /// Creates a new decomposition builder.
    ///
    /// Reserves a block of `count` indices from the parsing context.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Mutable parsing context to reserve indices from
    /// * `count` - Number of indices to reserve
    pub fn new(ctx: &mut ParsingContext, count: usize) -> Self {
        Self {
            base_idx: ctx.reserve_indices(count),
            nodes: Vec::with_capacity(count),
        }
    }

    /// Gets the index for a node at the given offset from base.
    ///
    /// # Arguments
    ///
    /// * `offset` - The offset from the base index (0-based)
    ///
    /// # Returns
    ///
    /// The absolute internal node index
    pub fn idx(&self, offset: usize) -> usize {
        self.base_idx + offset
    }

    /// Adds a node to the decomposition.
    ///
    /// # Arguments
    ///
    /// * `node` - The computation node to add
    pub fn add_node(&mut self, node: ComputationNode) {
        self.nodes.push(node);
    }

    /// Finishes building and returns the nodes.
    ///
    /// Consumes the builder and returns all accumulated computation nodes.
    ///
    /// # Returns
    ///
    /// A vector of all computation nodes added to the builder
    pub fn finish(self) -> Vec<ComputationNode> {
        self.nodes
    }
}

/// Mutable state that accumulates during parsing.
///
/// This context tracks the accumulated computation nodes, maintains the mapping
/// between ONNX and internal indices, and manages index allocation.
#[derive(Debug, Default)]
pub struct ParsingContext {
    /// The accumulated computation nodes
    pub nodes: BTreeMap<usize, ComputationNode>,
    /// Tracks ONNX -> internal index mappings
    pub mapper: NodeIndexMapper,
    /// Counter for assigning internal indices
    next_idx: usize,
}

impl ParsingContext {
    /// Creates a new empty parsing context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserves a block of indices for multi-node decomposition.
    ///
    /// This is an internal method used by DecompositionBuilder.
    /// Handlers should use DecompositionBuilder instead of calling this directly.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of indices to reserve
    ///
    /// # Returns
    ///
    /// The starting index of the reserved block
    pub(crate) fn reserve_indices(&mut self, count: usize) -> usize {
        let start_idx = self.next_idx;
        self.next_idx += count;
        start_idx
    }

    /// Adds nodes that already have their indices assigned (from reserve_indices).
    ///
    /// This is an internal method used by the parser.
    /// Handlers should use DecompositionBuilder instead of calling this directly.
    ///
    /// # Arguments
    ///
    /// * `nodes` - Vector of computation nodes with pre-assigned indices
    ///
    /// # Returns
    ///
    /// The index of the last node added
    ///
    /// # Panics
    ///
    /// Panics if the nodes vector is empty or if node indices were not properly reserved
    pub(crate) fn add_reserved_nodes(&mut self, nodes: Vec<ComputationNode>) -> usize {
        assert!(!nodes.is_empty());
        let mut last_idx = nodes[0].idx;
        for node in nodes {
            debug_assert!(
                node.idx < self.next_idx,
                "Node index {} not properly reserved (next_idx: {})",
                node.idx,
                self.next_idx
            );
            last_idx = node.idx;
            self.nodes.insert(node.idx, node);
        }
        last_idx
    }
}

/// Loads and downcasts a tract operator to a specific type.
///
/// # Type Parameters
///
/// * `C` - The target operator type to downcast to
///
/// # Arguments
///
/// * `op` - The operator to downcast
/// * `name` - Name of the operator (for error messages)
///
/// # Returns
///
/// A clone of the downcasted operator
///
/// # Panics
///
/// Panics if the operator cannot be downcasted to the target type
pub fn load_op<C: tract_onnx::prelude::Op + Clone>(
    op: &dyn tract_onnx::prelude::Op,
    name: String,
) -> C {
    // Extract the slope layer hyperparams
    let op: &C = match op.downcast_ref::<C>() {
        Some(b) => b,
        None => {
            panic!("Op mismatch: {name}");
        }
    };

    op.clone()
}

/// Extracts the raw values from a tract tensor and converts them to f32.
///
/// Handles various data types (F16, F32, F64, integer types, Bool, TDim) and
/// converts them all to f32 representation.
///
/// # Arguments
///
/// * `input` - The tract tensor to extract values from
///
/// # Returns
///
/// A `Tensor<f32>` containing the converted values, or an error if conversion fails
///
/// # Errors
///
/// Returns an error if the data type is unsupported or if tensor evaluation fails
pub fn extract_tensor_value(
    input: Arc<tract_onnx::prelude::Tensor>,
) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
    use crate::utils::parallel_utils::{IntoParallelRefIterator, ParallelIterator};

    let dt = input.datum_type();
    let dims = input.shape().to_vec();

    let mut const_value: Tensor<f32>;
    if dims.is_empty() && input.len() == 0 {
        const_value = Tensor::<f32>::new(None, &dims)?;
        return Ok(const_value);
    }

    match dt {
        DatumType::F16 => {
            let vec = input.as_slice::<tract_onnx::prelude::f16>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| (*x).into()).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::F32 => {
            let vec = input.as_slice::<f32>()?.to_vec();
            const_value = Tensor::<f32>::new(Some(&vec), &dims)?;
        }
        DatumType::F64 => {
            let vec = input.as_slice::<f64>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::I64 => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<i64>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::I32 => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<i32>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::I16 => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<i16>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::I8 => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<i8>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::U8 => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<u8>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::U16 => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<u16>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::U32 => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<u32>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::U64 => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<u64>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::Bool => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<bool>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as usize as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::TDim => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<tract_onnx::prelude::TDim>()?.to_vec();

            let cast: Result<Vec<f32>, &str> = vec
                .par_iter()
                .map(|x| match x.to_i64() {
                    Ok(v) => Ok(v as f32),
                    Err(_) => match x.to_i64() {
                        Ok(v) => Ok(v as f32),
                        Err(_) => Err("could not evaluate tdim"),
                    },
                })
                .collect();

            const_value = Tensor::<f32>::new(Some(&cast?), &dims)?;
        }
        _ => return Err("unsupported data type".into()),
    }
    const_value.reshape(&dims)?;

    Ok(const_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::ComputationNode;
    use crate::ops::Noop;

    /// Test when input shapes already match output shape - no broadcasting needed
    #[test]
    fn test_process_broadcast_inputs_matching_shapes() {
        let mut ctx = ParsingContext::new();
        ctx.nodes.insert(
            0,
            ComputationNode::new(0, Operator::Noop(Noop), vec![], vec![2, 3]),
        );
        ctx.next_idx = 1;

        let mut internal_input_indices = vec![0];
        let output_dims = vec![2, 3];

        let broadcast_nodes = GraphParser::process_broadcast_inputs(
            &ctx,
            &mut internal_input_indices,
            &output_dims,
            1,
        );

        assert_eq!(broadcast_nodes.len(), 0);
        assert_eq!(internal_input_indices, vec![0]);
    }

    /// Test when broadcasting is needed due to shape mismatch
    #[test]
    fn test_process_broadcast_inputs_shape_mismatch() {
        let mut ctx = ParsingContext::new();
        ctx.nodes.insert(
            0,
            ComputationNode::new(0, Operator::Noop(Noop), vec![], vec![5]),
        );
        ctx.next_idx = 1;

        let mut internal_input_indices = vec![0];
        let output_dims = vec![3, 4, 5];

        let broadcast_nodes = GraphParser::process_broadcast_inputs(
            &ctx,
            &mut internal_input_indices,
            &output_dims,
            1,
        );

        assert_eq!(broadcast_nodes.len(), 1);
        assert_eq!(broadcast_nodes[0].inputs, vec![0]);
        assert_eq!(broadcast_nodes[0].output_dims, vec![3, 4, 5]);
        assert_eq!(internal_input_indices[0], broadcast_nodes[0].idx);
    }

    /// Test multiple inputs with mixed broadcasting requirements
    #[test]
    fn test_process_broadcast_inputs_multiple_inputs_mixed() {
        let mut ctx = ParsingContext::new();
        ctx.nodes.insert(
            0,
            ComputationNode::new(0, Operator::Noop(Noop), vec![], vec![5]),
        );
        ctx.nodes.insert(
            1,
            ComputationNode::new(1, Operator::Noop(Noop), vec![], vec![3, 4, 5]),
        );
        ctx.next_idx = 2;

        let mut internal_input_indices = vec![0, 1];
        let output_dims = vec![3, 4, 5];

        let broadcast_nodes = GraphParser::process_broadcast_inputs(
            &ctx,
            &mut internal_input_indices,
            &output_dims,
            2,
        );

        assert_eq!(broadcast_nodes.len(), 1);
        assert_eq!(internal_input_indices[0], broadcast_nodes[0].idx);
        assert_eq!(internal_input_indices[1], 1);
    }

    /// Test when all inputs need broadcasting
    #[test]
    fn test_process_broadcast_inputs_all_inputs_mismatch() {
        let mut ctx = ParsingContext::new();
        ctx.nodes.insert(
            0,
            ComputationNode::new(0, Operator::Noop(Noop), vec![], vec![2]),
        );
        ctx.nodes.insert(
            1,
            ComputationNode::new(1, Operator::Noop(Noop), vec![], vec![3]),
        );
        ctx.next_idx = 2;

        let mut internal_input_indices = vec![0, 1];
        let output_dims = vec![2, 3];

        let broadcast_nodes = GraphParser::process_broadcast_inputs(
            &ctx,
            &mut internal_input_indices,
            &output_dims,
            2,
        );

        assert_eq!(broadcast_nodes.len(), 2);
        assert_eq!(internal_input_indices[0], broadcast_nodes[0].idx);
        assert_eq!(internal_input_indices[1], broadcast_nodes[1].idx);
        assert_eq!(broadcast_nodes[0].inputs[0], 0);
        assert_eq!(broadcast_nodes[1].inputs[0], 1);
    }
}
