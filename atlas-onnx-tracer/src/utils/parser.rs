use crate::{
    model::RunArgs,
    node::{
        ComputationNode,
        handlers::{HANDLERS, HandlerContext},
    },
    tensor::Tensor,
};

use std::{collections::BTreeMap, sync::Arc};
use tract_onnx::{
    prelude::{Node as TractNode, *},
    tract_hir::ops::scan::Scan,
};

pub struct GraphParser<'a> {
    graph: &'a Graph<TypedFact, Box<dyn TypedOp>>,
    run_args: &'a RunArgs,
    symbol_values: &'a SymbolValues,
}

impl<'a> GraphParser<'a> {
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

    /// Main entry point - orchestrates the entire parsing process
    pub fn parse(self) -> (BTreeMap<usize, ComputationNode>, NodeIndexMapper) {
        // nodes
        let mut context = ParsingContext::new();

        // Pass 1: Iterate through nodes in order and transform them
        for node in self.graph.nodes.iter() {
            self.visit_node(node.clone(), &mut context);
        }

        (context.nodes, context.mapper)
    }

    fn visit_node(
        &self,
        node: TractNode<TypedFact, Box<dyn TypedOp>>,
        context: &mut ParsingContext,
    ) {
        self.update_graph(context, node);
    }

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

    /// Resolves the internal indices for all input nodes of the given tract node
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

    /// Fetches the computation nodes corresponding to the given internal indices
    fn fetch_input_nodes(&self, indices: &[usize], ctx: &ParsingContext) -> Vec<ComputationNode> {
        indices
            .iter()
            .map(|&idx| ctx.nodes.get(&idx).unwrap().clone())
            .collect()
    }

    /// Retrieves the handler function for the node's operator
    fn get_operator_handler(
        &self,
        node: &TractNode<TypedFact, Box<dyn TypedOp>>,
    ) -> &'static dyn Fn(&mut HandlerContext) -> Vec<ComputationNode> {
        let op_name = node.op().name();
        HANDLERS
            .get(op_name.as_ref())
            .unwrap_or_else(|| panic!("Unsupported ONNX operator: {op_name}"))
    }

    fn update_graph_with_subgraph(&self, _scan_op: &Scan, _context: &mut ParsingContext) {
        unimplemented!("Sub-graphs (Scan) are not yet supported");
    }

    /// Panics
    /// Panics if more than one output, for now we only support single-output nodes
    pub fn node_output_shape(
        node: &TractNode<TypedFact, Box<dyn TypedOp>>,
        symbol_values: &SymbolValues,
    ) -> Vec<usize> {
        let output_shapes = Self::node_output_shapes(node, symbol_values);
        assert!(output_shapes.len() == 1);
        output_shapes[0].clone()
    }

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
}

/// Maps tract output indices to internal node indices using the node mapper
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
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a direct 1:1 mapping (no expansion occurred)
    pub fn register_direct(&mut self, onnx_idx: usize, internal_idx: usize) {
        self.mappings.insert(onnx_idx, internal_idx);
    }

    /// Register an expansion: onnx_idx -> [start_idx..=end_idx]
    /// The final node (end_idx) becomes the "output" of this expansion
    pub fn register_expansion(
        &mut self,
        onnx_idx: usize,
        expansion_range: std::ops::RangeInclusive<usize>,
    ) {
        let end_idx = *expansion_range.end();
        self.mappings.insert(onnx_idx, end_idx);
    }

    /// Get the final internal index for an ONNX node
    pub fn get(&self, onnx_idx: usize) -> Option<usize> {
        self.mappings.get(&onnx_idx).copied()
    }

    /// Get all mapped internal indices (the "anchors")
    pub fn internal_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.mappings.values().copied()
    }

    /// Check if an internal index is an anchor (final node of an expansion)
    pub fn is_anchor(&self, internal_idx: usize) -> bool {
        self.mappings.values().any(|&idx| idx == internal_idx)
    }
}

/// Builder for multi-node decompositions with explicit index reservation
pub struct DecompositionBuilder {
    base_idx: usize,
    nodes: Vec<ComputationNode>,
}

impl DecompositionBuilder {
    pub fn new(ctx: &mut ParsingContext, count: usize) -> Self {
        Self {
            base_idx: ctx.reserve_indices(count),
            nodes: Vec::with_capacity(count),
        }
    }

    /// Get the index for a node at the given offset from base
    pub fn idx(&self, offset: usize) -> usize {
        self.base_idx + offset
    }

    /// Add a node to the decomposition
    pub fn add_node(&mut self, node: ComputationNode) {
        self.nodes.push(node);
    }

    /// Finish building and return the nodes
    pub fn finish(self) -> Vec<ComputationNode> {
        self.nodes
    }
}

#[derive(Debug, Default)]
/// Mutable state that accumulates during parsing
pub struct ParsingContext {
    /// The accumulated computation nodes
    pub nodes: BTreeMap<usize, ComputationNode>,
    /// Tracks ONNX -> internal index mappings
    pub mapper: NodeIndexMapper,
    /// Counter for assigning internal indices
    next_idx: usize,
}

impl ParsingContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserve a block of indices for multi-node decomposition
    /// Returns the starting index
    ///
    /// This is an internal method used by DecompositionBuilder.
    /// Handlers should use DecompositionBuilder instead of calling this directly.
    pub(crate) fn reserve_indices(&mut self, count: usize) -> usize {
        let start_idx = self.next_idx;
        self.next_idx += count;
        start_idx
    }

    /// Add nodes that already have their indices assigned (from reserve_indices)
    ///
    /// This is an internal method used by the parser.
    /// Handlers should use DecompositionBuilder instead of calling this directly.
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

/// Extracts the raw values from a tensor.
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
