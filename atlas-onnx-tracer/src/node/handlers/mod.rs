pub mod activation;
pub mod arith;
pub mod index;
pub mod other;
pub mod reduce;
pub mod shape;

use once_cell::sync::Lazy;
use std::collections::HashMap;

use crate::{model::RunArgs, node::ComputationNode, utils::parser::ParsingContext};
use tract_onnx::prelude::{Node as TractNode, *};

// TODO: Mark operators that require rebasing and add the rebase nodes automatically (instead of manually implementing them with the builder)

/// Context passed to each operator handler containing all necessary information
/// to transform a Tract node into ComputationNode(s)
pub struct HandlerContext<'a> {
    /// The parsing context (for index allocation, node storage, etc.)
    pub ctx: &'a mut ParsingContext,
    /// The original Tract node being processed
    pub node: &'a TractNode<TypedFact, Box<dyn TypedOp>>,
    /// The graph containing this node (for accessing input nodes)
    pub graph: &'a Graph<TypedFact, Box<dyn TypedOp>>,
    /// Run arguments (scale, etc.)
    pub run_args: &'a RunArgs,
    /// Symbol values for shape evaluation
    pub symbol_values: &'a SymbolValues,
    /// Pre-computed internal input indices
    pub internal_input_indices: Vec<usize>,
    /// Pre-computed internal input nodes (for operators that need them)
    pub internal_input_nodes: Vec<ComputationNode>,
    /// Pre-computed output dimensions
    pub output_dims: Vec<usize>,
}

/// Type alias for handler functions
pub type OpHandlerFn = fn(&mut HandlerContext) -> Vec<ComputationNode>;

/// Global registry of all operator handlers
pub static HANDLERS: Lazy<HashMap<&'static str, OpHandlerFn>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.extend(arith::handlers());
    m.extend(activation::handlers());
    m.extend(shape::handlers());
    m.extend(reduce::handlers());
    m.extend(index::handlers());
    m.extend(other::handlers());
    m
});
