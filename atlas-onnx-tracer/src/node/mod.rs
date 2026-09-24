//! Node representation and helpers used by the ONNX tracer.
//! A `ComputationNode` models a single operation in the graph.
mod computation_node;

/// Node-specific handler functions and utilities.
pub mod handlers;

pub use computation_node::ComputationNode;
