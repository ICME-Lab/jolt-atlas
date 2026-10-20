//! The importer's single error type: every rejection names the model, the
//! node, and the reason. There are no panics on malformed or unsupported
//! input anywhere in this crate.

use thiserror::Error;

/// Why an ONNX model could not be imported.
#[derive(Error, Debug)]
pub enum ImportError {
    /// The file could not be read.
    #[error("cannot read {path}: {source}")]
    Io {
        /// Offending path.
        path: String,
        /// Underlying IO error.
        #[source]
        source: std::io::Error,
    },
    /// The protobuf payload could not be decoded.
    #[error("cannot decode {path}: {source}")]
    Decode {
        /// Offending path.
        path: String,
        /// Underlying prost error.
        #[source]
        source: prost::DecodeError,
    },
    /// An external-data reference is invalid or unsafe.
    #[error("tensor {tensor}: external data {reason}")]
    ExternalData {
        /// Tensor name.
        tensor: String,
        /// What is wrong with the reference.
        reason: String,
    },
    /// The model uses a protobuf feature we deliberately reject.
    #[error("{what} is not supported ({context})")]
    UnsupportedFeature {
        /// The rejected feature.
        what: String,
        /// Where it appeared.
        context: String,
    },
    /// An operator the importer does not know.
    #[error("node {node} ({op_type}): unsupported operator")]
    UnsupportedOp {
        /// Node name.
        node: String,
        /// ONNX op type.
        op_type: String,
    },
    /// An operator attribute combination the importer does not support.
    #[error("node {node} ({op_type}): {reason}")]
    UnsupportedAttribute {
        /// Node name.
        node: String,
        /// ONNX op type.
        op_type: String,
        /// Which attribute/value is unsupported.
        reason: String,
    },
    /// A graph-input dimension that needs a binding was not supplied.
    #[error("input {input}: dimension {dim} ({dim_param:?}) has no binding; pass it in LoadOptions")]
    UnboundDim {
        /// Graph input name.
        input: String,
        /// Dimension index.
        dim: usize,
        /// The symbolic name, if any.
        dim_param: String,
    },
    /// Shape inference failed.
    #[error("node {node} ({op_type}): shape inference failed: {reason}")]
    Shape {
        /// Node name.
        node: String,
        /// ONNX op type.
        op_type: String,
        /// What went wrong.
        reason: String,
    },
    /// A value that must be constant at import time is runtime-dependent.
    #[error("node {node} ({op_type}): input {input} must be constant at import time")]
    NonConstant {
        /// Node name.
        node: String,
        /// ONNX op type.
        op_type: String,
        /// Which input must be constant.
        input: String,
    },
    /// A rewrite pass rejected the graph.
    #[error("pass {pass}: node {node}: {reason}")]
    Pass {
        /// Pass name.
        pass: &'static str,
        /// Node name.
        node: String,
        /// Why the pass rejected it.
        reason: String,
    },
    /// A node survived all passes without mapping to the target IR.
    #[error("node {node} ({op_type}) cannot be lowered to the atlas-trace op set")]
    Unlowered {
        /// Node name.
        node: String,
        /// ONNX op type.
        op_type: String,
    },
    /// A float constant cannot be quantized.
    #[error("constant {tensor}: value {value} outside the quantizable range")]
    QuantRange {
        /// Tensor name.
        tensor: String,
        /// Offending value.
        value: f32,
    },
    /// The lowered graph failed atlas-trace validation or compilation.
    #[error("lowered graph rejected by atlas-trace: {0}")]
    Compile(#[from] atlas_trace::error::CompileError),
    /// Graph-structure problem (dangling edge, cycle, duplicate name).
    #[error("malformed graph: {0}")]
    Malformed(String),
}
