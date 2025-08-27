use serde::{Deserialize, Serialize};

use crate::jolt::precompiles::matmult::MatMultPrecompile;
pub mod matmult;

/// Specifies the ONNX precompile operators used in the Jolt ONNX VM.
/// Used to specifiy the precompile type and its input's in the [`JoltONNXTraceStep`]
#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum PrecompileOperators {
    /// Matrix multiplication precompile.
    MatMult(MatMultPrecompile),
}
