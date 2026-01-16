//! Unified trait for operator proving and verification.
//!
//! This module provides the `OperatorHandler` trait that abstracts the prove/verify
//! logic for all ONNX operators, eliminating duplicate code in the main proof module.

use atlas_onnx_tracer::{
    model::trace::{ModelExecutionIO, Trace},
    node::ComputationNode,
};

use std::collections::BTreeMap;

// =============================================================================
// Standard Sumcheck Operators (Add, Sub, Mul, Div, Square, Cube, Iff)
// =============================================================================

// Import the inner operator types
use atlas_onnx_tracer::ops::{
    Add, Broadcast, Constant, Cube, Div, Einsum, Iff, Input, MoveAxis, Mul, ReLU, Reshape, Square,
    Sub,
};

// Implement for standard sumcheck operators

// =============================================================================
// Direct Verification Operators (Broadcast, Reshape, MoveAxis)
// =============================================================================
