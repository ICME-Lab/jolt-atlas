//! Utility functions and helper modules for the ONNX tracer.

pub mod dims;
pub mod f32;
#[cfg(feature = "onnx-import")]
pub mod handler_builder;
pub mod metrics;
pub mod parallel_utils;
#[cfg(feature = "onnx-import")]
pub mod parser;
#[cfg(test)]
pub mod precision;
/// Pretty-printing utilities for displaying tensors and models.
pub mod pretty_print;
pub mod quantize;
