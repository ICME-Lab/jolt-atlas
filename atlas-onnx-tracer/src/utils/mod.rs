//! Utility functions and helper modules for the ONNX tracer.

pub mod dims;
pub mod f32;
pub mod handler_builder;
pub mod metrics;
pub mod parallel_utils;
pub mod parser;
#[cfg(test)]
pub mod precision;
/// Pretty-printing utilities for displaying tensors and models.
pub mod pretty_print;
/// Quantization Error Analysis (QEA) utilities: shared code for analyzing and improving quantization precision, and for running generation with both quantized and unquantized models.
pub mod qea;
pub mod quantize;
