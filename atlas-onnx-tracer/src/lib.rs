//! Atlas ONNX Tracer - Load and execute quantized ONNX models.
//!
//! This crate provides functionality for loading ONNX machine learning models,
//! quantizing their weights and activations, and executing them with fixed-point
//! arithmetic suitable for zero-knowledge proof systems.

#![warn(missing_docs)]
#![allow(non_snake_case)]

pub mod model;
pub mod node;
/// Operator definitions and implementations for ONNX operations.
pub mod ops;
/// Multi-dimensional tensor representation and operations.
pub mod tensor;
pub mod utils;
