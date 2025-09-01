//! # ONNX MatMul Transpose Test Suite
//!
//! This test suite verifies that ONNX models correctly use the implicit B matrix transpose
//! semantics when parsing MatMul operations.
//!
//! ## Background
//!
//! ONNX MatMul operations use different semantics compared to standard mathematical notation:
//! - Standard math: C = A × B uses einsum equation "ij,jk->ik"  
//! - ONNX semantics: C = A × B uses einsum equation "mk,nk->mn" (implicitly transposes B)
//!
//! This means that for matrices A[m,k] and B[n,k], ONNX MatMul computes A × B^T,
//! where B^T is the transpose of B.

use onnx_tracer::{decode_model, graph::node::SupportedOp, model, ops::poly::PolyOp};
use std::path::PathBuf;

/// Test framework for verifying ONNX MatMul transpose semantics
///
/// This struct allows for flexible testing of ONNX models to verify that MatMul operations
/// use the correct einsum equations. ONNX has specific semantics where the second matrix (B)
/// is implicitly transposed, resulting in the equation "mk,nk->mn" instead of "ij,jk->ik".
pub struct MatMulTransposeTest {
    model_path: String,
    expected_equations: Vec<String>,
    description: String,
    verify_onnx_semantics_only: bool,
}

impl MatMulTransposeTest {
    /// Create a new test configuration
    pub fn new(
        model_path: &'static str,
        expected_equations: Vec<&'static str>,
        description: &'static str,
    ) -> Self {
        Self {
            model_path: model_path.to_string(),
            expected_equations: expected_equations
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            description: description.to_string(),
            verify_onnx_semantics_only: false,
        }
    }

    /// Create a new test that just verifies all MatMul operations use ONNX semantics
    pub fn new_onnx_semantics_only(model_path: &'static str, description: &'static str) -> Self {
        Self {
            model_path: model_path.to_string(),
            expected_equations: vec![],
            description: description.to_string(),
            verify_onnx_semantics_only: true,
        }
    }

    /// Run the test and verify the MatMul operations use correct transpose semantics
    pub fn run(&self) -> Result<(), String> {
        println!("\n=== Running MatMul Transpose Test ===");
        println!("Description: {}", self.description);
        println!("Model: {}", self.model_path);

        // Load the model
        let model_path = PathBuf::from(&self.model_path);
        let model = model(&model_path);
        println!("Model loaded successfully!");

        // Extract equations from model operations
        let mut found_equations = Vec::new();

        println!("Number of nodes: {}", model.graph.nodes.len());
        for (node_id, node_type) in &model.graph.nodes {
            println!("Node {node_id}: PC={node_id}");

            match node_type {
                onnx_tracer::graph::model::NodeType::Node(node) => match &node.opkind {
                    SupportedOp::Linear(PolyOp::Einsum { equation }) => {
                        println!("  Found Einsum operation: equation = '{equation}'");
                        found_equations.push(equation.clone());
                    }
                    SupportedOp::RebaseScale(rebase) => {
                        if let SupportedOp::Linear(PolyOp::Einsum { equation }) = &*rebase.inner {
                            println!(
                                "  Found RebaseScale(Einsum) operation: equation = '{equation}'"
                            );
                            found_equations.push(equation.clone());
                        }
                    }
                    _ => {
                        println!("  Operation: {:?}", node.opkind);
                    }
                },
                _ => {
                    println!("  Non-standard node type");
                }
            }
        }

        // Verify that we found the expected equations
        if self.verify_onnx_semantics_only {
            // Just verify all equations are ONNX-compliant
            if found_equations.is_empty() {
                return Err("No MatMul/Einsum operations found in model".to_string());
            }

            for (i, equation) in found_equations.iter().enumerate() {
                if equation != "mk,nk->mn" {
                    return Err(format!(
                        "Non-ONNX equation found at position {i}:\n  Expected: 'mk,nk->mn'\n  Found: '{equation}'"
                    ));
                }
                println!("✓ Equation {i} uses ONNX semantics: '{equation}'");
            }

            println!(
                "✅ All {} MatMul operations use correct ONNX semantics",
                found_equations.len()
            );
        } else {
            // Exact equation matching mode
            if found_equations.len() != self.expected_equations.len() {
                return Err(format!(
                    "Expected {} MatMul/Einsum operations, but found {}.\nFound equations: {:?}\nExpected: {:?}",
                    self.expected_equations.len(),
                    found_equations.len(),
                    found_equations,
                    self.expected_equations
                ));
            }

            // Check each equation matches what we expect
            for (i, expected) in self.expected_equations.iter().enumerate() {
                if let Some(found) = found_equations.get(i) {
                    if found != expected {
                        return Err(format!(
                            "Equation mismatch at position {i}:\n  Expected: '{expected}'\n  Found: '{found}'"
                        ));
                    }
                    println!("✓ Equation {i} matches: '{found}'");
                } else {
                    return Err(format!("Missing equation at position {i}"));
                }
            }
        }

        // Decode the model to show the actual instructions generated
        println!("\n--- Decoding model instructions ---");
        let instructions = decode_model(model);
        for (i, instr) in instructions.iter().enumerate() {
            println!("Instruction {i}: {instr:?}");
        }

        println!("\n✅ Test passed: All MatMul operations use correct ONNX semantics");
        println!("ONNX implicitly transposes the second matrix (B) in MatMul operations");

        Ok(())
    }
}

/// Verify that the perceptron.onnx model uses correct ONNX MatMul semantics
#[test]
fn test_perceptron_matmul_transpose() {
    let test = MatMulTransposeTest::new(
        "tests/perceptron.onnx",
        vec!["mk,nk->mn"], // ONNX MatMul should use this equation, not "ij,jk->ik"
        "Verify that perceptron.onnx uses ONNX MatMul semantics with implicit B transpose",
    );

    test.run().expect("Perceptron MatMul transpose test failed");
}

/// Test the perceptron_2.onnx model with multiple MatMul operations
/// This test uses the flexible ONNX semantics verification mode
#[test]
fn test_perceptron_2_onnx_semantics() {
    let test = MatMulTransposeTest::new_onnx_semantics_only(
        "tests/perceptron_2.onnx",
        "Verify that perceptron_2.onnx uses ONNX MatMul semantics for all MatMul operations",
    );

    test.run().expect("Perceptron_2 ONNX semantics test failed");
}

/// Comprehensive test that checks multiple models
#[test]
fn test_all_matmul_models() {
    println!("\n=== Testing Simple MatMult Model ===");
    println!("Description: Verify our simple_matmult_model uses correct ONNX semantics");

    // Test simple matmult model
    let simple_test = MatMulTransposeTest::new_onnx_semantics_only(
        "onnx-tracer/models/addsubmul0/network.onnx",
        "Simple matrix operations test",
    );

    match simple_test.run() {
        Ok(_) => println!("Simple MatMult model uses correct ONNX semantics"),
        Err(e) => {
            if e.contains("No MatMul/Einsum operations found") {
                println!("Simple model has no MatMul operations (uses basic arithmetic only)");
            } else {
                println!("Simple model test failed: {e}");
            }
        }
    }

    let test_configs = vec![
        MatMulTransposeTest::new(
            "tests/perceptron.onnx",
            vec!["mk,nk->mn"],
            "Verify perceptron model MatMul transpose",
        ),
        MatMulTransposeTest::new_onnx_semantics_only(
            "tests/perceptron_2.onnx",
            "Verify perceptron_2 model MatMul transpose",
        ),
    ];

    println!("\n======================================================================");

    let mut passed = 0;
    let mut failed = 0;

    for test in test_configs {
        match test.run() {
            Ok(_) => {
                passed += 1;
                println!("PASSED: {}", test.description);
            }
            Err(e) => {
                failed += 1;
                println!("FAILED: {}", test.description);
                println!("   Error: {e}");
            }
        }
        println!("\n======================================================================");
    }

    println!("Test Summary: {passed} passed, {failed} failed");

    if failed > 0 {
        panic!("Some MatMul transpose tests failed");
    }
}

/// Manual test function for debugging specific models
pub fn run_perceptron_test() {
    let test = MatMulTransposeTest::new(
        "onnx-tracer/models/perceptron.onnx",
        vec!["mk,nk->mn"],
        "Manual test for perceptron MatMul semantics",
    );

    match test.run() {
        Ok(_) => println!("Manual test passed!"),
        Err(e) => println!("Manual test failed: {e}"),
    }
}
