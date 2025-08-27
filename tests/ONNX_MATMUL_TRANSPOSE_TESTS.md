# ONNX MatMul Implicit Transpose Test Suite

This test suite verifies that the ONNX-tracer correctly handles ONNX MatMul semantics, specifically the implicit transpose of the second matrix (B) in MatMul operations.

## Background

ONNX MatMul operations use different semantics compared to standard mathematical notation:
- **Standard math**: `C = A × B` uses einsum equation `"ij,jk->ik"`  
- **ONNX semantics**: `C = A × B` uses einsum equation `"mk,nk->mn"` (implicitly transposes B)

This means that for matrices A[m,k] and B[n,k], ONNX MatMul computes `A × B^T`, where B^T is the transpose of B.

## Test Results
 **perceptron.onnx**: Uses correct equation `"mk,nk->mn"` (wrapped in RebaseScale)  
**simple_matmult_model**: Uses correct equation `"mk,nk->mn"`  
**All tests passing**: ONNX implicit transpose behavior is correctly implemented

## Running Tests

```bash
# Run individual tests with output
cargo test test_perceptron_matmul_transpose -- --nocapture

# Run all MatMul transpose tests
cargo test --test onnx_matmul_transpose_test

# Run comprehensive test suite with detailed output
cargo test test_all_matmul_models -- --nocapture
```

## Test Structure

- **`MatMulTransposeTest`**: Configurable test framework for verifying einsum equations
- **`test_perceptron_matmul_transpose()`**: Verifies perceptron.onnx uses correct ONNX semantics  
- **`test_simple_matmult_model_transpose()`**: Verifies our model builder creates correct semantics
- **`test_all_matmul_models()`**: Comprehensive test runner for multiple models

1. **RebaseScale Wrapping**: ONNX models may wrap MatMul operations in `RebaseScale` for quantization
2. **Equation Verification**: Both direct `Einsum` operations and wrapped ones use `"mk,nk->mn"`
3. **ONNX Compliance**: All MatMul operations correctly implement ONNX implicit transpose semantics

## Adding New Models

To test additional ONNX models:

1. Add a new `MatMulTransposeTest` to the `test_all_matmul_models()` function:
   ```rust
   MatMulTransposeTest::new(
       "path/to/your_model.onnx",
       vec!["mk,nk->mn"], // Expected equations
       "Description of your test"
   ),
   ```


