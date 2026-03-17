//! Precision-test helpers for comparing quantized operator outputs to expected references.

use crate::tensor::Tensor;

/// Compare quantized `actual` vs `expected`, print summary stats, and enforce a worst-error bound.
///
/// Intended for operator precision tests where outputs are fixed-point with scale `scale`.
pub fn assert_quantized_precision(
    label: &str,
    input: &Tensor<i32>,
    actual: &[i32],
    expected: &[i32],
    scale: f64,
    input_range: (i32, i32),
    worst_error_bound_quantized: i32,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: actual/expected length mismatch"
    );
    assert_eq!(
        input.len(),
        actual.len(),
        "{label}: input/actual length mismatch"
    );

    let mut total_abs_error_quantized: i64 = 0;
    let mut worst_error_quantized: i32 = 0;
    let mut worst_error_index: usize = 0;

    for (idx, (actual_i, expected_i)) in actual.iter().zip(expected.iter()).enumerate() {
        let err = (actual_i - expected_i).abs();
        total_abs_error_quantized += err as i64;
        if err > worst_error_quantized {
            worst_error_quantized = err;
            worst_error_index = idx;
        }
    }

    let sample_size = actual.len();
    let avg_abs_error_quantized = total_abs_error_quantized as f64 / sample_size as f64;
    let avg_abs_error_real = avg_abs_error_quantized / scale;
    let worst_error_real = worst_error_quantized as f64 / scale;

    println!(
        "{label} precision stats (n={sample_size}, input_range=[{}, {})): \
        \nAverage error (Quantized)={avg_abs_error_quantized:.4},\
        \nWorst error (Quantized)={worst_error_quantized},\
        \nAverage error (Real)={avg_abs_error_real:.6},\
        \nWorst error (Real)={worst_error_real:.6},\
        \nWorst case=(idx={worst_error_index}, input={}, actual={}, expected={})",
        input_range.0,
        input_range.1,
        input[worst_error_index],
        actual[worst_error_index],
        expected[worst_error_index]
    );

    assert!(
        worst_error_quantized <= worst_error_bound_quantized,
        "{label}: worst quantized error too high: worst={worst_error_quantized}, \
        \nbound={worst_error_bound_quantized}, avg={avg_abs_error_quantized:.4}, \
        \ninput={}, actual={}, expected={}",
        input[worst_error_index],
        actual[worst_error_index],
        expected[worst_error_index]
    );
}
