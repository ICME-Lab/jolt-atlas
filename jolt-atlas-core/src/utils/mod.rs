//! Utility modules for dimension handling and tensor operations.

use atlas_onnx_tracer::tensor::Tensor;
use common::consts::XLEN;
use joltworks::utils::{interleave_bits, lookup_bits::LookupBits};
use rayon::prelude::*;

pub mod dims;

/// Computes lookup table indices from operand tensors.
///
/// # Arguments
/// * `operand_tensors` - Slice of operand tensors to compute indices from
/// * `is_interleaved_operands` - If true, expects 2 tensors and interleaves their bits
///
/// # Returns
/// A vector of `LookupBits` representing the lookup indices for each element.
///
/// # Panics
/// Panics if `is_interleaved_operands` is true but the number of operand tensors is not 2.
pub fn compute_lookup_indices_from_operands(
    operand_tensors: &[&Tensor<i32>],
    is_interleaved_operands: bool,
) -> Vec<LookupBits> {
    if is_interleaved_operands {
        // Interleaved mode: requires exactly 2 operand tensors
        assert_eq!(
            operand_tensors.len(),
            2,
            "Interleaved operands mode requires exactly 2 input tensors, but got {}",
            operand_tensors.len()
        );

        let left_operand = operand_tensors[0];
        let right_operand = operand_tensors[1];

        // Validate that both tensors have the same length
        assert_eq!(
            left_operand.len(),
            right_operand.len(),
            "Interleaved operands must have the same length: left={}, right={}",
            left_operand.len(),
            right_operand.len()
        );

        // Interleave bits from both operands to form lookup indices
        left_operand
            .data()
            .par_iter()
            .zip(right_operand.data().par_iter())
            .map(|(&left_val, &right_val)| {
                // Cast to u64 for interleaving
                let left_bits = left_val as u32;
                let right_bits = right_val as u32;
                let interleaved = interleave_bits(left_bits, right_bits);
                LookupBits::new(interleaved, XLEN * 2)
            })
            .collect()
    } else {
        // Single operand mode: requires exactly 1 input tensor
        assert_eq!(
            operand_tensors.len(),
            1,
            "Single operand mode requires exactly 1 input tensor, but got {}",
            operand_tensors.len()
        );

        let operand = operand_tensors[0];

        // Use tensor values directly as lookup indices
        operand
            .data()
            .par_iter()
            .map(|&value| {
                // Cast to u64 for consistent bit representation
                let index = value as u32 as u64;
                LookupBits::new(index, XLEN * 2)
            })
            .collect()
    }
}
