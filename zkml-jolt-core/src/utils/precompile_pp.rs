//! Utility functions for preprocessing precompile operands

use crate::jolt::bytecode::BytecodePreprocessing;
use itertools::Itertools;
use jolt_core::{
    field::JoltField,
    poly::{eq_poly::EqPolynomial, multilinear_polynomial::MultilinearPolynomial},
    utils::thread::unsafe_allocate_zero_vec,
};
use onnx_tracer::trace_types::ONNXInstr;
use std::collections::HashMap;

/// Helper functions for common precompile preprocessing operations
pub struct PreprocessingHelper;

impl PreprocessingHelper {
    /// Calculate padded dimensions to the next power of two
    pub fn calculate_padded_dimensions(k: usize, n: usize) -> (usize, usize) {
        (Self::pad_usize(k), Self::pad_usize(n))
    }

    /// Pad a usize to the next power of two if it is not already a power of two
    pub fn pad_usize(x: usize) -> usize {
        if x.is_power_of_two() {
            x
        } else {
            x.next_power_of_two()
        }
    }

    /// Extract operand instruction from td_lookup
    pub fn get_operand_instruction<'a>(
        td_lookup: &'a HashMap<usize, &ONNXInstr>,
        ts: Option<usize>,
        operation_name: &str,
    ) -> &'a ONNXInstr {
        let ts = ts.unwrap_or_else(|| panic!("{operation_name} instruction missing operand"));
        td_lookup
            .get(&ts)
            .unwrap_or_else(|| panic!("Missing instruction for td {ts}"))
    }

    /// Collect and resize addresses for a vector operand
    pub fn collect_and_resize_vector(
        instr: &ONNXInstr,
        bytecode_preprocessing: &BytecodePreprocessing,
        target_size: usize,
    ) -> Vec<usize> {
        let mut addresses = bytecode_preprocessing.collect_addresses(instr);
        addresses.resize(target_size, 0);
        addresses
    }

    /// Collect and pad addresses for a matrix operand
    pub fn collect_and_pad_matrix(
        instr: &ONNXInstr,
        bytecode_preprocessing: &BytecodePreprocessing,
        original_m: usize,
        m_padded: usize,
        n_padded: usize,
    ) -> Vec<usize> {
        let addresses = bytecode_preprocessing.collect_addresses(instr);
        Self::pad_matrix(addresses, original_m, m_padded, n_padded)
    }

    /// Pad the collected addresses of the matrix.
    /// This requires interpreting the flattened addresses as a 2D matrix and padding
    /// both dimensions to the next power of two.
    /// - `matrix`: Flattened vector of addresses representing the matrix
    /// - `k`: Original number of columns in the matrix
    /// - `n_padded`: Padded number of rows (next power of two)
    /// - `k_padded`: Padded number of columns (next power of two)
    ///
    /// Returns the padded flattened vector of addresses
    fn pad_matrix(matrix: Vec<usize>, m: usize, m_padded: usize, n_padded: usize) -> Vec<usize> {
        let mut matrix: Vec<Vec<usize>> = matrix.chunks(m).map(|chunk| chunk.to_vec()).collect();
        // Pad each row to k_padded
        for row in matrix.iter_mut() {
            row.resize(m_padded, 0);
        }
        // Pad rows to n_padded
        matrix.resize(n_padded, vec![0; m_padded]);
        // Flatten back to 1D
        matrix.into_iter().flatten().collect()
    }
}

/// Trait for preprocessing precompile operands
/// - `extract_rv`: Extracts read-values from val_final using the stored read-addresses.
pub trait PrecompilePreprocessingTrait {
    /// Extracts read-values from val_final using the stored read-addresses.
    /// Given the val_final lookup table, computes the actual values of operands and results
    /// by indexing into val_final with the preprocessed memory addresses.
    fn extract_rv<T>(&self, val_final: &[i64], field_selector: impl Fn(&Self) -> &T) -> Vec<i64>
    where
        T: AsRef<[usize]>,
    {
        field_selector(self)
            .as_ref()
            .iter()
            .map(|&k| val_final[k])
            .collect_vec()
    }

    /// From the addresses compute the one-hot poly needed for the read-checking instance
    fn compute_ra<F, T>(
        &self,
        r: &[F],
        field_selector: impl Fn(&Self) -> &T,
        K: usize,
    ) -> MultilinearPolynomial<F>
    where
        T: AsRef<[usize]>,
        F: JoltField,
    {
        let E = EqPolynomial::evals(r);
        let mut ra = unsafe_allocate_zero_vec::<F>(K);
        field_selector(self)
            .as_ref()
            .iter()
            .enumerate()
            .for_each(|(j, &k)| ra[k] += E[j]);
        MultilinearPolynomial::from(ra)
    }
}
