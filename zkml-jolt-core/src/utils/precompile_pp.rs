//! Utility functions for preprocessing precompile operands

use crate::jolt::bytecode::BytecodePreprocessing;
use onnx_tracer::{tensor::Tensor, trace_types::ONNXInstr};
use std::collections::HashMap;

/// Helper functions for common precompile preprocessing operations
pub struct PreprocessingHelper;

impl PreprocessingHelper {
    /// Calculate padded dimensions to the next power of two
    pub fn calculate_padded_dims(original_dims: &[usize]) -> Vec<usize> {
        original_dims
            .iter()
            .map(|&dim| Self::pad_usize(dim))
            .collect()
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
        td_lookup: &'a HashMap<usize, ONNXInstr>,
        ts: Option<usize>,
        operation_name: &str,
    ) -> &'a ONNXInstr {
        let ts = ts.unwrap_or_else(|| panic!("{operation_name} instruction missing operand"));
        td_lookup
            .get(&ts)
            .unwrap_or_else(|| panic!("Missing instruction for td {ts}"))
    }

    /// Collect and pad addresses for a matrix operand
    pub fn collect_and_pad(
        instr: &ONNXInstr,
        bytecode_preprocessing: &BytecodePreprocessing,
        original_dims: &[usize],
    ) -> Vec<usize> {
        let addresses = bytecode_preprocessing.collect_addresses(instr);
        Self::pad_vec_usize(&addresses, original_dims)
    }

    /// Pad a vector of usize to the next power-of-two dimensions
    pub fn pad_vec_usize(vec: &[usize], original_dims: &[usize]) -> Vec<usize> {
        let mut usize_tensor: Tensor<usize> =
            Tensor::new(Some(vec), original_dims).expect("dims should be correct");
        let padded_dims = Self::calculate_padded_dims(original_dims);
        usize_tensor
            .pad_to_dims(&padded_dims)
            .expect("padding sizes should be valid");
        usize_tensor.data().to_vec()
    }
}

pub type DimExtractor =
    fn(&ONNXInstr, &HashMap<usize, ONNXInstr>) -> (Vec<usize>, Vec<usize>, Vec<usize>);

/// Configuration for different einsum equation types
#[derive(Debug, Clone)]
pub struct EinsumConfig {
    pub equation: &'static str,
    pub dims_extractor: DimExtractor,
}

/// Registry mapping einsum patterns to their configurations using a BTreeMap for O(log n) lookup
pub static EINSUM_REGISTRY: &[(&str, EinsumConfig)] = &[
    (
        "mk,kn->mn",
        EinsumConfig {
            equation: "mk,kn->mn",
            dims_extractor: extract_mk_kn_mn_dims,
        },
    ),
    (
        "amk,kn->amn",
        EinsumConfig {
            equation: "mk,kn->mn",
            dims_extractor: extract_mk_kn_mn_dims,
        },
    ),
    (
        "amk,kn->mn",
        EinsumConfig {
            equation: "mk,kn->mn",
            dims_extractor: extract_mk_kn_mn_dims,
        },
    ),
    (
        "mk,kn->amn",
        EinsumConfig {
            equation: "mk,kn->mn",
            dims_extractor: extract_mk_kn_mn_dims,
        },
    ),
    (
        "k,nk->n",
        EinsumConfig {
            equation: "k,nk->n",
            dims_extractor: extract_k_nk_n_dims,
        },
    ),
    (
        "mk,nk->n",
        EinsumConfig {
            equation: "k,nk->n",
            dims_extractor: extract_k_nk_n_dims,
        },
    ),
    (
        "k,nk->mn",
        EinsumConfig {
            equation: "k,nk->n",
            dims_extractor: extract_k_nk_n_dims,
        },
    ),
    (
        "mk,nk->mn",
        EinsumConfig {
            equation: "k,nk->n",
            dims_extractor: extract_k_nk_n_dims,
        },
    ),
    (
        "mbk,nbk->bmn",
        EinsumConfig {
            equation: "mbk,nbk->bmn",
            dims_extractor: extract_mbk_nbk_bmn_dims,
        },
    ),
    (
        "mbk,nbk->abmn",
        EinsumConfig {
            equation: "mbk,nbk->bmn",
            dims_extractor: extract_mbk_nbk_bmn_dims,
        },
    ),
    (
        "bmk,kbn->mbn",
        EinsumConfig {
            equation: "bmk,kbn->mbn",
            dims_extractor: extract_bmk_kbn_mbn_dims,
        },
    ),
    (
        "abmk,kbn->mbn",
        EinsumConfig {
            equation: "bmk,kbn->mbn",
            dims_extractor: extract_bmk_kbn_mbn_dims,
        },
    ),
];

/// Dimension extraction functions for different einsum patterns
fn extract_mk_kn_mn_dims(
    instr: &ONNXInstr,
    td_lookup: &HashMap<usize, ONNXInstr>,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let _a_instr = PreprocessingHelper::get_operand_instruction(td_lookup, instr.ts1, "MatMult");
    let b_instr = PreprocessingHelper::get_operand_instruction(td_lookup, instr.ts2, "MatMult");

    let m = if instr.output_dims.len() == 3 {
        instr.output_dims[1]
    } else {
        instr.output_dims[0]
    };
    let k = b_instr.output_dims[0];
    let n = b_instr.output_dims[1];

    (vec![m, k], vec![k, n], vec![m, n])
}

fn extract_k_nk_n_dims(
    instr: &ONNXInstr,
    td_lookup: &HashMap<usize, ONNXInstr>,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let _a_instr = PreprocessingHelper::get_operand_instruction(td_lookup, instr.ts1, "k,nk->n");
    let b_instr = PreprocessingHelper::get_operand_instruction(td_lookup, instr.ts2, "k,nk->n");

    let n = b_instr.output_dims[0];
    let k = b_instr.output_dims[1];

    (vec![k], vec![n, k], vec![n])
}

fn extract_mbk_nbk_bmn_dims(
    instr: &ONNXInstr,
    td_lookup: &HashMap<usize, ONNXInstr>,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let a_instr =
        PreprocessingHelper::get_operand_instruction(td_lookup, instr.ts1, "mbk,nbk->bmn");
    let b_instr =
        PreprocessingHelper::get_operand_instruction(td_lookup, instr.ts2, "mbk,nbk->bmn");

    let m = a_instr.output_dims[0];
    let b = a_instr.output_dims[1];
    let k = a_instr.output_dims[2];
    let n = b_instr.output_dims[0];

    (vec![m, b, k], vec![n, b, k], vec![b, m, n])
}

fn extract_bmk_kbn_mbn_dims(
    instr: &ONNXInstr,
    td_lookup: &HashMap<usize, ONNXInstr>,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let _a_instr =
        PreprocessingHelper::get_operand_instruction(td_lookup, instr.ts1, "bmk,kbn->mbn");
    let b_instr =
        PreprocessingHelper::get_operand_instruction(td_lookup, instr.ts2, "bmk,kbn->mbn");

    let m = instr.output_dims[0];
    let b = instr.output_dims[1];
    let n = instr.output_dims[2];
    let k = b_instr.output_dims[0];

    (vec![b, m, k], vec![k, b, n], vec![m, b, n])
}
