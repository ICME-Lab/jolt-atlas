//! Dimension handling utilities for normalizing einsum and sum patterns.
//!
//! `atlas-onnx-import` lowers every einsum to one of four canonical
//! full-letter equations (`CANONICAL_EINSUMS` in its `lower.rs`); the
//! [`EINSUM_REGISTRY`] here maps each onto a prover layout family and extracts
//! the flattened dimensions that layout folds over. The module also normalizes
//! multi-dimensional sum operations to a 2D representation.

use atlas_trace::{Model, Op};

use crate::node_view::ComputationNode;
use crate::node_view::ModelView;
use atlas_algebra::{field::JoltField, utils::thread::unsafe_allocate_zero_vec};
use rayon::prelude::*;

/// Convert a row-major linear index into coordinates for the given dimensions.
pub fn linear_to_coord(mut index: usize, dims: &[usize]) -> Vec<usize> {
    let mut coord = vec![0; dims.len()];
    for axis in (0..dims.len()).rev() {
        coord[axis] = index % dims[axis];
        index /= dims[axis];
    }
    coord
}

/// Convert row-major coordinates into a linear index for the given dimensions.
pub fn coord_to_linear(coord: &[usize], dims: &[usize]) -> usize {
    let mut index = 0usize;
    let mut stride = 1usize;
    for axis in (0..dims.len()).rev() {
        index += coord[axis] * stride;
        stride *= dims[axis];
    }
    index
}

/// Function type for extracting dimension information from a computation node.
pub type DimExtractor = fn(&ComputationNode, &Model) -> EinsumDims;

/// Configuration for different einsum equation types
#[derive(Debug, Clone)]
pub struct EinsumConfig {
    /// The einsum equation pattern in standard notation (e.g., "mk,kn->mn")
    pub equation: &'static str,
    /// Function to extract dimensions from a computation node
    pub dims_extractor: DimExtractor,
}

/// Registry mapping the canonical einsum equations to their prover layouts.
///
/// The keys are exactly the equations `atlas-onnx-import` can emit
/// (`CANONICAL_EINSUMS`), plus `k,nk->n` — the layout-canonical name of the
/// vector-matrix family that `mk,nk->mn` normalizes into. Stored as a flat
/// slice and scanned linearly by [`lookup_einsum_config`].
pub static EINSUM_REGISTRY: &[(&str, EinsumConfig)] = &[
    (
        "mk,kn->mn",
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
        // Transposed-weight matmul; the k,nk->n layout is vector-matrix, so
        // the extractor enforces m == 1.
        "mk,nk->mn",
        EinsumConfig {
            equation: "k,nk->n",
            dims_extractor: extract_k_nk_n_dims,
        },
    ),
    (
        // Rank-3 batched matmul: identical memory layout to abmk,abnk->abmn
        // with a batch pack of (1, a).
        "amk,ank->amn",
        EinsumConfig {
            equation: "abmk,abnk->abmn",
            dims_extractor: extract_amk_ank_amn_dims,
        },
    ),
    (
        "abmk,abnk->abmn",
        EinsumConfig {
            equation: "abmk,abnk->abmn",
            dims_extractor: extract_abmk_abnk_abmn_dims,
        },
    ),
];

/// Look up the canonical [`EinsumConfig`] for a raw ONNX einsum `equation`.
///
/// Panics if the pattern is not registered, i.e. not supported by the Einsum
/// proof system.
pub fn lookup_einsum_config(equation: &str) -> &'static EinsumConfig {
    EINSUM_REGISTRY
        .iter()
        .find(|(pattern, _)| *pattern == equation)
        .map(|(_, config)| config)
        .unwrap_or_else(|| {
            panic!("Einsum equation ({equation}) not supported by Einsum proof system")
        })
}

fn extract_mk_kn_mn_dims(computation_node: &ComputationNode, model: &Model) -> EinsumDims {
    let [a_idx, b_idx] = computation_node.inputs[..] else {
        panic!("Expected exactly two inputs for mk,kn->mk operation")
    };
    let _a_node = &model.view(a_idx);
    let b_node = &model.view(b_idx);
    assert_eq!(
        computation_node.output_dims.len(),
        2,
        "mk,kn->mn expects a rank-2 output"
    );
    let m = computation_node.output_dims[0];
    let k = b_node.output_dims[0];
    let n = b_node.output_dims[1];
    EinsumDims::new(vec![m, k], vec![k, n], vec![m, n])
}

fn extract_k_nk_n_dims(computation_node: &ComputationNode, model: &Model) -> EinsumDims {
    let [a_idx, b_idx] = computation_node.inputs[..] else {
        panic!("Expected exactly two inputs for k,nk->n operation")
    };
    let a_node = &model.view(a_idx);
    let b_node = &model.view(b_idx);
    let n = b_node.output_dims[0];
    let k = b_node.output_dims[1];
    // The k,nk->n layout is strictly vector-matrix: it folds only the k axis,
    // so a left operand with m > 1 rows would feed m*k coefficients into a
    // log(k)-round sumcheck. Fail here, at dispatch, rather than deep inside
    // the dot engine.
    assert_eq!(
        a_node.output_dims.iter().product::<usize>(),
        k,
        "mk,nk->mn is only provable with m == 1 (vector-matrix); got left dims {:?}",
        a_node.output_dims
    );
    EinsumDims::new(vec![k], vec![n, k], vec![n])
}

fn extract_amk_ank_amn_dims(computation_node: &ComputationNode, model: &Model) -> EinsumDims {
    let [a_idx, b_idx] = computation_node.inputs[..] else {
        panic!("Expected exactly two inputs for amk,ank->amn operation")
    };
    let a_node = &model.view(a_idx);
    let b_node = &model.view(b_idx);
    let a = a_node.output_dims[0];
    let m = a_node.output_dims[1];
    let k = a_node.output_dims[2];
    let n = b_node.output_dims[1];
    EinsumDims::new(vec![a, m, k], vec![a, n, k], vec![a, m, n])
}

fn extract_abmk_abnk_abmn_dims(computation_node: &ComputationNode, model: &Model) -> EinsumDims {
    let [a_idx, b_idx] = computation_node.inputs[..] else {
        panic!("Expected exactly two inputs for abmk,abnk->abmn operation")
    };
    let a_node = &model.view(a_idx);
    let b_node = &model.view(b_idx);
    let a = a_node.output_dims[0];
    let b = a_node.output_dims[1];
    let m = a_node.output_dims[2];
    let k = a_node.output_dims[3];
    let n = b_node.output_dims[2];

    EinsumDims::new(vec![a * b, m, k], vec![a * b, n, k], vec![a * b, m, n])
}

/// Stores preprocessed dims (from the Model) for einsum equations
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumDims {
    left_operand: Vec<usize>,
    right_operand: Vec<usize>,
    output: Vec<usize>,
}

impl EinsumDims {
    /// Creates a new `EinsumDims` instance with the specified dimensions.
    pub fn new(left_operand: Vec<usize>, right_operand: Vec<usize>, output: Vec<usize>) -> Self {
        Self {
            left_operand,
            right_operand,
            output,
        }
    }

    /// Returns the dimensions of the left operand.
    pub fn left_operand(&self) -> &[usize] {
        &self.left_operand
    }

    /// Returns the dimensions of the right operand.
    pub fn right_operand(&self) -> &[usize] {
        &self.right_operand
    }

    /// Returns the dimensions of the output.
    pub fn output(&self) -> &[usize] {
        &self.output
    }
}

/// Complete configuration for a sum operation.
///
/// Contains both the normalized dimension information and the axis along which
/// the sum is performed (after normalization to 2D).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumConfig {
    /// Normalized dimension information
    pub dims: SumDims,
    /// The axis to sum along (0 or 1 after normalization to 2D)
    pub axis: SumAxis,
}

impl SumConfig {
    /// Creates a new `SumConfig` with the given dimensions and axis.
    pub fn new(dims: SumDims, axis: SumAxis) -> Self {
        Self { dims, axis }
    }

    /// Returns the normalized dimension information for the sum operation.
    pub fn dims(&self) -> &SumDims {
        &self.dims
    }

    /// Returns the axis along which the sum is performed (0 or 1).
    pub fn axis(&self) -> SumAxis {
        self.axis
    }

    /// Returns the operand dims for this sum operation.
    pub fn operand_dims(&self) -> &[usize] {
        self.dims.operand()
    }
}

impl SumDims {
    /// Creates a new `SumDims` with the given operand and output dimensions.
    pub fn new(operand: Vec<usize>, output: Vec<usize>) -> Self {
        Self { operand, output }
    }

    /// Returns the operand (input) dimensions.
    pub fn operand(&self) -> &[usize] {
        &self.operand
    }

    /// Returns the output dimensions.
    pub fn output(&self) -> &[usize] {
        &self.output
    }
}

/// Extracts and normalizes sum operation dimensions from a computation node.
///
/// This function processes a sum operation from the ONNX model, extracting the input/output
/// dimensions and normalizing them to a canonical 2D representation for the prover.
/// The normalization handles tensors of varying dimensionality (1D, 2D, 3D+) uniformly.
///
/// # Arguments
///
/// * `computation_node` - The computation node representing the sum operation
/// * `model` - The full model containing all nodes and their dimensions
///
/// # Returns
///
/// A `SumConfig` containing normalized dimensions and the adjusted axis index.
///
/// # Panics
///
/// - If the computation node doesn't have exactly one input
/// - If the operator is not a `Sum` operator
/// - If the sum operation has multiple axes (only single-axis sum is supported)
/// - If the axis is out of bounds for the input tensor's dimensionality
/// - If the input has 3+ dimensions and the leading dimension is not 1
pub fn sum_config(computation_node: &ComputationNode, model: &Model) -> SumConfig {
    // Extract the single input index
    let [input_idx] = computation_node.inputs[..] else {
        panic!("Expected exactly one input for Sum operation")
    };

    // Ensure we have a Sum operator and extract it
    let Op::Sum(sum_operator) = &computation_node.operator else {
        panic!("Expected Sum operator")
    };

    // Extract the single axis (multi-axis sum not yet supported)
    let [axis] = sum_operator.axes[..] else {
        unimplemented!("Only single-axis sum is currently supported")
    };

    // Get dimension information from the model
    let input_dims = &model.view(input_idx).output_dims;
    let output_dims = &computation_node.output_dims;
    let ndim = input_dims.len();

    // Validate axis is within bounds
    assert!(axis < ndim, "Axis {axis} out of bounds for {ndim}D tensor");

    // Normalize to 2D representation for the prover
    let (axis, operand_dims, output_dims) = normalize_sum_to_2d(axis, input_dims, output_dims);

    SumConfig {
        dims: SumDims::new(operand_dims, output_dims),
        axis,
    }
}

/// Recursively normalizes sum dimensions down to 2D
///
/// - **1D**: Appends a dummy dimension so it becomes a special case of 2D.
/// - **2D**: Already in canonical form — returned as-is.
/// - **3D+**: Strips the leading batch dimension (must be 1) and recurses,
///   so 4D, 5D, etc. are automatically supported without new match arms (as long as the batch dimension is 1).
///
/// # Panics
///
/// Panics if `input_dims` has 3 or more dimensions and the leading dimension is not 1.
fn normalize_sum_to_2d(
    axis: usize,
    input_dims: &[usize],
    output_dims: &[usize],
) -> (SumAxis, Vec<usize>, Vec<usize>) {
    match input_dims.len() {
        1 => (SumAxis::Axis0, vec![input_dims[0], 1], vec![1]),
        2 => (axis.into(), input_dims.to_vec(), output_dims.to_vec()),
        ndim => {
            if input_dims[0] != 1 {
                unimplemented!(
                    "Only batch size of 1 is supported for {ndim}D tensors in Sum operations, but got leading dimension of {}",
                    input_dims[0]
                );
            }
            normalize_sum_to_2d(axis - 1, &input_dims[1..], &output_dims[1..])
        }
    }
}

/// Represents the axis along which a sum operation is performed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumAxis {
    /// Sum along axis 0 (rows)
    Axis0,
    /// Sum along axis 1 (columns)
    Axis1,
}

impl SumAxis {
    /// Returns the numeric index of the axis (0 or 1).
    pub fn axis_index(&self) -> usize {
        match self {
            SumAxis::Axis0 => 0,
            SumAxis::Axis1 => 1,
        }
    }
}

impl From<usize> for SumAxis {
    fn from(axis: usize) -> Self {
        match axis {
            0 => SumAxis::Axis0,
            1 => SumAxis::Axis1,
            _ => panic!("Invalid axis index: {axis}"),
        }
    }
}

/// Dimension information for a sum operation.
///
/// Stores the operand (input) and output dimensions after normalization to 2D.
/// These dimensions are used by the prover to generate the appropriate constraints.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SumDims {
    /// Dimensions of the input tensor (normalized to 2D)
    pub operand: Vec<usize>,
    /// Dimensions of the output tensor (normalized to 2D)
    pub output: Vec<usize>,
}

/// Transposes a flattened matrix stored in row-major order.
///
/// # Arguments
/// * `flat_vector` - The input vector representing a flattened matrix in row-major order
/// * `num_rows` - The number of rows in the original matrix
/// * `num_cols` - The number of columns in the original matrix
///
/// # Returns
/// A new vector representing the transposed matrix in row-major order
pub fn transpose_flat_matrix<F: JoltField>(
    flat_vector: Vec<F>,
    num_rows: usize,
    num_cols: usize,
) -> Vec<F> {
    const MIN_SIZE_FOR_PARALLEL: usize = 1024;

    let mut transposed = unsafe_allocate_zero_vec(num_rows * num_cols);
    let total_size = num_rows * num_cols;

    if total_size >= MIN_SIZE_FOR_PARALLEL {
        transposed
            .par_chunks_mut(num_rows)
            .enumerate()
            .for_each(|(j, col)| {
                for i in 0..num_rows {
                    col[i] = flat_vector[i * num_cols + j];
                }
            });
    } else {
        for i in 0..num_rows {
            for j in 0..num_cols {
                transposed[j * num_rows + i] = flat_vector[i * num_cols + j];
            }
        }
    }
    transposed
}

#[cfg(test)]
mod tests {
    use super::EINSUM_REGISTRY;

    /// Every equation atlas-onnx-import can emit (its `CANONICAL_EINSUMS`)
    /// must be registered, and the batched forms must share one layout.
    #[test]
    fn test_registry_covers_canonical_importer_equations() {
        for pattern in ["mk,kn->mn", "mk,nk->mn", "amk,ank->amn", "abmk,abnk->abmn"] {
            EINSUM_REGISTRY
                .iter()
                .find(|(registered, _)| *registered == pattern)
                .unwrap_or_else(|| panic!("canonical importer equation {pattern} not registered"));
        }
        for pattern in ["amk,ank->amn", "abmk,abnk->abmn"] {
            let config = EINSUM_REGISTRY
                .iter()
                .find(|(registered, _)| *registered == pattern)
                .map(|(_, config)| config)
                .expect("expected pattern in einsum registry");
            assert_eq!(config.equation, "abmk,abnk->abmn");
        }
    }
}
