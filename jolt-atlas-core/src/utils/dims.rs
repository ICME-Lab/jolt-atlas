use atlas_onnx_tracer::{model::Model, node::ComputationNode, ops::Operator};
use joltworks::{field::JoltField, utils::thread::unsafe_allocate_zero_vec};
use rayon::prelude::*;

pub type DimExtractor = fn(&ComputationNode, &Model) -> EinsumDims;

/// Configuration for different einsum equation types
#[derive(Debug, Clone)]
pub struct EinsumConfig {
    pub equation: &'static str,
    pub dims_extractor: DimExtractor,
}

/// Registry mapping einsum patterns to their configurations using a BTreeMap for O(log n) lookup
pub static EINSUM_REGISTRY: &[(&str, EinsumConfig)] = &[
    /* **************** mk,kn->mn **************** */
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
        "amk,kn->bmn",
        EinsumConfig {
            equation: "mk,kn->mn",
            dims_extractor: extract_mk_kn_mn_dims,
        },
    ),
    (
        "abmk,kn->mn",
        EinsumConfig {
            equation: "mk,kn->mn",
            dims_extractor: extract_mk_kn_mn_dims,
        },
    ),
    // Note: The equation "mbk,bkn->amn" contains two contraction dimensions (b and k).
    // However, since these dimensions appear consecutively in both operands (as "bk"),
    // we can flatten them into a single dimension and treat this as the simpler
    // "mk,kn->mn" pattern. This optimization reduces duplicate prover logic while
    // maintaining mathematical correctness.
    (
        "mbk,bkn->amn",
        EinsumConfig {
            equation: "mk,kn->mn",
            dims_extractor: extract_mbk_bkn_amn_dims,
        },
    ),
    (
        "mbk,bkn->mn",
        EinsumConfig {
            equation: "mk,kn->mn",
            dims_extractor: extract_mbk_bkn_amn_dims,
        },
    ),
    /* **************** k,nk->n **************** */
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
        "ak,k->mn",
        EinsumConfig {
            equation: "k,nk->n",
            dims_extractor: extract_ak_k_mn_dims,
        },
    ),
    /* **************** mbk,nbk->bmn **************** */
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
    /* **************** bmk,kbn->mbn **************** */
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
    /* **************** bmk,bkn->mbn **************** */
    (
        "abmk,abkn->mbn",
        EinsumConfig {
            equation: "bmk,bkn->mbn",
            dims_extractor: extract_bmk_bkn_mbn_dims,
        },
    ),
    (
        "bmk,bkn->mbn",
        EinsumConfig {
            equation: "bmk,bkn->mbn",
            dims_extractor: extract_bmk_bkn_mbn_dims,
        },
    ),
    /* **************** mbk,bnk->bmn **************** */
    (
        "mbk,bnk->bmn",
        EinsumConfig {
            equation: "mbk,bnk->bmn",
            dims_extractor: extract_mbk_bnk_bmn_dims,
        },
    ),
    (
        "mbk,bnk->abmn",
        EinsumConfig {
            equation: "mbk,bnk->bmn",
            dims_extractor: extract_mbk_bnk_bmn_dims,
        },
    ),
    (
        "mbk,abnk->abmn",
        EinsumConfig {
            equation: "mbk,bnk->bmn",
            dims_extractor: extract_mbk_bnk_bmn_dims,
        },
    ),
];

fn extract_mk_kn_mn_dims(computation_node: &ComputationNode, model: &Model) -> EinsumDims {
    let [a_idx, b_idx] = computation_node.inputs[..] else {
        panic!("Expected exactly two inputs for mk,kn->mk operation")
    };
    let _a_node = &model[a_idx];
    let b_node = &model[b_idx];
    let m = if computation_node.output_dims.len() == 3 {
        computation_node.output_dims[1]
    } else {
        computation_node.output_dims[0]
    };
    let k = b_node.output_dims[0];
    let n = b_node.output_dims[1];
    EinsumDims::new(vec![m, k], vec![k, n], vec![m, n])
}

fn extract_k_nk_n_dims(computation_node: &ComputationNode, model: &Model) -> EinsumDims {
    let [a_idx, b_idx] = computation_node.inputs[..] else {
        panic!("Expected exactly two inputs for k,nk->n operation")
    };
    let _a_node = &model[a_idx];
    let b_node = &model[b_idx];
    let n = b_node.output_dims[0];
    let k = b_node.output_dims[1];
    EinsumDims::new(vec![k], vec![n, k], vec![n])
}

fn extract_ak_k_mn_dims(computation_node: &ComputationNode, model: &Model) -> EinsumDims {
    let [a_idx, b_idx] = computation_node.inputs[..] else {
        panic!("Expected exactly two inputs for k,nk->n operation")
    };
    let _a_node = &model[a_idx];
    let b_node = &model[b_idx];
    let k = b_node.output_dims[0];
    EinsumDims::new(vec![k], vec![k], vec![1, 1])
}

fn extract_mbk_nbk_bmn_dims(computation_node: &ComputationNode, model: &Model) -> EinsumDims {
    let [a_idx, b_idx] = computation_node.inputs[..] else {
        panic!("Expected exactly two inputs for mbk,nbk->bmn operation")
    };
    let a_node = &model[a_idx];
    let b_node = &model[b_idx];
    let m = a_node.output_dims[0];
    let b = a_node.output_dims[1];
    let k = a_node.output_dims[2];
    let n = b_node.output_dims[0];
    EinsumDims::new(vec![m, b, k], vec![n, b, k], vec![b, m, n])
}

fn extract_mbk_bnk_bmn_dims(computation_node: &ComputationNode, model: &Model) -> EinsumDims {
    let [a_idx, b_idx] = computation_node.inputs[..] else {
        panic!("Expected exactly two inputs for mbk,bnk->bmn operation")
    };
    let a_node = &model[a_idx];
    let _b_node = &model[b_idx];
    let m = a_node.output_dims[0];
    let b = a_node.output_dims[1];
    let k = a_node.output_dims[2];
    let n = computation_node
        .output_dims
        .last()
        .copied()
        .expect("Expected at least 1 output dimension for mbk,bnk->bmn operation");
    EinsumDims::new(vec![m, b, k], vec![b, n, k], vec![b, m, n])
}

fn extract_bmk_kbn_mbn_dims(computation_node: &ComputationNode, model: &Model) -> EinsumDims {
    let [a_idx, b_idx] = computation_node.inputs[..] else {
        panic!("Expected exactly two inputs for bmk,kbn->mbn operation")
    };
    let _a_node = &model[a_idx];
    let b_node = &model[b_idx];
    let m = computation_node.output_dims[0];
    let b = computation_node.output_dims[1];
    let n = computation_node.output_dims[2];
    let k = b_node.output_dims[0];
    EinsumDims::new(vec![b, m, k], vec![k, b, n], vec![m, b, n])
}

fn extract_bmk_bkn_mbn_dims(computation_node: &ComputationNode, model: &Model) -> EinsumDims {
    let [a_idx, b_idx] = computation_node.inputs[..] else {
        panic!("Expected exactly two inputs for bmk,bkn->mbn operation")
    };
    let a_node = &model[a_idx];
    let _b_node = &model[b_idx];
    let m = computation_node.output_dims[0];
    let b = computation_node.output_dims[1];
    let n = computation_node.output_dims[2];
    let k = a_node
        .output_dims
        .last()
        .copied()
        .expect("Expected at least 1 dimension for a_node in bmk,bkn->mbn operation");
    EinsumDims::new(vec![b, m, k], vec![b, k, n], vec![m, b, n])
}

fn extract_mbk_bkn_amn_dims(computation_node: &ComputationNode, model: &Model) -> EinsumDims {
    let [a_idx, b_idx] = computation_node.inputs[..] else {
        panic!("Expected exactly two inputs for mbk,bkn->amn operation")
    };
    let a_node = &model[a_idx];
    let b_node = &model[b_idx];
    let m = a_node.output_dims[0];
    let b = a_node.output_dims[1];
    let k = a_node.output_dims[2];
    let n = b_node.output_dims[2];

    let bk = b * k;

    EinsumDims::new(vec![m, bk], vec![bk, n], vec![m, n])
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Stores preprocessed dims (from the Model) for einsum equations
pub struct EinsumDims {
    left_operand: Vec<usize>,
    right_operand: Vec<usize>,
    output: Vec<usize>,
}

impl EinsumDims {
    pub fn new(left_operand: Vec<usize>, right_operand: Vec<usize>, output: Vec<usize>) -> Self {
        Self {
            left_operand,
            right_operand,
            output,
        }
    }

    pub fn left_operand(&self) -> &[usize] {
        &self.left_operand
    }

    pub fn right_operand(&self) -> &[usize] {
        &self.right_operand
    }

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
    let Operator::Sum(sum_operator) = &computation_node.operator else {
        panic!("Expected Sum operator")
    };

    // Extract the single axis (multi-axis sum not yet supported)
    let [axis] = sum_operator.axes[..] else {
        unimplemented!("Only single-axis sum is currently supported")
    };

    // Get dimension information from the model
    let input_dims = &model[input_idx].output_dims;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumAxis {
    Axis0,
    Axis1,
}

impl SumAxis {
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
