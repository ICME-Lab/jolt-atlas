use atlas_onnx_tracer::{model::Model, node::ComputationNode};

pub type DimExtractor = fn(&ComputationNode, &Model) -> EinsumDims;

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

fn extract_bmk_kbn_mbn_dims(computation_node: &ComputationNode, model: &Model) -> EinsumDims {
    let [a_idx, b_idx] = computation_node.inputs[..] else {
        panic!("Expected exactly two inputs for mbk,nbk->bmn operation")
    };
    let _a_node = &model[a_idx];
    let b_node = &model[b_idx];
    let m = computation_node.output_dims[0];
    let b = computation_node.output_dims[1];
    let n = computation_node.output_dims[2];
    let k = b_node.output_dims[0];
    EinsumDims::new(vec![b, m, k], vec![k, b, n], vec![m, b, n])
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
