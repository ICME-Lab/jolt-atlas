//! A shape-class subset of the op-surface table under Dory (the full table
//! runs under Akita). One case per committed-polynomial shape class rather
//! than one per op.

use joltworks::poly::commitment::dory::DoryScheme;

use super::op_surface::run_named;

/// Fused rescale: remainder + clamp one-hot chunks and dense polys.
#[test]
fn dory_mul_random_1k() {
    run_named::<DoryScheme>("mul_random_1k");
}

/// The smallest polynomials (`num_vars` below Dory's column split).
#[test]
fn dory_add_tiny_t8() {
    run_named::<DoryScheme>("add_tiny_t8");
}

/// Sparse one-hot commitments with `K = dict rows`.
#[test]
fn dory_gather_small_dict32() {
    run_named::<DoryScheme>("gather_small_dict32");
}

/// The matmul whale shape class.
#[test]
fn dory_einsum_mk_kn_mn_16x32x64() {
    run_named::<DoryScheme>("einsum_mk_kn_mn_16x32x64");
}

/// Large-table activation one-hots.
#[test]
fn dory_tanh_random_1k() {
    run_named::<DoryScheme>("tanh_random_1k");
}

/// The most heterogeneous committed-poly set (four softmax stages).
#[test]
fn dory_softmax_model_scale_2x8() {
    run_named::<DoryScheme>("softmax_model_scale_2x8");
}
