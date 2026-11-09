//! Division helpers for neural teleportation (Cos/Sin's `a mod P` domain reduction).
//!
//! `a = P·q + r` holds exactly, elementwise, so as multilinear polynomials
//! `ã(X) = P·q̃(X) + r̃(X)` holds identically everywhere, not just on the boolean hypercube.
//! Once `q` and `r` each have an evaluation claim at the same point, deriving `a`'s claim is
//! pure field arithmetic — no sumcheck is needed to enforce the identity.

use atlas_trace::Tensor;
pub use jolt_atlas_verifier::onnx_proof::neural_teleport::division::*;

/// Computes quotient and remainder for neural teleportation division.
/// Returns (quotient, remainder) where input = DIVISOR * quotient + remainder
pub fn compute_division(input: &Tensor<i32>, tau: i32) -> (Tensor<i32>, Tensor<i32>) {
    let (remainder_data, quotient_data): (Vec<i32>, Vec<i32>) = input
        .iter()
        .map(|&x| {
            let mut r = x % tau;
            let mut q = x / tau;
            // Ensure remainder has same sign as divisor (Euclidean division)
            if (r < 0 && tau > 0) || (r > 0 && tau < 0) {
                r += tau;
                q -= 1;
            }
            (r, q)
        })
        .unzip();

    let quotient = Tensor::<i32>::construct(quotient_data, input.dims().to_vec());
    let remainder = Tensor::<i32>::construct(remainder_data, input.dims().to_vec());

    (quotient, remainder)
}
