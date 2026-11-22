//! Provers for the Atlas model's Booleanity and digit range checks.

use atlas_algebra::utils::errors::ProofVerifyError;
use jolt_atlas_verifier::onnx_proof::coefficient_checks::CoefficientCheck;

mod booleanity;
mod digit_range;

pub(super) use booleanity::BooleanityProver;
pub(super) use digit_range::DigitRangeProver;

#[cfg(test)]
mod tests;

fn invalid(message: &str) -> ProofVerifyError {
    ProofVerifyError::Malformed(message.to_owned())
}
