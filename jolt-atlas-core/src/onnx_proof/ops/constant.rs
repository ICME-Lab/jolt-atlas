use atlas_algebra::field::JoltField;
pub use jolt_atlas_verifier::onnx_proof::ops::constant::*;

/// The claims of a [`public_value`]: none, since it produces no opening.
pub fn public_value_claims<F: JoltField>() -> Vec<F> {
    Vec::new()
}
