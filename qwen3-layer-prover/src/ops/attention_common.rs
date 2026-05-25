use joltworks::{field::JoltField, utils::errors::ProofVerifyError};

use crate::{
    claim::{LegacyClaim, Shape},
    error::{ProverError, Result},
};

pub(crate) const QWEN3_GQA_GROUP_SIZE: usize = 2;

pub(crate) fn validate_gqa(q_heads: usize, kv_heads: usize) -> Result<()> {
    if kv_heads == 0 || q_heads / kv_heads != QWEN3_GQA_GROUP_SIZE {
        return Err(ProverError::InvalidGqa { q_heads, kv_heads });
    }
    Ok(())
}

pub(crate) fn validate_claim<F: JoltField>(claim: &LegacyClaim<F>, shape: &Shape) -> Result<()> {
    if shape.dims().contains(&0) {
        return Err(ProverError::InvalidTensorShape(shape.0.clone()));
    }
    if claim.logical_shape != *shape {
        return Err(ProverError::ShapeMismatch {
            name: "attention output claim",
            expected: shape.0.clone(),
            actual: claim.logical_shape.0.clone(),
        });
    }
    let expected_domain = shape.padded_power_of_two();
    if claim.domain_shape != expected_domain {
        return Err(ProverError::ShapeMismatch {
            name: "attention output claim domain",
            expected: expected_domain.0,
            actual: claim.domain_shape.0.clone(),
        });
    }
    let expected = claim.domain_shape.point_len();
    if claim.point.len() != expected {
        return Err(ProverError::ShapeMismatch {
            name: "attention output claim point",
            expected: vec![expected],
            actual: vec![claim.point.len()],
        });
    }
    Ok(())
}

pub(crate) fn verify_claim<F: JoltField>(
    claim: &LegacyClaim<F>,
    shape: &Shape,
) -> std::result::Result<(), ProofVerifyError> {
    if claim.logical_shape != *shape {
        return Err(ProofVerifyError::InvalidInputLength(
            shape.numel(),
            claim.logical_shape.numel(),
        ));
    }
    let expected_domain = shape.padded_power_of_two();
    if claim.domain_shape != expected_domain {
        return Err(ProofVerifyError::InvalidInputLength(
            expected_domain.numel(),
            claim.domain_shape.numel(),
        ));
    }
    let expected = claim.domain_shape.point_len();
    if claim.point.len() != expected {
        return Err(ProofVerifyError::InvalidInputLength(
            expected,
            claim.point.len(),
        ));
    }
    Ok(())
}

pub(crate) fn ensure_len(name: &'static str, shape: &Shape, actual: usize) -> Result<()> {
    let expected = shape.numel();
    if actual == expected {
        Ok(())
    } else {
        Err(ProverError::TensorLenMismatch {
            name,
            shape: shape.0.clone(),
            expected,
            actual,
        })
    }
}

pub(crate) fn split3<F>(point: &[F], dim0: usize, dim1: usize, _dim2: usize) -> (&[F], &[F], &[F]) {
    let v0 = log2_ceil(dim0);
    let v1 = log2_ceil(dim1);
    (&point[..v0], &point[v0..v0 + v1], &point[v0 + v1..])
}

pub(crate) fn drop_gqa_lsb_for_dims<F: JoltField>(
    r_h: &[F],
    q_heads: usize,
    kv_heads: usize,
) -> Result<&[F]> {
    validate_gqa(q_heads, kv_heads)?;
    let q_vars = log2_ceil(q_heads);
    let kv_vars = log2_ceil(kv_heads);
    if r_h.len() != q_vars || kv_vars + 1 != q_vars {
        return Err(ProverError::InvalidGqa { q_heads, kv_heads });
    }
    Ok(&r_h[..kv_vars])
}

pub(crate) fn drop_gqa_lsb_for_dims_verify<F: JoltField>(
    r_h: &[F],
    q_heads: usize,
    kv_heads: usize,
) -> std::result::Result<&[F], ProofVerifyError> {
    let q_vars = log2_ceil(q_heads);
    let kv_vars = log2_ceil(kv_heads);
    if q_heads / kv_heads != QWEN3_GQA_GROUP_SIZE || r_h.len() != q_vars || kv_vars + 1 != q_vars {
        return Err(ProofVerifyError::InvalidInputLength(q_vars, r_h.len()));
    }
    Ok(&r_h[..kv_vars])
}

pub(crate) fn log2_ceil(value: usize) -> usize {
    value.next_power_of_two().trailing_zeros() as usize
}
