//! The univariate polynomial types the external `akita-*` crates bind as `jolt_poly::*`,
//! re-exported from [`atlas_algebra::poly::univariate`]. Nothing in the atlas crates names this
//! package; it exists while `akita-*` is an external dependency.

pub use atlas_algebra::poly::univariate::{
    CompressedPoly, OmittedConstantPoly, UnivariatePoly, UnivariatePolynomial,
};
