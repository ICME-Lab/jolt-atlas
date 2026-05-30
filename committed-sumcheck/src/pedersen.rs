//! Scalar Pedersen commitments over BN254 G1.
//!
//! The only commitment form in this crate is
//!
//! ```text
//! C = value * G + blinding * H
//! ```
//!
//! Round-polynomial coefficients should be committed independently with this
//! scalar form. There is intentionally no vector commitment abstraction here.

use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::AffineRepr;
use ark_std::UniformRand;
use rand_core::CryptoRngCore;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PedersenParams {
    pub value_generator: G1Projective,
    pub blinding_generator: G1Projective,
}

impl PedersenParams {
    pub fn new(value_generator: G1Projective, blinding_generator: G1Projective) -> Self {
        Self {
            value_generator,
            blinding_generator,
        }
    }

    /// Test/dev helper. Production code should use domain-separated generator
    /// derivation so nobody knows a discrete-log relation between `G` and `H`.
    pub fn random<R: CryptoRngCore>(rng: &mut R) -> Self {
        Self {
            value_generator: G1Projective::rand(rng),
            blinding_generator: G1Projective::rand(rng),
        }
    }

    pub fn generator_g1() -> G1Projective {
        G1Affine::generator().into_group()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Commitment(pub G1Projective);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Opening {
    pub value: Fr,
    pub blinding: Fr,
}

impl Opening {
    pub fn random<R: CryptoRngCore>(value: Fr, rng: &mut R) -> Self {
        Self {
            value,
            blinding: Fr::rand(rng),
        }
    }
}

pub fn commit(params: &PedersenParams, opening: &Opening) -> Commitment {
    Commitment(
        params.value_generator * opening.value + params.blinding_generator * opening.blinding,
    )
}

pub fn commit_parts(params: &PedersenParams, value: Fr, blinding: Fr) -> Commitment {
    commit(params, &Opening { value, blinding })
}

pub fn commitment_without_value(
    params: &PedersenParams,
    commitment: &Commitment,
    value: Fr,
) -> G1Projective {
    commitment.0 - params.value_generator * value
}
