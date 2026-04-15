//! Curve traits for cryptographic operations.
//!
//! This module defines the `JoltCurve` trait which abstracts over pairing-friendly
//! elliptic curves used for polynomial commitments and zero-knowledge proofs.

use crate::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

/// A group element suitable for cryptographic operations.
///
/// Abstracts over elliptic curve group operations needed for
/// Pedersen commitments, polynomial commitments, and other cryptographic primitives.
pub trait JoltGroupElement:
    Clone
    + Copy
    + Debug
    + Default
    + Eq
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + AddAssign
    + SubAssign
    + CanonicalSerialize
    + CanonicalDeserialize
{
    type Scalar: JoltField;

    fn zero() -> Self;

    fn is_zero(&self) -> bool;

    fn double(&self) -> Self;

    fn scalar_mul(&self, scalar: &Self::Scalar) -> Self;
}

/// A pairing-friendly curve suitable for PCS and ZK operations.
pub trait JoltCurve: Clone + Sync + Send + 'static {
    type F: JoltField;
    type G1: JoltGroupElement<Scalar = Self::F>;
    type G2: JoltGroupElement<Scalar = Self::F>;
    type G1Affine: Clone + Copy + Debug + Send + Sync + 'static;
    type GT: Clone
        + Debug
        + Default
        + Eq
        + Send
        + Sync
        + 'static
        + Add<Output = Self::GT>
        + for<'a> Add<&'a Self::GT, Output = Self::GT>
        + AddAssign
        + CanonicalSerialize
        + CanonicalDeserialize;

    fn g1_generator() -> Self::G1;
    fn g2_generator() -> Self::G2;
    fn g1_to_affine(point: &Self::G1) -> Self::G1Affine;
    fn pairing(g1: &Self::G1, g2: &Self::G2) -> Self::GT;
    fn multi_pairing(g1s: &[Self::G1], g2s: &[Self::G2]) -> Self::GT;
    fn g1_msm(bases: &[Self::G1], scalars: &[Self::F]) -> Self::G1;
    fn g1_affine_msm(bases: &[Self::G1Affine], scalars: &[Self::F]) -> Self::G1;
    fn g2_msm(bases: &[Self::G2], scalars: &[Self::F]) -> Self::G2;
    fn random_g1<R: rand_core::RngCore>(rng: &mut R) -> Self::G1;
}

// --- BN254 implementation ---

use ark_bn254::{Bn254, Fq12, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{pairing::Pairing, AdditiveGroup, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{One, Zero};
use ark_std::UniformRand;
use std::ops::MulAssign;

macro_rules! impl_group_ops {
    ($Name:ident, $Inner:ty, $Field:ty) => {
        impl Add for $Name {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                $Name(self.0 + rhs.0)
            }
        }
        impl<'a> Add<&'a $Name> for $Name {
            type Output = Self;
            fn add(self, rhs: &'a $Name) -> Self {
                $Name(self.0 + rhs.0)
            }
        }
        impl Sub for $Name {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                $Name(self.0 - rhs.0)
            }
        }
        impl<'a> Sub<&'a $Name> for $Name {
            type Output = Self;
            fn sub(self, rhs: &'a $Name) -> Self {
                $Name(self.0 - rhs.0)
            }
        }
        impl Neg for $Name {
            type Output = Self;
            fn neg(self) -> Self {
                $Name(-self.0)
            }
        }
        impl AddAssign for $Name {
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0;
            }
        }
        impl SubAssign for $Name {
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0;
            }
        }
        impl Mul<$Field> for $Name {
            type Output = Self;
            fn mul(mut self, rhs: $Field) -> Self {
                self.0.mul_assign(rhs);
                self
            }
        }
    };
}

macro_rules! impl_group_element {
    ($Name:ident, $Proj:ty, $Field:ty) => {
        impl JoltGroupElement for $Name {
            type Scalar = $Field;

            fn zero() -> Self {
                $Name(<$Proj>::zero())
            }
            fn is_zero(&self) -> bool {
                self.0.is_zero()
            }
            fn double(&self) -> Self {
                $Name(AdditiveGroup::double(&self.0))
            }
            fn scalar_mul(&self, scalar: &$Field) -> Self {
                let mut result = self.0;
                result.mul_assign(*scalar);
                $Name(result)
            }
        }
    };
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Bn254G1(pub G1Projective);
impl_group_ops!(Bn254G1, G1Projective, Fr);
impl_group_element!(Bn254G1, G1Projective, Fr);

impl From<G1Projective> for Bn254G1 {
    fn from(value: G1Projective) -> Self {
        Bn254G1(value)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Bn254G2(pub G2Projective);
impl_group_ops!(Bn254G2, G2Projective, Fr);
impl_group_element!(Bn254G2, G2Projective, Fr);

#[derive(Clone, Copy, Debug, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Bn254GT(pub Fq12);

impl Default for Bn254GT {
    fn default() -> Self {
        // `Add` models the GT group law, which is multiplicative in Fq12.
        Self(Fq12::one())
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Add for Bn254GT {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Bn254GT(self.0 * rhs.0)
    }
}
#[allow(clippy::suspicious_arithmetic_impl)]
impl<'a> Add<&'a Bn254GT> for Bn254GT {
    type Output = Self;
    fn add(self, rhs: &'a Bn254GT) -> Self {
        Bn254GT(self.0 * rhs.0)
    }
}
#[allow(clippy::suspicious_op_assign_impl)]
impl AddAssign for Bn254GT {
    fn add_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

#[derive(Clone, Debug, Default)]
pub struct Bn254Curve;

impl JoltCurve for Bn254Curve {
    type F = Fr;
    type G1 = Bn254G1;
    type G2 = Bn254G2;
    type G1Affine = G1Affine;
    type GT = Bn254GT;

    fn g1_generator() -> Self::G1 {
        Bn254G1(G1Affine::generator().into())
    }

    fn g2_generator() -> Self::G2 {
        Bn254G2(G2Affine::generator().into())
    }

    #[inline]
    fn g1_to_affine(point: &Self::G1) -> G1Affine {
        point.0.into_affine()
    }

    fn pairing(g1: &Self::G1, g2: &Self::G2) -> Self::GT {
        Bn254GT(Bn254::pairing(g1.0, g2.0).0)
    }

    fn multi_pairing(g1s: &[Self::G1], g2s: &[Self::G2]) -> Self::GT {
        debug_assert_eq!(g1s.len(), g2s.len());
        let g1_affines: Vec<G1Affine> = g1s.iter().map(|g| g.0.into_affine()).collect();
        let g2_affines: Vec<G2Affine> = g2s.iter().map(|g| g.0.into_affine()).collect();
        Bn254GT(Bn254::multi_pairing(&g1_affines, &g2_affines).0)
    }

    fn g1_msm(bases: &[Self::G1], scalars: &[Fr]) -> Self::G1 {
        debug_assert_eq!(bases.len(), scalars.len());
        let affine_bases: Vec<G1Affine> = bases.iter().map(|b| b.0.into_affine()).collect();
        Self::g1_affine_msm(&affine_bases, scalars)
    }

    #[inline]
    fn g1_affine_msm(bases: &[G1Affine], scalars: &[Fr]) -> Self::G1 {
        debug_assert_eq!(bases.len(), scalars.len());
        Bn254G1(VariableBaseMSM::msm(bases, scalars).expect("msm length mismatch"))
    }

    fn g2_msm(bases: &[Self::G2], scalars: &[Fr]) -> Self::G2 {
        debug_assert_eq!(bases.len(), scalars.len());
        let affine_bases: Vec<G2Affine> = bases.iter().map(|b| b.0.into_affine()).collect();
        Bn254G2(VariableBaseMSM::msm(&affine_bases, scalars).expect("msm length mismatch"))
    }

    fn random_g1<R: rand_core::RngCore>(rng: &mut R) -> Self::G1 {
        Bn254G1(G1Projective::rand(rng))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::UniformRand;
    use rand::thread_rng;

    #[test]
    fn test_g1_operations() {
        let g = Bn254Curve::g1_generator();
        let zero = Bn254G1::zero();
        assert!(zero.is_zero());
        assert!(!g.is_zero());
        assert_eq!(g + zero, g);
        assert_eq!(g - g, zero);
    }

    #[test]
    fn test_g1_msm() {
        let g = Bn254Curve::g1_generator();
        let scalars = vec![Fr::from(2u64), Fr::from(3u64)];
        let bases = vec![g, g];
        let result = Bn254Curve::g1_msm(&bases, &scalars);
        let expected = g.scalar_mul(&Fr::from(5u64));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pairing_bilinearity() {
        let mut rng = thread_rng();
        let a = Fr::rand(&mut rng);
        let b = Fr::rand(&mut rng);
        let g1 = Bn254Curve::g1_generator();
        let g2 = Bn254Curve::g2_generator();
        let pairing1 = Bn254Curve::pairing(&g1.scalar_mul(&a), &g2.scalar_mul(&b));
        let pairing2 = Bn254Curve::pairing(&g1.scalar_mul(&(a * b)), &g2);
        assert_eq!(pairing1, pairing2);
    }
}
