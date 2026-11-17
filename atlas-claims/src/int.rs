//! Integer coefficients: expressions a relation states once, for every field.

use std::fmt;
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use atlas_algebra::field::{AdditiveGroup, One, Ring, Zero};

use crate::{Expr, Term};

/// An integer coefficient of a symbolic expression.
///
/// A relation whose coefficients are integers (signs, powers of two, divisors) builds its
/// expressions over `Int` and a field reads them through [`Expr::cast`]. Arithmetic is modulo
/// `2^128`; a relation's coefficients are products of a few model constants, far below that.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Int(pub i128);

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Zero for Int {
    fn zero() -> Self {
        Self(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for Int {
    fn one() -> Self {
        Self(1)
    }
}

impl Add for Int {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl Add<&Int> for Int {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self {
        self + *rhs
    }
}

impl AddAssign for Int {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Int {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

impl Sub<&Int> for Int {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self {
        self - *rhs
    }
}

impl SubAssign for Int {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for Int {
    type Output = Self;

    fn neg(self) -> Self {
        Self(self.0.wrapping_neg())
    }
}

impl Mul for Int {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self(self.0.wrapping_mul(rhs.0))
    }
}

impl Mul<&Int> for Int {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self {
        self * *rhs
    }
}

impl MulAssign for Int {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Sum for Int {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self(0), Add::add)
    }
}

impl<'a> Sum<&'a Int> for Int {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self(0), |acc, x| acc + *x)
    }
}

impl Product for Int {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self(1), Mul::mul)
    }
}

impl<'a> Product<&'a Int> for Int {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self(1), |acc, x| acc * *x)
    }
}

impl AdditiveGroup for Int {}

impl Ring for Int {
    fn from_u64(v: u64) -> Self {
        Self(i128::from(v))
    }

    fn from_i64(v: i64) -> Self {
        Self(i128::from(v))
    }

    fn from_u128(v: u128) -> Self {
        Self(v as i128)
    }

    fn from_i128(v: i128) -> Self {
        Self(v)
    }
}

impl<O: Clone, P: Clone, C: Clone> Expr<Int, O, P, C> {
    /// The expression over the ring `F`: every coefficient through the integer embedding.
    pub fn cast<F: Ring>(&self) -> Expr<F, O, P, C> {
        Expr {
            terms: self
                .terms
                .iter()
                .map(|term| Term {
                    coefficient: F::from_i128(term.coefficient.0),
                    factors: term.factors.clone(),
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{constant, opening};
    use atlas_algebra::field::Prime128Offset275 as Fr;

    #[test]
    fn cast_is_the_integer_embedding() {
        let over_int: Expr<Int, u8> =
            constant(Int(-3)) * opening(7u8) + opening(1u8) * opening(1u8) - constant(Int(1 << 20));
        let over_field: Expr<Fr, u8> = constant(Fr::from_i64(-3)) * opening(7u8)
            + opening(1u8) * opening(1u8)
            - constant(Fr::from_u64(1 << 20));
        assert_eq!(over_int.cast::<Fr>(), over_field);
    }
}
