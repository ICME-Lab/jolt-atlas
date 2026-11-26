use jolt_field::{Field, One, Ring, Zero};
use jolt_poly::EqPolynomial;

use super::{field::F, Result};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub rows: usize,
    pub width: usize,
}

impl Shape {
    pub fn new(rows: usize, width: usize) -> Result<Self> {
        let capacity = rows.checked_next_power_of_two().and_then(|r| {
            width
                .checked_next_power_of_two()
                .and_then(|w| r.checked_mul(w))
        });
        if rows == 0 || width == 0 || capacity.is_none_or(|n| n > usize::MAX / 128) {
            return Err("invalid tensor dimensions");
        }
        Ok(Self { rows, width })
    }

    pub(crate) fn row_bits(self) -> usize {
        self.rows.next_power_of_two().ilog2() as usize
    }

    pub(crate) fn bits(self) -> usize {
        self.row_bits() + self.width.next_power_of_two().ilog2() as usize
    }

    pub(crate) fn len(self) -> usize {
        self.rows * self.width
    }

    pub(crate) fn at<T: Copy + Into<i128>>(self, values: &[T], index: usize) -> F {
        let stride = self.width.next_power_of_two();
        let (row, col) = (index / stride, index % stride);
        if row < self.rows && col < self.width {
            F::from_i128(values[row * self.width + col].into())
        } else {
            F::zero()
        }
    }

    pub(crate) fn evaluate<T: Copy + Into<i128>>(self, values: &[T], point: &[F]) -> F {
        let (row, col) = point.split_at(self.row_bits());
        let col = weights(col);
        inner(
            weights(row),
            values.chunks_exact(self.width).map(|v| {
                inner(
                    col.iter().copied(),
                    v.iter().map(|&x| F::from_i128(x.into())),
                )
            }),
        )
    }
}

#[derive(Clone)]
pub(crate) struct Claim {
    pub point: Vec<F>,
    pub value: F,
}

impl Claim {
    pub(crate) fn new(point: Vec<F>, value: F) -> Self {
        Self { point, value }
    }
}

pub(crate) fn weights(point: &[F]) -> Vec<F> {
    EqPolynomial::evals(point, None)
}

pub(crate) fn equality(left: &[F], right: &[F]) -> F {
    left.iter()
        .zip(right)
        .map(|(&x, &y)| x * y + (F::one() - x) * (F::one() - y))
        .product()
}

pub(crate) fn inner(a: impl IntoIterator<Item = F>, b: impl IntoIterator<Item = F>) -> F {
    a.into_iter()
        .zip(b)
        .fold(F::zero(), |sum, (x, y)| x.mul_add(y, sum))
}

pub(crate) fn bind_rows(width: usize, rows: &[i32], point: &[F]) -> Vec<F> {
    let mut bound = vec![F::zero(); width.next_power_of_two()];
    for (row, factor) in rows.chunks_exact(width).zip(weights(point)) {
        for (total, &value) in bound.iter_mut().zip(row) {
            *total = F::from_i32(value).mul_add(factor, *total);
        }
    }
    bound
}
