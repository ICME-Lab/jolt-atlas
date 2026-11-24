use jolt_field::JoltField;
use jolt_poly::EqPolynomial;

use super::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub(crate) rows: usize,
    pub(crate) width: usize,
}

impl Shape {
    pub fn new(rows: usize, width: usize) -> Result<Self> {
        let padded = rows.checked_next_power_of_two().and_then(|r| {
            width
                .checked_next_power_of_two()
                .and_then(|w| r.checked_mul(w))
        });
        if rows == 0 || width == 0 || padded.is_none_or(|n| n > usize::MAX / 128) {
            return Err("invalid tensor shape");
        }
        Ok(Self { rows, width })
    }

    pub(crate) fn row_vars(self) -> usize {
        self.rows.next_power_of_two().ilog2() as usize
    }
    pub(crate) fn vars(self) -> usize {
        self.row_vars() + self.width.next_power_of_two().ilog2() as usize
    }

    pub(crate) fn check<T>(self, values: &[T]) -> Result<()> {
        if values.len() != self.rows * self.width {
            return Err("invalid public tensor shape");
        }
        Ok(())
    }

    pub(crate) fn padded<T: Copy, U: Copy>(
        self,
        values: &[T],
        zero: U,
        convert: impl Fn(T) -> U,
    ) -> Vec<U> {
        let mut out = vec![zero; 1 << self.vars()];
        for (dst, src) in out
            .chunks_exact_mut(self.width.next_power_of_two())
            .zip(values.chunks_exact(self.width))
        {
            for (dst, &src) in dst.iter_mut().zip(src) {
                *dst = convert(src);
            }
        }
        out
    }

    pub(crate) fn eval<F: JoltField, T: Copy + Into<i64>>(self, values: &[T], point: &[F]) -> F {
        let (row, col) = point.split_at(self.row_vars());
        dot(bind_rows(self.width, values, row), weights(col))
    }
}

pub(crate) fn weights<F: JoltField>(point: &[F]) -> Vec<F> {
    EqPolynomial::<F>::evals(point, None)
}

pub(crate) fn eq<F: JoltField>(left: &[F], right: &[F]) -> F {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(&a, &b)| a * b + (F::one() - a) * (F::one() - b))
        .product()
}

pub(crate) fn dot<F: JoltField>(
    a: impl IntoIterator<Item = F>,
    b: impl IntoIterator<Item = F>,
) -> F {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}

pub(crate) fn bind_rows<F: JoltField, T: Copy + Into<i64>>(
    width: usize,
    values: &[T],
    point: &[F],
) -> Vec<F> {
    let mut out = vec![F::zero(); width.next_power_of_two()];
    for (row, weight) in values.chunks_exact(width).zip(weights(point)) {
        for (dst, &src) in out.iter_mut().zip(row) {
            *dst += weight * F::from_i64(src.into());
        }
    }
    out
}

pub(crate) fn integer_dot(a: &[i32], b: &[i32]) -> i128 {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| i128::from(a) * i128::from(b))
        .sum()
}
