use crate::{field::F, fold::basis};
use jolt_field::{Ring, Zero};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Shape {
    pub rows: usize,
    pub cols: usize,
}

impl Shape {
    pub fn new(rows: usize, cols: usize) -> Result<Self, &'static str> {
        if rows == 0 || cols == 0 || rows > 1 << 20 || cols > 1 << 20 {
            return Err("invalid tensor dimensions");
        }
        Ok(Self { rows, cols })
    }

    pub fn row_bits(self) -> usize {
        self.rows.next_power_of_two().ilog2() as usize
    }

    pub fn bits(self) -> usize {
        self.row_bits() + self.stride().ilog2() as usize
    }

    pub fn stride(self) -> usize {
        self.cols.next_power_of_two()
    }

    pub fn padded(self) -> usize {
        1 << self.bits()
    }

    pub fn index(self, index: usize) -> Option<usize> {
        let (r, c) = (index / self.stride(), index % self.stride());
        (r < self.rows && c < self.cols).then_some(r * self.cols + c)
    }

    pub fn at(self, values: &[i32], index: usize) -> F {
        self.index(index)
            .map_or_else(F::zero, |i| F::from_i32(values[i]))
    }

    pub fn active(self, point: &[F]) -> F {
        let (row, col) = point.split_at(self.row_bits());
        basis(row).into_iter().take(self.rows).sum::<F>()
            * basis(col).into_iter().take(self.cols).sum::<F>()
    }

    pub fn eval<T: Copy + Into<i128>>(self, values: &[T], point: &[F]) -> F {
        self.eval_with(point, |r, c| values[r * self.cols + c].into())
    }

    pub fn eval_with(self, point: &[F], at: impl Fn(usize, usize) -> i128) -> F {
        let (row, col) = point.split_at(self.row_bits());
        let col = basis(col);
        basis(row)
            .into_iter()
            .take(self.rows)
            .enumerate()
            .map(|(i, r)| {
                r * col
                    .iter()
                    .take(self.cols)
                    .enumerate()
                    .map(|(j, &c)| c.mul_integer(at(i, j)))
                    .sum::<F>()
            })
            .sum()
    }
}

pub(crate) fn add_rows(out: &mut [F], matrix: &[i32], width: usize, weights: &[F]) {
    let add = |out: &mut [F], matrix: &[i32], weights: &[F]| {
        for (row, &w) in matrix.chunks_exact(width).zip(weights) {
            for (out, &x) in out.iter_mut().zip(row) {
                *out += w.mul_integer(i128::from(x));
            }
        }
    };
    let rows = weights.len().min(matrix.len() / width);
    let chunk = (32_768 / width).max(16);
    if rows <= chunk {
        return add(out, matrix, weights);
    }
    let sum = matrix[..rows * width]
        .par_chunks(chunk * width)
        .zip(weights[..rows].par_chunks(chunk))
        .fold(
            || vec![F::zero(); out.len()],
            |mut sum, (matrix, weights)| {
                add(&mut sum, matrix, weights);
                sum
            },
        )
        .reduce_with(|mut left, right| {
            left.iter_mut().zip(right).for_each(|(a, b)| *a += b);
            left
        })
        .unwrap();
    out.iter_mut().zip(sum).for_each(|(a, b)| *a += b);
}

pub(crate) fn eval_rows(matrix: &[i32], width: usize, point: &[F], rows: usize) -> Vec<F> {
    let weights = basis(point);
    let mut values: Vec<_> = matrix
        .chunks_exact(width)
        .map(|r| {
            r.iter()
                .zip(&weights)
                .map(|(&x, &w)| w.mul_integer(i128::from(x)))
                .sum()
        })
        .collect();
    values.resize(rows.next_power_of_two(), F::zero());
    values
}

pub(crate) fn dot(a: &[i32], b: &[i32], narrow: bool) -> i128 {
    if !narrow {
        return a
            .iter()
            .zip(b)
            .map(|(&x, &y)| i128::from(x) * i128::from(y))
            .sum();
    }
    let mut stripes = [0i64; 4];
    let mut left = a.chunks_exact(4);
    let mut right = b.chunks_exact(4);
    for (a, b) in left.by_ref().zip(right.by_ref()) {
        for (i, out) in stripes.iter_mut().enumerate() {
            *out += i64::from(a[i]) * i64::from(b[i]);
        }
    }
    let tail: i64 = left
        .remainder()
        .iter()
        .zip(right.remainder())
        .map(|(&x, &y)| i64::from(x) * i64::from(y))
        .sum();
    i128::from(stripes.into_iter().sum::<i64>() + tail)
}

// The caller bounds the absolute sum of each dot product by i64::MAX.
pub(crate) fn dot_pair(a: &[i32], b: &[i32], weights: &[i32]) -> [i128; 2] {
    let mut a = a.chunks_exact(4);
    let mut b = b.chunks_exact(4);
    let mut weights = weights.chunks_exact(4);
    let mut sums = [[0i64; 4]; 2];
    for ((a, b), w) in a.by_ref().zip(b.by_ref()).zip(weights.by_ref()) {
        for i in 0..4 {
            let w = i64::from(w[i]);
            sums[0][i] += i64::from(a[i]) * w;
            sums[1][i] += i64::from(b[i]) * w;
        }
    }
    let tails = [a.remainder(), b.remainder()];
    std::array::from_fn(|i| {
        let tail: i64 = tails[i]
            .iter()
            .zip(weights.remainder())
            .map(|(&x, &w)| i64::from(x) * i64::from(w))
            .sum();
        i128::from(sums[i].into_iter().sum::<i64>() + tail)
    })
}
