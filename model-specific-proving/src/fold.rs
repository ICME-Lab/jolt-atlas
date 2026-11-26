use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rayon::prelude::*;

pub(crate) enum Column<'a, F> {
    Source(usize, Box<dyn Fn(usize) -> F + Sync + 'a>),
    Table {
        values: Vec<F>,
        len: usize,
        shift: u32,
    },
}

impl<'a, F: Field> Column<'a, F> {
    pub(crate) fn source(len: usize, read: impl Fn(usize) -> F + Sync + 'a) -> Self {
        Self::Source(len, Box::new(read))
    }

    pub(crate) fn table(values: Vec<F>, len: usize, repeat: usize) -> Self {
        Self::Table {
            values,
            len,
            shift: repeat.ilog2(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Source(len, _) | Self::Table { len, .. } => *len,
        }
    }

    fn at(&self, index: usize) -> F {
        match self {
            Self::Source(_, read) => read(index),
            Self::Table { values, shift, .. } => values[(index >> shift) & (values.len() - 1)],
        }
    }

    fn bind(&mut self, point: F) {
        let half = self.len() / 2;
        if let Self::Table { values, len, shift } = self {
            *len = half;
            if half >= values.len() << *shift {
                return;
            }
            if values.len() == 1 {
                *shift = half.ilog2();
                return;
            }
            let split = values.len() / 2;
            let (lower, upper) = values.split_at_mut(split);
            let bind = |(a, b): (&mut F, &F)| *a += point * (*b - *a);
            if split >= 4096 {
                lower.par_iter_mut().zip(upper.par_iter()).for_each(bind);
            } else {
                lower.iter_mut().zip(upper.iter()).for_each(bind);
            }
            values.truncate(split);
            return;
        }

        let value = |i| {
            let a = self.at(i);
            a + point * (self.at(i + half) - a)
        };
        let values = if half >= 4096 {
            (0..half)
                .into_par_iter()
                .with_min_len(1024)
                .map(value)
                .collect()
        } else {
            (0..half).map(value).collect()
        };
        *self = Self::table(values, half, 1);
    }
}

pub(crate) struct Fold<'a, F, G> {
    pub(crate) columns: Vec<Column<'a, F>>,
    pub(crate) degree: usize,
    pub(crate) relation: G,
}

impl<F: Field, G> Fold<'_, F, G> {
    pub(crate) fn terminal(&self) -> Vec<F> {
        self.columns.iter().map(|column| column.at(0)).collect()
    }
}

impl<F: Field, G: Fn(&[F]) -> F + Sync> ProveRounds<F> for Fold<'_, F, G> {
    fn num_rounds(&self) -> usize {
        self.columns[0].len().ilog2() as usize
    }

    fn prove_round(
        &mut self,
        previous: Option<F>,
        _: usize,
        sum: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(point) = previous {
            self.finish_rounds(point)?;
        }
        let half = self.columns[0].len() / 2;
        let scan = |start: usize, end: usize| {
            let mut sums = vec![F::zero(); self.degree + 1];
            let mut values = vec![F::zero(); self.columns.len()];
            let mut delta = values.clone();
            for i in start..end {
                for (j, column) in self.columns.iter().enumerate() {
                    let lo = column.at(i);
                    values[j] = lo;
                    delta[j] = column.at(i + half) - lo;
                }
                sums[0] += (self.relation)(&values);
                for (a, &d) in values.iter_mut().zip(&delta) {
                    *a += d;
                }
                for total in &mut sums[2..] {
                    for (a, &d) in values.iter_mut().zip(&delta) {
                        *a += d;
                    }
                    *total += (self.relation)(&values);
                }
            }
            sums
        };
        let mut values = if half >= 4096 {
            (0..half.div_ceil(1024))
                .into_par_iter()
                .map(|block| scan(block * 1024, ((block + 1) * 1024).min(half)))
                .reduce_with(|mut a, b| {
                    for (a, b) in a.iter_mut().zip(b) {
                        *a += b;
                    }
                    a
                })
                .expect("at least one Sumcheck pair")
        } else {
            scan(0, half)
        };
        values[1] = sum - values[0];
        Ok(UnivariatePoly::from_evals(&values))
    }

    fn finish_rounds(&mut self, point: F) -> Result<(), SumcheckError<F>> {
        for column in &mut self.columns {
            column.bind(point);
        }
        Ok(())
    }
}

pub(crate) struct InnerProducts<F> {
    pub(crate) pairs: Vec<(Vec<F>, Vec<F>)>,
    pub(crate) rounds: usize,
}

impl<F: Field> ProveRounds<F> for InnerProducts<F> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        prev: Option<F>,
        _: usize,
        sum: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = prev {
            self.finish_rounds(r)?;
        }
        let mut zero = F::zero();
        let mut two = F::zero();
        for (a, b) in &self.pairs {
            let mid = a.len() / 2;
            for i in 0..mid {
                zero += a[i] * b[i];
                two += (a[i + mid] + a[i + mid] - a[i]) * (b[i + mid] + b[i + mid] - b[i]);
            }
        }
        Ok(UnivariatePoly::from_evals(&[zero, sum - zero, two]))
    }

    fn finish_rounds(&mut self, point: F) -> Result<(), SumcheckError<F>> {
        for (a, b) in &mut self.pairs {
            for values in [a, b] {
                let mid = values.len() / 2;
                for i in 0..mid {
                    let low = values[i];
                    values[i] = low + point * (values[i + mid] - low);
                }
                values.truncate(mid);
            }
        }
        Ok(())
    }
}

pub(crate) fn agreement<F: Field>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x * y + (F::one() - x) * (F::one() - y))
        .product()
}

pub(crate) fn basis<F: Field>(point: &[F]) -> Vec<F> {
    let mut weights = vec![F::one()];
    for &r in point.iter().rev() {
        let count = weights.len();
        for i in 0..count {
            weights.push(weights[i] * r);
            weights[i] *= F::one() - r;
        }
    }
    weights
}
