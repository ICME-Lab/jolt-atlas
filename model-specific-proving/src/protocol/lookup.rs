//! Read fixed public tables using sparse chunk indices.

use rayon::prelude::*;
use std::{collections::BTreeMap, sync::Arc};

use jolt_poly::UnivariatePoly;
use jolt_sumcheck::prove_single;
use jolt_sumcheck::{ProveRounds, SumcheckError};

use super::{
    pcs::{Advice, Kind, Layout, Opening, Part, Slot},
    session::{
        bind_halves, instance, Claim, Column, ProofField, Session, PARALLEL_ROUND_SIZE,
        ROUND_CHUNK_SIZE,
    },
    tensor::{dot, eq, weights},
    Result,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Table<'a> {
    Slice(&'a [i32]),
    Prepend(i32, &'a [i32]),
}

impl Table<'_> {
    fn len(self) -> usize {
        match self {
            Self::Slice(values) => values.len(),
            Self::Prepend(_, tail) => 1 + tail.len(),
        }
    }

    pub(crate) fn at(self, index: usize) -> i32 {
        match self {
            Self::Slice(values) => values[index],
            Self::Prepend(first, _) if index == 0 => first,
            Self::Prepend(_, tail) => tail[index - 1],
        }
    }

    fn evaluate<F: ProofField>(self, rows: &[F], cols: &[F]) -> F {
        let evaluate_row =
            |row: &[i32]| dot(row.iter().copied().map(F::from_i32), cols.iter().copied());
        match self {
            Self::Slice(values) => dot(
                values.chunks_exact(cols.len()).map(evaluate_row),
                rows.iter().copied(),
            ),
            Self::Prepend(first, tail) => {
                let (head, rest) = tail.split_at(cols.len() - 1);
                let head = F::from_i32(first) * cols[0]
                    + dot(
                        head.iter().copied().map(F::from_i32),
                        cols[1..].iter().copied(),
                    );
                head * rows[0]
                    + dot(
                        rest.chunks_exact(cols.len()).map(evaluate_row),
                        rows[1..].iter().copied(),
                    )
            }
        }
    }
}

pub(crate) struct Lookup {
    cols: Vec<Slot>,
    address_vars: usize,
}

pub(crate) struct Read<'w, F> {
    pub(crate) cell: Vec<F>,
    pub(crate) address: F,
    pub(crate) output: Box<[F]>,
    pub(crate) witness: Option<&'w [u32]>,
}

impl<'w, F> Read<'w, F> {
    fn witness(&self) -> &'w [u32] {
        self.witness.expect("prover lookup witness")
    }
}

struct Batch<'a, 'w, F> {
    table: Box<[Table<'a>]>,
    reads: Vec<(&'a Lookup, Read<'w, F>)>,
}

#[derive(Default)]
pub(crate) struct Batches<'a, 'w, F>(Vec<Batch<'a, 'w, F>>);

impl<'a, 'w, F: ProofField> Batches<'a, 'w, F> {
    pub(crate) fn push(&mut self, table: &[Table<'a>], lookup: &'a Lookup, read: Read<'w, F>) {
        if let Some(batch) = self.0.iter_mut().find(|b| {
            let (first, q) = &b.reads[0];
            q.cell.len() == read.cell.len() && lookup.parts_fit(first) && b.table.as_ref() == table
        }) {
            batch.reads.push((lookup, read));
        } else {
            self.0.push(Batch {
                table: table.into(),
                reads: vec![(lookup, read)],
            });
        }
    }

    pub(crate) fn finish(mut self, io: &mut Session<'_, '_, F>) -> Result<()> {
        // Softmax groups precede activation groups in the transcript.
        self.0.sort_by_key(|b| std::cmp::Reverse(b.table.len()));
        for b in self.0 {
            Lookup::batch(&b.reads, &b.table, io)?;
        }
        Ok(())
    }
}

impl Lookup {
    pub(crate) fn new(layout: &mut Layout, cells: usize, size: usize) -> Result<Self> {
        const DIGIT_BITS: usize = 4;
        if size < 2 || !size.is_power_of_two() || size.ilog2() > u32::BITS {
            return Err("invalid lookup table size");
        }
        let address_vars = size.ilog2() as usize;
        let first = (address_vars - 1) % DIGIT_BITS + 1;
        let sizes = std::iter::once(first).chain(std::iter::repeat_n(
            DIGIT_BITS,
            (address_vars - first) / DIGIT_BITS,
        ));
        let cols = sizes
            .map(|address_vars| layout.add(cells, Kind::Lookup { address_vars }))
            .collect::<Result<_>>()?;
        Ok(Self { cols, address_vars })
    }

    fn parts<'a, F>(&self, point: &'a [F]) -> Vec<&'a [F]> {
        let mut rest = point;
        self.cols
            .iter()
            .map(|slot| {
                let (part, tail) = rest.split_at(slot.vars() - slot.cell_vars);
                rest = tail;
                part
            })
            .collect()
    }

    pub(crate) fn record(&self, advice: &mut Advice, indices: Arc<Vec<u32>>) -> Result<()> {
        let mut shift = self.address_vars;
        for &slot in &self.cols {
            let n = slot.vars() - slot.cell_vars;
            shift -= n;
            advice.put(
                slot,
                Part::Lookup {
                    addresses: Arc::clone(&indices),
                    shift,
                },
            )?;
        }
        Ok(())
    }

    fn batch<F: ProofField>(
        queries: &[(&Self, Read<'_, F>)],
        tables: &[Table<'_>],
        io: &mut Session<'_, '_, F>,
    ) -> Result<()> {
        let [first, ..] = queries else {
            return Err("no lookup requests");
        };
        let n = first.0.address_vars;
        let size = 1 << n;
        if tables.iter().any(|t| t.len() != size)
            || queries.iter().any(|(_, q)| q.output.len() != tables.len())
        {
            return Err("inconsistent lookup shapes");
        }
        io.pow(tables.len() + 1 + queries.len())?;
        let mix = io.challenge(tables.len() + 1)?;
        let batch = io.challenge(queries.len())?;
        let table = |i| {
            mix[0] * F::from_u64(i as u64)
                + dot(
                    tables.iter().map(|col| F::from_i32(col.at(i))),
                    mix[1..].iter().copied(),
                )
        };
        let sum = queries
            .iter()
            .zip(&batch)
            .map(|((_, q), &w)| {
                w * (mix[0] * q.address + dot(q.output.iter().copied(), mix[1..].iter().copied()))
            })
            .sum();
        let stmt = instance(sum, n, 2);
        let end = io.reduce(&stmt, |t| {
            let mut reads = BTreeMap::new();
            for ((_, q), &w) in queries.iter().zip(&batch) {
                for (&address, cell) in q.witness().iter().zip(weights(&q.cell)) {
                    *reads.entry(address as usize).or_insert_with(F::zero) += w * cell;
                }
            }
            let mut fold = TableFold::new(size, reads, table);
            prove_single(&mut fold, &stmt, t).map_err(|_| "Sumcheck failed")
        })?;
        let address = first.0.link(queries, &end.challenges, &batch, io)?;
        if !io.proving() {
            let (rows, cols) = end.challenges.split_at(n / 2);
            let rows = weights(rows);
            let cols = weights(cols);
            let index = end.challenges.iter().fold(F::zero(), |i, &bit| i + i + bit);
            let mut evaluated = mix[0] * index;
            for (table, &coefficient) in tables.iter().zip(&mix[1..]) {
                evaluated += coefficient * table.evaluate(&rows, &cols);
            }
            if end.output_claim != address * evaluated {
                return Err("invalid table evaluation");
            }
        }
        Ok(())
    }

    fn parts_fit(&self, other: &Self) -> bool {
        self.address_vars == other.address_vars
    }

    fn link<F: ProofField>(
        &self,
        queries: &[(&Self, Read<'_, F>)],
        address: &[F],
        mix: &[F],
        io: &mut Session<'_, '_, F>,
    ) -> Result<F> {
        let parts = self.parts(address);
        let mut columns = Vec::new();
        let [value] = io.values(|| {
            let mut shift = self.address_vars;
            let digits: Vec<_> = parts
                .iter()
                .map(|p| {
                    shift -= p.len();
                    (shift, weights(p))
                })
                .collect();
            let (sums, prepared): (Vec<_>, Vec<_>) = queries
                .par_iter()
                .zip(mix)
                .map(|((_, q), &w)| {
                    let addresses = q.witness();
                    let cols: Vec<_> = digits
                        .iter()
                        .map(|(shift, weights)| Column::lookup(addresses, *shift, weights.clone()))
                        .chain(std::iter::once(Column::equality(&q.cell)))
                        .collect();
                    let sum = (0..addresses.len())
                        .map(|i| cols.iter().map(|c| c.at(i)).product::<F>())
                        .sum::<F>();
                    (w * sum, cols)
                })
                .unzip();
            columns = prepared;
            [sums.into_iter().sum()]
        })?;
        // The table Sumcheck needs this same weighted sum of address evaluations.
        let endpoints = std::array::from_fn::<_, 8, _>(|i| i.min(parts.len() - 1));
        let (end, evaluations) = io.fold_weighted(
            instance(value, queries[0].1.cell.len(), parts.len() + 1),
            mix,
            || columns,
            endpoints,
            parts.len(),
            |v: &[F]| v.iter().copied().product(),
        )?;
        let half = F::from_u64(2).inverse().expect("odd field");
        let ones: Vec<_> = parts
            .iter()
            .map(|p| (vec![half; p.len()], (0..p.len()).map(|_| half).product()))
            .collect();
        let mut actual = F::zero();
        for (((lookup, q), &weight), values) in queries.iter().zip(mix).zip(evaluations) {
            let mut term = weight * eq(&q.cell, &end.challenges);
            for (((&slot, part), (selector, value)), evaluation) in
                lookup.cols.iter().zip(&parts).zip(&ones).zip(values)
            {
                term *= evaluation;
                io.openings.push(Opening::new(
                    slot,
                    Claim {
                        point: [*part, &end.challenges].concat(),
                        value: evaluation,
                    },
                ));
                let point = [selector.as_slice(), &q.cell].concat();
                io.openings.push(Opening::new(
                    slot,
                    Claim {
                        point,
                        value: *value,
                    },
                ));
            }
            actual += term;
        }
        if !io.proving() && actual != end.output_claim {
            return Err("invalid lookup address link");
        }
        Ok(value)
    }
}

struct TableFold<F, T> {
    table: T,
    sparse: BTreeMap<usize, F>,
    size: usize,
    rounds: usize,
    prefix: Vec<F>,
    dense: Option<[Vec<F>; 2]>,
}
impl<F: ProofField, T: Fn(usize) -> F> TableFold<F, T> {
    fn new(size: usize, sparse: BTreeMap<usize, F>, table: T) -> Self {
        let mut fold = Self {
            table,
            sparse,
            size,
            rounds: size.ilog2() as usize,
            prefix: vec![F::one()],
            dense: None,
        };
        if fold.sparse.len() >= size / 8 {
            fold.materialize();
        }
        fold
    }

    fn at(&self, i: usize) -> F {
        self.prefix
            .iter()
            .enumerate()
            .map(|(block, &v)| v * (self.table)(block * self.size + i))
            .sum()
    }

    fn materialize(&mut self) {
        let mut reads = vec![F::zero(); self.size];
        for (i, value) in std::mem::take(&mut self.sparse) {
            reads[i] = value;
        }
        self.dense = Some([reads, (0..self.size).map(|i| self.at(i)).collect()]);
        self.prefix.clear();
    }

    fn bind(&mut self, r: F) {
        self.size /= 2;
        if let Some(columns) = &mut self.dense {
            for col in columns {
                bind_halves(col, r);
            }
            return;
        }
        for (i, value) in std::mem::take(&mut self.sparse) {
            let w = if i < self.size { F::one() - r } else { r };
            *self.sparse.entry(i % self.size).or_insert_with(F::zero) += w * value;
        }
        self.prefix = self
            .prefix
            .iter()
            .flat_map(|&v| [v * (F::one() - r), v * r])
            .collect();
        if self.sparse.len() >= self.size.div_ceil(4) {
            self.materialize();
        }
    }
}

impl<F: ProofField, T: Fn(usize) -> F> ProveRounds<F> for TableFold<F, T> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }
    fn prove_round(
        &mut self,
        r: Option<F>,
        _: usize,
        _: F,
    ) -> std::result::Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = r {
            self.bind(r);
        }
        let n = self.size / 2;
        let mut samples = [F::zero(); 3];
        let add = |samples: &mut [F; 3], a: F, b: F, l: F, r: F| {
            samples[0] += a * l;
            samples[1] += b * r;
            samples[2] += (b + b - a) * (r + r - l);
        };
        if let Some([a, t]) = &self.dense {
            if n >= PARALLEL_ROUND_SIZE {
                let zeros = || [F::zero(); 3];
                samples = (0..n)
                    .into_par_iter()
                    .with_min_len(ROUND_CHUNK_SIZE)
                    .fold(zeros, |mut acc, i| {
                        add(&mut acc, a[i], a[i + n], t[i], t[i + n]);
                        acc
                    })
                    .reduce(zeros, |left, right| {
                        std::array::from_fn(|i| left[i] + right[i])
                    });
            } else {
                for i in 0..n {
                    add(&mut samples, a[i], a[i + n], t[i], t[i + n]);
                }
            }
        } else {
            let mut pairs = BTreeMap::<_, [F; 2]>::new();
            for (&i, &w) in &self.sparse {
                pairs.entry(i % n).or_insert([F::zero(); 2])[i / n] = w;
            }
            for (i, [a, b]) in pairs {
                add(&mut samples, a, b, self.at(i), self.at(i + n));
            }
        }
        Ok(UnivariatePoly::from_evals(&samples))
    }
    fn finish_rounds(&mut self, _: F) -> std::result::Result<(), SumcheckError<F>> {
        Ok(())
    }
}
