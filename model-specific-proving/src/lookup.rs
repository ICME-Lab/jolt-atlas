use std::sync::Arc;

use crate::{
    advice::{Advice, Requests, Slot},
    channel::{Channel, Result},
    field::F,
    fold::{agreement, basis, Column},
    tensor::Shape,
};
use jolt_field::{One, Ring, Zero};

#[derive(Clone, Copy)]
pub(crate) enum Table<'a> {
    Values(&'a [i32]),
    Pair(&'a [i32], &'a [i32], i32, u128),
}

impl Table<'_> {
    pub fn count(self) -> usize {
        match self {
            Self::Values(x) | Self::Pair(x, ..) => x.len(),
        }
    }

    fn same(self, other: Self) -> bool {
        match (self, other) {
            (Self::Values(a), Self::Values(b)) => std::ptr::eq(a, b),
            (Self::Pair(a, b, c, d), Self::Pair(e, f, g, h)) => {
                std::ptr::eq(a, e) && std::ptr::eq(b, f) && c == g && d == h
            }
            _ => false,
        }
    }

    pub(crate) fn values(self, i: usize) -> (F, F) {
        match self {
            Self::Values(a) => (F::from_i32(a[i]), F::zero()),
            Self::Pair(prob, sums, limit, scale) => {
                let previous = if i == 0 { limit + 1 } else { sums[i - 1] };
                (
                    F::from_i32(prob[i]),
                    F::from_i32(sums[i]) + F::from_u128(scale) * F::from_i32(previous),
                )
            }
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct Lookup {
    pub addresses: Option<Arc<[u32]>>,
    digits: Vec<(Slot, usize, usize)>,
    vars: usize,
}

impl Lookup {
    pub fn record(
        advice: &mut Advice,
        shape: Shape,
        size: usize,
        addresses: Option<Vec<u32>>,
    ) -> Result<Self> {
        if !size.is_power_of_two() || size < 2 {
            return Err("invalid lookup size");
        }
        let n = size.ilog2() as usize;
        let step = if (17..=24).contains(&n) { 3 } else { 4 };
        let domain = Shape::new(shape.rows.next_power_of_two(), shape.stride())?;
        let mut digits = Vec::new();
        for shift in (0..n).step_by(step) {
            let width = (n - shift).min(step);
            let positions = 1 << width;
            let slot = advice.planes(domain, positions as u8, |i| {
                let index = addresses.as_ref().unwrap()[i] as usize;
                1 << ((index >> shift) & (positions - 1))
            })?;
            digits.push((slot, shift, width));
        }
        Ok(Self {
            addresses: addresses.map(Arc::from),
            digits,
            vars: domain.bits(),
        })
    }
}

struct Read {
    lookup: Lookup,
    point: Vec<F>,
    address: F,
    outputs: (F, F),
}

#[derive(Default)]
pub(crate) struct Lookups<'a>(Vec<(Table<'a>, usize, Vec<Read>)>);

impl<'a> Lookups<'a> {
    pub fn read(
        &mut self,
        lookup: &Lookup,
        table: Table<'a>,
        point: Vec<F>,
        address: F,
        outputs: (F, F),
    ) {
        let i = self
            .0
            .iter()
            .position(|(t, n, _)| t.same(table) && *n == lookup.vars);
        let reads = match i {
            Some(i) => &mut self.0[i].2,
            None => {
                self.0.push((table, lookup.vars, Vec::new()));
                &mut self.0.last_mut().unwrap().2
            }
        };
        reads.push(Read {
            lookup: lookup.clone(),
            point,
            address,
            outputs,
        });
    }

    pub fn finish(self, io: &mut Channel<'_, '_>, requests: &mut Requests) -> Result<()> {
        for (table, vars, reads) in self.0 {
            let len = 1 << vars;
            let parts = (64 * 1024 * 1024 / (len * reads[0].lookup.digits.len() * 4)).clamp(1, 16);
            Self::batch(io, requests, table, vars, &reads, parts)?;
        }
        Ok(())
    }

    fn batch(
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        table: Table<'a>,
        vars: usize,
        reads: &[Read],
        batch: usize,
    ) -> Result<()> {
        let n = table.count();
        let b = n.ilog2() as usize;
        let draw = io.sample(reads.len() + 3, reads.len() + 3)?;
        let (a, mix) = draw.split_at(3);
        let value = |i| {
            let (p, s) = table.values(i);
            a[0] * F::from_u64(i as u64) + a[1] * p + a[2] * s
        };
        let start = reads
            .iter()
            .zip(mix)
            .map(|(q, &m)| m * (a[0] * q.address + a[1] * q.outputs.0 + a[2] * q.outputs.1))
            .sum();
        let (at, [histogram]) = io.fold(
            start,
            b,
            2,
            || {
                let mut bins = vec![F::zero(); n];
                for (r, &m) in reads.iter().zip(mix) {
                    let codes = r.lookup.addresses.as_ref().unwrap();
                    for (&addr, w) in codes.iter().zip(basis(&r.point)) {
                        bins[addr as usize] += m * w;
                    }
                }
                [Column::table(bins, n, 1), Column::source(n, value)]
            },
            [0],
            |v| v[0] * v[1],
        )?;
        let table_eval: F = basis(&at.challenges)
            .into_iter()
            .enumerate()
            .map(|(i, w)| w * value(i))
            .sum();
        if at.output_claim != histogram * table_eval {
            return Err("invalid lookup table evaluation");
        }

        let parts = if reads.len() <= batch {
            vec![histogram]
        } else {
            let count = reads.len().div_ceil(batch);
            let partials = io.fields(count, || {
                use rayon::prelude::*;
                let selection = basis(&at.challenges);
                reads
                    .par_chunks(batch)
                    .zip(mix.par_chunks(batch))
                    .map(|(reads, mix)| {
                        reads
                            .iter()
                            .zip(mix)
                            .map(|(r, &scale)| {
                                let codes = r.lookup.addresses.as_ref().unwrap();
                                scale
                                    * codes
                                        .iter()
                                        .zip(basis(&r.point))
                                        .map(|(&code, w)| selection[code as usize] * w)
                                        .sum::<F>()
                            })
                            .sum()
                    })
                    .collect()
            })?;
            if partials.iter().copied().sum::<F>() != histogram {
                return Err("invalid lookup batch sum");
            }
            partials
        };
        let weights: Vec<_> = reads[0]
            .lookup
            .digits
            .iter()
            .map(|&(_, shift, width)| basis(&at.challenges[b - shift - width..b - shift]))
            .collect();
        for ((read, mix), partial) in reads.chunks(batch).zip(mix.chunks(batch)).zip(parts) {
            Self::link(io, requests, vars, read, mix, &weights, partial)?;
        }
        Ok(())
    }

    fn link(
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        vars: usize,
        reads: &[Read],
        mix: &[F],
        weights: &[Vec<F>],
        partial: F,
    ) -> Result<()> {
        let digits = &reads[0].lookup.digits;
        let len = 1 << vars;
        let size = digits.len() + 1;
        let formula = |v: &[F]| {
            v.chunks_exact(size)
                .zip(mix)
                .map(|(row, &scale)| scale * row.iter().product::<F>())
                .sum()
        };
        let terminals: Vec<_> = (0..reads.len() * size)
            .filter(|i| i % size < digits.len())
            .collect();
        let (end, results) = io.fold_many(
            partial,
            vars,
            size,
            || {
                let mut cols = Vec::with_capacity(reads.len() * size);
                for r in reads {
                    let codes = r.lookup.addresses.as_ref().unwrap();
                    for (&(_, shift, width), weights) in digits.iter().zip(weights) {
                        let mask = (1usize << width) - 1;
                        cols.push(Column::source(len, move |i| {
                            weights[(codes[i] as usize >> shift) & mask]
                        }));
                    }
                    cols.push(Column::table(basis(&r.point), len, 1));
                }
                cols
            },
            &terminals,
            formula,
        )?;
        let mut terminal = Vec::with_capacity(reads.len() * size);
        for (r, values) in reads.iter().zip(results.chunks_exact(digits.len())) {
            terminal.extend_from_slice(values);
            terminal.push(agreement(&r.point, &end.challenges));
            for ((&(slot, _, width), weight), &v) in r.lookup.digits.iter().zip(weights).zip(values)
            {
                requests.weighted(slot, &end.challenges, v, weight.clone());
                requests.weighted(slot, &end.challenges, F::one(), vec![F::one(); 1 << width]);
            }
        }
        if formula(&terminal) != end.output_claim {
            return Err("invalid lookup terminal evaluation");
        }
        Ok(())
    }
}
